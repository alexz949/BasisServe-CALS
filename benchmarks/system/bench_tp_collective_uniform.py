"""TP-generic attention-output collective benchmark for uniform C1 checkpoints.

Five arms share one synthetic query-head output vector and real C1 factors:

    dense_ar     full-width row-parallel ``o_proj`` then AllReduce(hidden)
    lr_ar_wire   shared-basis low-rank AllReduce at the wire-matched rank
    lr_ar_cap    shared-basis low-rank AllReduce at the capacity-matched rank
    c1_ar        C1 source-private coordinates reduced over the global width
    basiskv_ag   C1 source-private coordinates gathered at the local width

``c1_ar`` and ``basiskv_ag`` compute the identical function, so their difference
isolates the collective boundary.  ``lr_ar_cap`` reduces the same number of
coordinates as ``c1_ar``, so their difference isolates shared versus
source-private bases.  Ring-byte formulas are emitted per row so every
normalized traffic claim in the paper can be re-derived from the CSV.

Geometry is derived from the model config and the process-group size; nothing
is specialized to one tensor-parallel width.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file

from basisserve.core.attention_tp_layout import AttentionTPLayout
from basisserve.core.tp_output import factorize_row_parallel_weight_distributed
from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.ragged_allgather import StaticRaggedPlan
from benchmarks.system.common import metadata, save
from evaluation.benchmark_tp4_interconnect import _critical_cuda_timings, _summary


ARMS = ("dense_ar", "lr_ar_wire", "lr_ar_cap", "c1_ar", "basiskv_ag")


def quantile(values, fraction):
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return ordered[position]


def timed(function, device, *, warmup, iterations):
    raw = _critical_cuda_timings(function, warmup=warmup, iterations=iterations, device=device)
    return dict(
        **_summary(raw),
        # The paper protocol reports median with p10/p90, which _summary omits.
        median_ms=statistics.median(raw),
        p10_ms=quantile(raw, 0.10),
        p90_ms=quantile(raw, 0.90),
        stddev_ms=statistics.pstdev(raw),
        warmup=warmup,
        iterations=iterations,
        aggregation="per-iteration maximum across all TP ranks",
        timing="CUDA events on the current stream; no allocation inside timed functions",
    )


def ring_bytes(*, collective, world_size, elements, element_size):
    """Bytes leaving one rank for a ring implementation of one collective."""

    if world_size <= 1:
        return 0.0
    if collective == "all_reduce":
        return 2.0 * (world_size - 1) / world_size * elements * element_size
    if collective == "all_gather":
        # ``elements`` is the local block; each rank forwards it to P-1 peers.
        return float((world_size - 1) * elements * element_size)
    raise ValueError(collective)


class OutputBlock:
    """One layer's attention-output block under every arm, at one batch size."""

    def __init__(self, *, encoder, decoder, wo, batch, layout, group_rank, communicator, lr):
        device = encoder.device
        self.batch = batch
        self.layout = layout
        self.r = int(encoder.shape[-1])
        local_kv = layout.local_kv_heads
        per_kv = layout.num_attention_heads // layout.num_key_value_heads
        self.local_q = layout.local_query_heads
        self.local_width = self.local_q * self.r
        self.global_width = layout.num_attention_heads * self.r
        hidden = layout.hidden_size
        world = layout.tp_size

        owned = encoder[group_rank * local_kv : (group_rank + 1) * local_kv]
        self.encoder = owned.repeat_interleave(per_kv, 0).transpose(1, 2).contiguous()
        self.decoder = decoder.reshape(self.global_width, hidden).contiguous()
        local_in = layout.local_o_input_width
        self.wo = wo[:, group_rank * local_in : (group_rank + 1) * local_in].T.contiguous()

        # Synthetic query-head outputs with real factors and real o_proj: this is a
        # communication-block microbenchmark, not a model-quality measurement.
        self.input = torch.randn(batch, local_in, device=device, dtype=torch.bfloat16)
        self.head_input = self.input.view(batch, self.local_q, layout.head_dim).permute(1, 2, 0)
        self.dense = torch.empty(batch, hidden, device=device, dtype=torch.bfloat16)
        self.output = torch.empty_like(self.dense)

        self.ar = torch.empty(self.global_width, batch, device=device, dtype=torch.bfloat16)
        self.ar_slot = self.ar[group_rank * self.local_width : (group_rank + 1) * self.local_width]

        plan = StaticRaggedPlan.from_source_widths((self.local_width,) * world)
        self.ag = communicator.prepare_uniform(
            plan, tokens=batch, dtype=torch.bfloat16, backend="uniform_nccl"
        )
        self.ag_slot = self.ag.local_feature_major_view_fast().view(self.local_q, self.r, batch)
        self.ag_slot.zero_()
        self.ag_full = self.ag.gather_inplace_fast()

        self.lr = {}
        for name, factors in lr.items():
            width = int(factors.output_basis.shape[1])
            self.lr[name] = dict(
                width=width,
                # local_input_factor is [local_d_in, rank]; output_basis is [d_out, rank].
                a=factors.local_input_factor.contiguous(),
                r=factors.output_basis.T.contiguous(),
                latent=torch.empty(batch, width, device=device, dtype=torch.bfloat16),
            )

    # dense row-parallel: full-width projection then full-width AllReduce
    def dense_ar_projection(self):
        torch.mm(self.input, self.wo, out=self.dense)

    def dense_ar_collective(self):
        dist.all_reduce(self.dense)

    def dense_ar_reconstruction(self):
        self.output.copy_(self.dense)

    # shared-basis low-rank AllReduce
    def _lr_projection(self, name):
        state = self.lr[name]
        torch.mm(self.input, state["a"], out=state["latent"])

    def _lr_collective(self, name):
        dist.all_reduce(self.lr[name]["latent"])

    def _lr_reconstruction(self, name):
        state = self.lr[name]
        torch.mm(state["latent"], state["r"], out=self.output)

    def lr_ar_wire_projection(self):
        self._lr_projection("wire")

    def lr_ar_wire_collective(self):
        self._lr_collective("wire")

    def lr_ar_wire_reconstruction(self):
        self._lr_reconstruction("wire")

    def lr_ar_cap_projection(self):
        self._lr_projection("cap")

    def lr_ar_cap_collective(self):
        self._lr_collective("cap")

    def lr_ar_cap_reconstruction(self):
        self._lr_reconstruction("cap")

    # C1 source-private coordinates, reduced over the replicated global width
    def c1_ar_projection(self):
        self.ar.zero_()
        torch.bmm(self.encoder, self.head_input, out=self.ar_slot.view(self.local_q, self.r, self.batch))

    def c1_ar_collective(self):
        dist.all_reduce(self.ar)

    def c1_ar_reconstruction(self):
        torch.mm(self.ar.T, self.decoder, out=self.output)

    # C1 source-private coordinates, gathered at the local width
    def basiskv_ag_projection(self):
        torch.bmm(self.encoder, self.head_input, out=self.ag_slot)

    def basiskv_ag_collective(self):
        self.ag.gather_inplace_fast()

    def basiskv_ag_reconstruction(self):
        torch.mm(self.ag_full.T, self.decoder, out=self.output)

    def total(self, arm):
        def run():
            getattr(self, f"{arm}_projection")()
            getattr(self, f"{arm}_collective")()
            getattr(self, f"{arm}_reconstruction")()

        return run

    def payload(self, arm, world_size):
        element = torch.empty((), dtype=torch.bfloat16).element_size()
        if arm == "dense_ar":
            width, collective = self.layout.hidden_size, "all_reduce"
        elif arm == "c1_ar":
            width, collective = self.global_width, "all_reduce"
        elif arm == "basiskv_ag":
            width, collective = self.local_width, "all_gather"
        else:
            width, collective = self.lr[arm.removeprefix("lr_ar_")]["width"], "all_reduce"
        elements = self.batch * width
        return dict(
            collective_kind=collective,
            communicated_width=width,
            communicated_elements_per_rank=elements,
            payload_bytes_per_rank=elements * element,
            ring_bytes_per_rank=ring_bytes(
                collective=collective,
                world_size=world_size,
                elements=elements,
                element_size=element,
            ),
        )

    @torch.inference_mode()
    def validate(self, world_size):
        """Correctness gates required before any timing is reported."""

        self.total("c1_ar")()
        reduced_coordinates = self.ar.clone()
        c1_output = self.output.clone()
        self.total("basiskv_ag")()
        assert torch.equal(reduced_coordinates, self.ag_full), "AR/AG coordinates differ"
        torch.testing.assert_close(c1_output, self.output, rtol=0, atol=0)

        # Local BF16 encoding against the same contraction accumulated in FP32.
        reference = torch.bmm(self.encoder.float(), self.head_input.float()).bfloat16()
        torch.testing.assert_close(self.ag_slot, reference, rtol=0.01, atol=0.003)

        # Every AllReduce arm against an explicit all_gather-and-sum reference.
        # NCCL accumulates the P partials in BF16 in an unspecified tree order, so
        # the FP32 reference may differ by the BF16 summation error bound
        # (P - 1) * u * sum|partial| with u = 2**-8.  A fixed atol would either
        # reject near-cancelling elements or hide real reduction bugs, so the
        # bound is derived from the gathered partials themselves.
        checks = {}
        unit_roundoff = 2.0 ** -8
        for arm in ("dense_ar", "lr_ar_wire", "lr_ar_cap"):
            getattr(self, f"{arm}_projection")()
            buffer = self.dense if arm == "dense_ar" else self.lr[arm.removeprefix("lr_ar_")]["latent"]
            partials = [torch.empty_like(buffer) for _ in range(world_size)]
            dist.all_gather(partials, buffer)
            getattr(self, f"{arm}_collective")()
            stacked = torch.stack(partials).float()
            expected = stacked.sum(0)
            bound = 2.0 * (world_size - 1) * unit_roundoff * stacked.abs().sum(0)
            deviation = (buffer.float() - expected).abs()
            worst = float((deviation / bound.clamp_min(1e-6)).max())
            assert worst <= 1.0, (
                f"{arm} reduction exceeds the BF16 summation bound: "
                f"worst normalized deviation {worst:.3f}, "
                f"max absolute deviation {float(deviation.max()):.3e}"
            )
            checks[f"{arm}_allgather_sum_reference"] = True
            checks[f"{arm}_worst_normalized_deviation"] = worst
        assert torch.isfinite(self.output).all() and torch.isfinite(self.dense).all()
        return dict(
            c1_ar_ag_coordinates_bitwise_equal=True,
            c1_ar_ag_output_bitwise_equal=True,
            local_encoding_fp32_reference=True,
            **checks,
        )


def resolve_model(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, local_files_only=True))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="uniform C1 manifest.json")
    p.add_argument("--model", required=True, help="base model repo id resolved from the HF cache")
    p.add_argument("--output", type=Path, default=Path("results/systems"))
    p.add_argument("--batches", default="1,2,4,8,16,32")
    p.add_argument("--layers", default="", help="comma-separated layers; default is one mid layer")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    if a.smoke:
        a.warmup, a.iters, a.repeats, a.batches = 5, 20, 1, "1,8"

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl")
    assert "NCCL_ALGO" not in os.environ and "NCCL_PROTO" not in os.environ
    world_size = dist.get_world_size()
    group_rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(2026 + group_rank)

    manifest = json.loads(a.manifest.read_text())
    assert manifest["compression"]["allocation"] == "uniform_per_layer_per_head", (
        f"this harness requires a uniform checkpoint, got "
        f"{manifest['compression']['allocation']!r}"
    )
    bank = a.manifest.parent
    model = resolve_model(a.model)
    config = json.loads((model / "config.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    layout_of = lambda: AttentionTPLayout(
        hidden_size=config["hidden_size"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        tp_size=world_size,
    )
    layout = layout_of()
    assert layout.kv_partition_mode == "sharded" and layout.kv_replication_factor == 1, (
        "this suite requires sharded KV ownership without replication"
    )

    layers = (
        [int(v) for v in a.layers.split(",") if v]
        if a.layers
        else [len(manifest["layers"]) // 2]
    )
    batches = [int(v) for v in a.batches.split(",") if v]
    communicator = FeatureRaggedCommunicator.from_distributed(device=device)
    run_metadata = metadata()
    run_metadata["allgather_nccl_version"] = communicator.nccl_version
    run_metadata["tp_layout"] = dict(
        tp_size=world_size,
        local_query_heads=layout.local_query_heads,
        local_kv_heads=layout.local_kv_heads,
        kv_partition_mode=layout.kv_partition_mode,
        kv_replication_factor=layout.kv_replication_factor,
        local_o_input_width=layout.local_o_input_width,
        attention_type=layout.attention_type,
    )

    records = []
    for layer in layers:
        entry = manifest["layers"][layer]
        ranks = entry["ranks"]
        assert len(set(ranks)) == 1, f"layer {layer} is not uniform across heads: {ranks}"
        r = int(ranks[0])
        # Uniform manifests reference their bank relatively, e.g.
        # ``../../c1/factor-banks/R64-S6/layer_000.safetensors``.
        layer_path = (bank / entry["file"]).resolve()
        factors = load_file(str(layer_path), device=str(device))
        encoder = factors["value_coordinate_encoders"].bfloat16()
        decoder = factors["head_output_decoders"].bfloat16()
        assert encoder.shape == (layout.num_key_value_heads, layout.head_dim, r), encoder.shape
        assert decoder.shape == (layout.num_attention_heads, r, layout.hidden_size), decoder.shape

        name = f"model.layers.{layer}.self_attn.o_proj.weight"
        with safe_open(str(model / index[name]), framework="pt", device="cpu") as handle:
            wo = handle.get_tensor(name).to(device)
        local_in = layout.local_o_input_width
        local_wo = wo[:, group_rank * local_in : (group_rank + 1) * local_in].contiguous()

        # Wire-matched and capacity-matched shared-basis LR, so the strong
        # baseline is never handicapped relative to BasisKV.
        local_width = layout.local_query_heads * r
        widths = dict(
            wire=max(1, world_size * local_width // 2),
            cap=world_size * local_width,
        )
        lr = {
            key: factorize_row_parallel_weight_distributed(
                local_wo, width, factor_dtype=torch.bfloat16
            )
            for key, width in widths.items()
        }

        for batch in batches:
            try:
                block = OutputBlock(
                    encoder=encoder,
                    decoder=decoder,
                    wo=wo,
                    batch=batch,
                    layout=layout,
                    group_rank=group_rank,
                    communicator=communicator,
                    lr=lr,
                )
                audit = block.validate(world_size)
            except torch.cuda.OutOfMemoryError as error:
                records.append(dict(layer=layer, batch=batch, status="oom", error=str(error)[:300]))
                torch.cuda.empty_cache()
                continue
            for repeat in range(a.repeats):
                for arm in ARMS:
                    stages = {
                        stage: timed(
                            getattr(block, f"{arm}_{stage}"),
                            device,
                            warmup=a.warmup,
                            iterations=a.iters,
                        )
                        for stage in ("projection", "collective", "reconstruction")
                    }
                    total = timed(
                        block.total(arm), device, warmup=a.warmup, iterations=a.iters
                    )
                    records.append(
                        dict(
                            status="complete",
                            layer=layer,
                            batch=batch,
                            repeat=repeat,
                            arm=arm,
                            v_rank=r,
                            local_wire_width=local_width,
                            projection=stages["projection"],
                            collective=stages["collective"],
                            reconstruction=stages["reconstruction"],
                            total=total,
                            audit=audit,
                            **block.payload(arm, world_size),
                        )
                    )
            del block
            torch.cuda.empty_cache()

    gathered = [None] * world_size
    dist.all_gather_object(gathered, records)
    if group_rank == 0:
        a.output.mkdir(parents=True, exist_ok=True)
        payload = dict(
            metadata=run_metadata,
            model=a.model,
            manifest=str(a.manifest),
            world_size=world_size,
            batches=batches,
            layers=layers,
            per_rank_records=gathered,
        )
        save(a.output / "tp8_collective.json", payload)
        columns = [
            "model", "revision", "method", "context", "batch", "tp", "v_rank", "base_rank",
            "residual_rank", "page_size", "sparse_budget", "sink", "recent", "k_residency",
            "dtype", "gpu", "repeat", "run_id", "stage", "median_ms", "p10_ms", "p90_ms",
            "collective", "communicated_width", "communicated_elements_per_rank",
            "payload_bytes_per_rank", "ring_bytes_per_rank",
        ]
        rows = []
        gpu = torch.cuda.get_device_name(0)
        revision = manifest.get("model", {}).get("revision") or ""
        for record in records:
            if record.get("status") != "complete":
                continue
            for stage in ("projection", "collective", "reconstruction", "total"):
                summary = record[stage]
                rows.append({
                    "model": a.model, "revision": revision, "method": record["arm"],
                    "context": "n/a", "batch": record["batch"], "tp": world_size,
                    "v_rank": record["v_rank"], "base_rank": "", "residual_rank": "",
                    "page_size": "", "sparse_budget": "", "sink": "", "recent": "",
                    "k_residency": "n/a", "dtype": "bfloat16", "gpu": gpu,
                    "repeat": record["repeat"], "run_id": f"{record['arm']}_b{record['batch']}",
                    "stage": stage,
                    "median_ms": summary.get("median_ms", summary.get("p50_ms")),
                    "p10_ms": summary.get("p10_ms"), "p90_ms": summary.get("p90_ms"),
                    "collective": record["collective_kind"],
                    "communicated_width": record["communicated_width"],
                    "communicated_elements_per_rank": record["communicated_elements_per_rank"],
                    "payload_bytes_per_rank": record["payload_bytes_per_rank"],
                    "ring_bytes_per_rank": record["ring_bytes_per_rank"],
                })
        with (a.output / "tp8_collective.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {a.output/'tp8_collective.csv'}", flush=True)
    dist.barrier()
    communicator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
