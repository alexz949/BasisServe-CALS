"""Checks of the SVD modes of the ShadowKV TP port (basisserve/core/shadowkv_upstream_tp.py: centralized,
request_parallel, replicated). Run under torchrun, one process per rank.

--part layout   No model. Every rank builds the same synthetic pre-RoPE Keys from a seed (one matrix per request and
                layer, [T, 8*128], different for every request, layer and head), passes its own heads to
                tp_get_svd, and then checks that U @ SV^T of every (request, layer, local head) reproduces that
                head's Keys. Two cases per mode: low-rank Keys (rank 64 < 160: exact up to BF16 rounding, so a
                factor written to the wrong request, layer, head or prefill-group offset fails; controls show
                that the wrong request / head / layer do fail), and Keys with a decaying full spectrum, where the
                rank-160 reconstructions of the modes are compared with each other (U and SV themselves are not
                compared: signs and bases are not unique). Batches of 8 (two prefill groups of 4) and 6 (a group
                of 4 and a tail of 2), three layers. --backend gloo runs on CPU tensors (more ranks than GPUs).
--part model    The real model (TP > 1), 8192-token prompts, one prefill group of 4. The model is built once with the
                replicated mode's extra K rows and prefilled in each mode. Reports: the replicated full-Key
                projection against the Keys the owning ranks compute (other ranks' heads; the local slice must be
                bitwise equal); each mode's rank-160 reconstruction error against the true local Keys and its
                difference from centralized; and the prefill logits across modes.

Usage: torchrun --nproc-per-node=W benchmarks/system/check_shadowkv_svd_modes.py --part layout|model --output DIR
"""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/system"))
HEADS, HEAD_DIM, RANK = 8, 128, 160


def synthetic(request, layer, tokens, low_rank, device):
    """[T, 8*128] BF16 Keys of one request and layer, the same on every rank."""
    generator = torch.Generator().manual_seed(100003 * layer + 101 * request + (0 if low_rank else 7))
    width = HEADS * HEAD_DIM
    if low_rank:
        keys = torch.randn(tokens, 64, generator=generator) @ torch.randn(64, width, generator=generator) / 8
    else:
        spectrum = (1 + torch.arange(width)).float().pow(-0.7)
        keys = (torch.randn(tokens, width, generator=generator) * spectrum) @ torch.randn(
            width, width, generator=generator) / width ** 0.5
    return keys.to(torch.bfloat16).to(device)


def relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def layout(args, rank, world, device):
    from basisserve.core import shadowkv_upstream_tp as port

    local, layers, tokens = HEADS // world, 3, args.tokens
    mine = slice(rank * local * HEAD_DIM, (rank + 1) * local * HEAD_DIM)
    report, reconstructions = {}, {}
    for low_rank in (True, False):
        for batch, groups in ((8, (4, 4)), (6, (4, 2))):
            for mode in port.SVD_MODES:
                port.SVD_MODE = mode
                port.reset_counts()
                cache = SimpleNamespace(batch_size=batch, num_layers=layers, num_key_value_heads=local,
                                        head_dim=HEAD_DIM, rank=RANK, dtype=torch.bfloat16, prefilled_batch=0,
                                        full_k=None)
                for size in groups:  # one prefill group after another, layer by layer, as the forward does
                    for layer in range(layers):
                        full = torch.stack([synthetic(cache.prefilled_batch + j, layer, tokens, low_rank, device)
                                            for j in range(size)])
                        if mode == "replicated":
                            cache.full_k = full
                        port.tp_get_svd(cache, full[..., mine].contiguous(), layer)
                    cache.prefilled_batch += size
                worst, controls, kept = 0.0, [], {}
                for request in range(batch):
                    for layer in range(layers):
                        keys = synthetic(request, layer, tokens, low_rank, device)
                        u = cache.U[layer, request].to(device).float()
                        for head in range(local):
                            position = rank * local + head
                            rebuilt = u @ cache.SV[layer, request, head].to(device).float().T
                            truth = keys[:, position * HEAD_DIM:(position + 1) * HEAD_DIM]
                            worst = max(worst, relative(rebuilt, truth))
                            kept[(request, layer, head)] = rebuilt
                            if low_rank:  # what a misplaced factor would look like
                                other = (position + 1) % HEADS
                                controls += [
                                    relative(rebuilt, synthetic((request + 1) % batch, layer, tokens, True, device)[
                                        :, position * HEAD_DIM:(position + 1) * HEAD_DIM]),
                                    relative(rebuilt, keys[:, other * HEAD_DIM:(other + 1) * HEAD_DIM]),
                                    relative(rebuilt, synthetic(request, (layer + 1) % layers, tokens, True, device)[
                                        :, position * HEAD_DIM:(position + 1) * HEAD_DIM])]
                name = f"{'low_rank' if low_rank else 'full_spectrum'}_b{batch}_{mode}"
                report[name] = dict(reconstruction_error_max=worst,
                                    counts={label: dict(row) for label, row in port.COUNTS.items()})
                if low_rank:
                    report[name]["wrong_place_error_min"] = min(controls)
                else:
                    reference = reconstructions.setdefault(batch, kept)
                    report[name]["difference_from_centralized_max"] = max(
                        relative(kept[key], reference[key]) for key in kept)
    return report


def model(args, rank, world, device):
    from safetensors.torch import load_file
    import run_v2q8_exp_smoke as smoke
    from benchmarks.system import audit_tp1_request as frozen
    from basisserve.core import shadowkv_upstream_tp as port

    batch, tokens = 4, args.tokens
    prompts = load_file(str(smoke.INPUTS[4]))["cohort_0"][:batch, :tokens].long().to(device)
    assert prompts.shape == (batch, tokens)
    llama = port.build(frozen.MODEL, batch=batch, max_length=tokens + 128 + 8, device=device, svd_mode="replicated")
    local = llama.num_key_value_heads
    mine = slice(rank * local * HEAD_DIM, (rank + 1) * local * HEAD_DIM)
    report, reference, reference_logits = {}, None, None
    for mode in port.SVD_MODES:
        port.SVD_MODE = mode
        port.reset_counts()
        cache = port.fresh_cache(llama, length=tokens, output_tokens=128)
        truth, projection, original = [], [], cache.get_svd

        def get_svd(new_k_cache, layer_idx):
            truth.append(new_k_cache.clone())
            if port.SVD_MODE == "replicated":
                parts = [torch.empty_like(new_k_cache) for _ in range(world)]
                dist.all_gather(parts, new_k_cache.contiguous())
                owned, full = torch.cat(parts, dim=-1), cache.full_k
                assert torch.equal(full[..., mine], new_k_cache)  # the local slice is the upstream product itself
                other = torch.ones(full.shape[-1], dtype=torch.bool, device=device)
                other[mine] = False
                a, b = full[..., other].float(), owned[..., other].float()
                projection.append(dict(layer=layer_idx, max_abs_diff=float((a - b).abs().max()),
                                       mean_abs_diff=float((a - b).abs().mean()), mean_abs_key=float(b.abs().mean()),
                                       relative_diff=relative(a, b), bitwise_equal_fraction=float((a == b).float().mean())))
            return original(new_k_cache, layer_idx)

        cache.get_svd = get_svd
        logits = llama.batch_prefill(prompts)[:, -1]
        errors, rebuilt = [], []
        for layer in range(cache.num_layers):
            keys = truth[layer].float().view(batch, tokens, local, HEAD_DIM).transpose(1, 2)  # [B, heads, T, 128]
            # upstream layout: U [B, T, 160], SV [B, heads, 128, 160]
            approx = torch.matmul(cache.U[layer].to(device).float()[:, None],
                                  cache.SV[layer].to(device).float().transpose(-1, -2))
            errors.append(relative(approx, keys))
            rebuilt.append(approx.to(torch.bfloat16))
        row = dict(reconstruction_error_mean=sum(errors) / len(errors), reconstruction_error_max=max(errors),
                   counts={label: dict(r) for label, r in port.COUNTS.items()},
                   prefill_argmax=logits.argmax(-1).tolist())
        if reference is None:
            reference, reference_logits = rebuilt, logits
        else:
            differences = [relative(a, b) for a, b in zip(rebuilt, reference)]
            row.update(difference_from_centralized_mean=sum(differences) / len(differences),
                       difference_from_centralized_max=max(differences),
                       prefill_logits_max_abs_diff_from_centralized=float((logits - reference_logits).abs().max()))
        if projection:
            row["full_key_projection_vs_owning_rank"] = dict(
                max_abs_diff=max(p["max_abs_diff"] for p in projection),
                mean_abs_diff=sum(p["mean_abs_diff"] for p in projection) / len(projection),
                mean_abs_key=sum(p["mean_abs_key"] for p in projection) / len(projection),
                relative_diff_max=max(p["relative_diff"] for p in projection),
                bitwise_equal_fraction_min=min(p["bitwise_equal_fraction"] for p in projection),
                local_slice_bitwise_equal=True, layers=len(projection))
        report[mode] = row
        del rebuilt
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("layout", "model"), required=True)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.tokens = args.tokens or (1536 if args.part == "layout" else 8192)
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
        device = torch.device("cuda", local_rank)
    else:
        assert args.part == "layout"
        torch.set_num_threads(8)
        dist.init_process_group("gloo")
        device = torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260924)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        report = (layout if args.part == "layout" else model)(args, rank, world, device)
    (args.output / f"rank_{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    if rank == 0:
        ranks = [json.loads((args.output / f"rank_{r}.json").read_text()) for r in range(world)]
        result = dict(status="complete", part=args.part, backend=args.backend, tp=world, tokens=args.tokens,
                      kv_heads_per_rank=HEADS // world, ranks=ranks)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(part=args.part, tp=world, status="complete")), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
