"""Worker-side inspection calls for BasisServe vLLM models."""

from __future__ import annotations

from typing import Any

import torch


class BasisServeWorkerExtension:
    """Expose small model statistics through vLLM's string RPC interface."""

    def basisserve_cuda_statistics(self) -> dict[str, Any]:
        """Return device-local CUDA statistics for any vLLM model."""

        return {
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_device": torch.cuda.get_device_name(),
        }

    def basisserve_runtime_statistics(self) -> dict[str, Any]:
        return self.basisserve_cuda_statistics() | {
            "routing_statistics": self.get_model().routing_statistics()
        }

    def basisserve_c1_statistics(self) -> dict[str, Any]:
        return self.basisserve_cuda_statistics() | self.get_model().c1_statistics()

    def basisserve_tp4_statistics(self) -> dict[str, Any]:
        import socket
        from vllm.distributed import get_tp_group
        group = get_tp_group()
        communicator = group.device_communicator
        custom = communicator.ca_comm
        return self.basisserve_cuda_statistics() | {
            "hostname": socket.gethostname(), "tp_rank": group.rank_in_group,
            "tp_size": group.world_size,
            "custom_all_reduce_enabled": custom is not None and not custom.disabled,
            "nccl_version": communicator.pynccl_comm.nccl_version,
        }

    def basisserve_tp4_validate_weights(self, model_path: str) -> list[dict]:
        """Check real loaded Q/K and folded V weights against original shards."""
        import json
        from pathlib import Path
        from safetensors import safe_open
        from vllm.distributed import get_tp_group
        model = self.get_model()
        root = Path(model_path)
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        rank = get_tp_group().rank_in_group
        rows = []
        layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        for layer_id in (0, (layers - 1) // 2, layers - 1):
            attn = model.model.layers[layer_id].self_attn
            weight = attn.qkv_proj.weight
            prefix = f"model.layers.{layer_id}.self_attn."
            source = {}
            for name in ("q", "k", "v"):
                key = prefix + name + "_proj.weight"
                with safe_open(str(root / index[key]), framework="pt", device="cpu") as handle:
                    full = handle.get_tensor(key)
                width = attn.num_heads * 128 if name == "q" else 256
                source[name] = full[rank * width:(rank + 1) * width].to(weight.device)
            q_width = attn.num_heads * 128
            kv_offset = q_width + 256
            assert torch.equal(weight[:q_width], source["q"])
            assert torch.equal(weight[q_width:kv_offset], source["k"])
            generator = torch.Generator(device=weight.device).manual_seed(71)
            hidden = torch.randn(17, hidden_size, device=weight.device, generator=generator) / hidden_size**0.5
            dense = (hidden @ source["v"].float().T).view(17, 2, 128)
            expected = torch.einsum("thd,hdr->thr", dense, attn.qkv_proj.encoder.float()).flatten(1)
            actual = hidden @ weight[kv_offset:].float().T
            error = ((actual - expected).norm() / expected.norm()).item()
            assert error < 0.01 and torch.isfinite(actual).all()
            rows.append(dict(layer=layer_id, tp_rank=rank, dense_qk_exact=True,
                             value_rank=attn.value_rank, folded_value_relative_l2=error))
        return rows

    def basisserve_profile_start(self) -> None:
        """Profile rank zero only; benchmark timing runs never enable this."""
        if torch.distributed.get_rank() == 0:
            torch.cuda.synchronize()
            self._basisserve_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=False,
            )
            self._basisserve_profiler.start()

    def basisserve_profile_stop(self, output_prefix: str) -> None:
        if torch.distributed.get_rank() == 0:
            import csv
            import gzip
            from pathlib import Path
            import shutil
            import tempfile

            torch.cuda.synchronize()
            profiler = self._basisserve_profiler
            profiler.stop()
            prefix = Path(output_prefix)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            kernels = {}
            for event in profiler.events():
                if event.device_type == torch.autograd.DeviceType.CUDA:
                    row = kernels.setdefault(event.name, [0, 0.0])
                    row[0] += 1
                    row[1] += event.device_time_total
            with prefix.with_suffix(".kernels.csv").open("w") as handle:
                writer = csv.writer(handle)
                writer.writerow(["kernel", "count", "total_us", "mean_us"])
                for name, (count, total) in sorted(kernels.items(), key=lambda item: -item[1][1]):
                    writer.writerow([name, count, total, total / count])
            with tempfile.TemporaryDirectory(prefix="basisserve-profile-") as temporary:
                trace = Path(temporary) / "trace.json"
                profiler.export_chrome_trace(str(trace))
                with trace.open("rb") as source, gzip.open(prefix.with_suffix(".trace.json.gz"), "wb") as target:
                    shutil.copyfileobj(source, target)
            del self._basisserve_profiler


__all__ = ["BasisServeWorkerExtension"]
