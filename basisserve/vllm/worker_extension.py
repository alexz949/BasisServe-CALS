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
