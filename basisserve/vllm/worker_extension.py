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


__all__ = ["BasisServeWorkerExtension"]
