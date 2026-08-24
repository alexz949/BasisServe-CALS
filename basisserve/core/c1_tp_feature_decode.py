"""Feature-major transport backend for the validated C1 TP serving boundary."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from basisserve.core.c1_tp_decode import PackedC1TPLayer
from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator


_FEATURE_BACKENDS = ("feature_direct", "feature_rma")


class C1FeatureMajorTPDecoder(nn.Module):
    """Decode one validated, process-owned C1 layer through a feature-major arena.

    ``PackedC1TPLayer`` is produced by ``C1TPFactorLoader`` and already carries
    the verified checkpoint hashes, TP ownership, rank schedule,
    ``StaticRaggedPlan``, and process-rank-ordered compact global decoder.
    """

    def __init__(
        self,
        packed: PackedC1TPLayer,
        *,
        communicator: FeatureRaggedCommunicator,
        backend: str = "feature_direct",
    ) -> None:
        super().__init__()
        if backend not in _FEATURE_BACKENDS:
            raise ValueError(
                f"unknown feature-major backend {backend!r}; expected one of {_FEATURE_BACKENDS}"
            )
        if communicator.rank != packed.process_rank:
            raise ValueError("packed C1 factors belong to another process rank")
        if communicator.world_size != len(packed.plan.source_widths):
            raise ValueError("packed C1 plan and communicator world sizes differ")
        expected_device = torch.device("cuda", communicator.device)
        if packed.global_decoder.device != expected_device:
            raise ValueError("packed C1 decoder is not on the communicator device")
        if packed.local_encoder.device != expected_device:
            raise ValueError("packed C1 encoder is not on the communicator device")
        if packed.global_decoder.dtype != packed.local_encoder.dtype:
            raise TypeError("packed C1 encoder and decoder dtypes differ")

        self.layer_index = packed.layer_index
        self.process_rank = packed.process_rank
        self.ownership = packed.ownership
        self.plan = packed.plan
        self.communicator = communicator
        self.backend = backend
        self.register_buffer("global_decoder", packed.global_decoder)

    @property
    def local_wire_width(self) -> int:
        return self.plan.source_widths[self.process_rank]

    @property
    def out_features(self) -> int:
        return int(self.global_decoder.shape[1])

    def forward(self, local_coordinates: Tensor) -> Tensor:
        """Decode token-major coordinates ending in this rank's wire width."""

        if torch.is_grad_enabled() and local_coordinates.requires_grad:
            raise RuntimeError("C1 TP decode is inference-only")
        if (
            local_coordinates.ndim < 1
            or int(local_coordinates.shape[-1]) != self.local_wire_width
        ):
            raise ValueError(
                "local C1 coordinates must end in this rank's ragged wire width: "
                f"expected {self.local_wire_width}, got {tuple(local_coordinates.shape)}"
            )
        self._validate_coordinates(local_coordinates)
        leading_shape = tuple(local_coordinates.shape[:-1])
        flat_coordinates = local_coordinates.reshape(-1, self.local_wire_width)
        output = self.communicator.all_gather_decode(
            flat_coordinates,
            self.plan,
            self.global_decoder,
            backend=self.backend,
            local_is_feature_major=False,
        )
        return output.reshape(*leading_shape, self.out_features)

    def forward_feature_major(self, local_coordinates: Tensor) -> Tensor:
        """Decode ``[local_wire_width, tokens]`` emitted directly by attention."""

        if (
            local_coordinates.ndim != 2
            or int(local_coordinates.shape[0]) != self.local_wire_width
        ):
            raise ValueError(
                "feature-major C1 coordinates must have shape "
                f"[{self.local_wire_width}, tokens], got {tuple(local_coordinates.shape)}"
            )
        self._validate_coordinates(local_coordinates)
        return self.communicator.all_gather_decode(
            local_coordinates,
            self.plan,
            self.global_decoder,
            backend=self.backend,
            local_is_feature_major=True,
        )

    def rma_output_view(self) -> Tensor:
        """Return the next registered destination ``[local_width, tokens]``.

        Acquire a fresh view immediately before every decode call because the
        RMA communicator alternates between two receive slots.
        """

        if self.backend != "feature_rma":
            raise RuntimeError("rma_output_view is only valid for feature_rma")
        return self.communicator.local_feature_major_view(self.plan)

    def _validate_coordinates(self, local_coordinates: Tensor) -> None:
        if (
            local_coordinates.device != self.global_decoder.device
            or local_coordinates.dtype != self.global_decoder.dtype
        ):
            raise ValueError("local C1 coordinates must match decoder dtype/device")


__all__ = ["C1FeatureMajorTPDecoder"]
