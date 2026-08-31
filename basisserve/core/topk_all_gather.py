"""Fixed-size Top-K packets for source-local tensor-parallel AllGather.

Each rank selects a fixed number of coordinates from its local activation,
sorts the selected coordinates into source order, and packs one byte-aligned
bitmap followed by the raw bytes of the selected values. The byte packet is
reinterpreted as the value dtype before the collective so NCCL uses its native
BF16/FP16 path. The fixed packet size permits one all_gather_into_tensor call.

The correctness-first decoder reconstructs a dense zero-filled activation
before applying the replicated output projection. A fused sparse projection
can replace that decoder later without changing the packet format.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from basisserve.core.tp_output import RuntimeBreakdownRecorder, _all_gather_last_dim


def _group_world_size(process_group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group=process_group)


def fixed_topk_count(width: int, keep_ratio: float) -> int:
    """Return the nearest fixed-K endpoint for one source-local vector."""

    if width <= 0:
        raise ValueError("Top-K width must be positive")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep ratio must lie in (0, 1]")
    return min(width, max(1, int(round(width * keep_ratio))))


def fixed_topk_packet_bytes(width: int, kept: int, value_dtype: torch.dtype) -> int:
    """Return bitmap plus raw-value bytes for one fixed-K packet."""

    if width <= 0:
        raise ValueError("packet width must be positive")
    if not 1 <= kept <= width:
        raise ValueError(f"kept count must lie in [1, {width}]")
    probe = torch.empty((), dtype=value_dtype)
    if not probe.is_floating_point():
        raise TypeError("Top-K packet values must use a floating-point dtype")
    return math.ceil(width / 8) + kept * probe.element_size()


def pack_fixed_topk(
    source: Tensor,
    *,
    kept: int,
    value_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack source-local Top-K as bitmap followed by coordinate-sorted values."""

    if source.ndim < 1:
        raise ValueError("Top-K source must have at least one dimension")
    width = int(source.shape[-1])
    if not 1 <= kept <= width:
        raise ValueError(f"kept count must lie in [1, {width}]")
    dtype = value_dtype or source.dtype
    packet_bytes = fixed_topk_packet_bytes(width, kept, dtype)
    bitmap_bytes = math.ceil(width / 8)

    flat = source.reshape(-1, width)
    indices = torch.topk(
        flat.float().abs(),
        k=kept,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    # Bitmap records membership only, so values must use deterministic
    # coordinate order rather than torch.topk's value order.
    indices = indices.sort(dim=-1).values
    values = flat.gather(1, indices).to(dtype=dtype).contiguous()

    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    padded_width = bitmap_bytes * 8
    if padded_width != width:
        mask = F.pad(mask, (0, padded_width - width), value=False)
    bits = mask.reshape(-1, bitmap_bytes, 8).to(torch.int16)
    bit_weights = (1 << torch.arange(8, device=source.device, dtype=torch.int16)).view(
        1, 1, 8
    )
    bitmap = (bits * bit_weights).sum(dim=-1).to(torch.uint8)
    value_bytes = values.view(torch.uint8).reshape(flat.shape[0], -1)
    packet = torch.cat((bitmap, value_bytes), dim=-1)
    if int(packet.shape[-1]) != packet_bytes:
        raise RuntimeError("internal fixed Top-K packet-size mismatch")
    return (
        packet.reshape(*source.shape[:-1], packet_bytes),
        indices.reshape(*source.shape[:-1], kept),
        values.reshape(*source.shape[:-1], kept),
    )


def unpack_fixed_topk(
    packet: Tensor,
    *,
    width: int,
    kept: int,
    value_dtype: torch.dtype,
) -> Tensor:
    """Decode fixed Top-K packets into dense zero-filled vectors."""

    if packet.dtype != torch.uint8:
        raise TypeError("fixed Top-K packets must have dtype uint8")
    if packet.ndim < 1:
        raise ValueError("fixed Top-K packet must have at least one dimension")
    packet_bytes = fixed_topk_packet_bytes(width, kept, value_dtype)
    if int(packet.shape[-1]) != packet_bytes:
        raise ValueError(
            f"packet final width must be {packet_bytes}, got {packet.shape[-1]}"
        )
    bitmap_bytes = math.ceil(width / 8)
    flat_packet = packet.reshape(-1, packet_bytes).contiguous()
    bitmap = flat_packet[:, :bitmap_bytes]
    shifts = torch.arange(8, device=packet.device, dtype=torch.uint8).view(1, 1, 8)
    mask = torch.bitwise_and(
        torch.bitwise_right_shift(bitmap.unsqueeze(-1), shifts),
        1,
    ).bool()
    mask = mask.reshape(flat_packet.shape[0], bitmap_bytes * 8)[:, :width]

    value_bytes = flat_packet[:, bitmap_bytes:].contiguous()
    values = value_bytes.view(value_dtype).reshape(flat_packet.shape[0], kept)
    dense = torch.zeros(
        (flat_packet.shape[0], width),
        device=packet.device,
        dtype=value_dtype,
    )
    dense.masked_scatter_(mask, values.reshape(-1))
    return dense.reshape(*packet.shape[:-1], width)


class FixedTopKAllGatherOutput(nn.Module):
    """Source-local Top-K, one packed AllGather, then replicated output projection.

    full_output_weight has shape [d_out, world_size * local_width] and is
    replicated across TP ranks. This prototype intentionally uses a dense
    zero-fill decoder before the projection.
    """

    def __init__(
        self,
        full_output_weight: Tensor,
        *,
        keep_ratio: float,
        bias: Tensor | None = None,
        process_group: dist.ProcessGroup | None = None,
        communication_dtype: torch.dtype | None = None,
        debug_sync: bool = False,
    ) -> None:
        super().__init__()
        if full_output_weight.ndim != 2:
            raise ValueError("full_output_weight must be [d_out, d_in]")
        world_size = _group_world_size(process_group)
        if int(full_output_weight.shape[1]) % world_size:
            raise ValueError("full output input width must be divisible by world size")
        self.full_output_weight = nn.Parameter(
            full_output_weight.detach().contiguous(),
            requires_grad=False,
        )
        self.process_group = process_group
        self.communication_dtype = communication_dtype or full_output_weight.dtype
        self.debug_sync = bool(debug_sync)
        self.keep_ratio = float(keep_ratio)
        self._local_in_features = int(full_output_weight.shape[1]) // world_size
        self._kept = fixed_topk_count(self._local_in_features, self.keep_ratio)
        fixed_topk_packet_bytes(
            self._local_in_features,
            self._kept,
            self.communication_dtype,
        )
        if bias is None:
            self.register_parameter("bias", None)
        else:
            if tuple(bias.shape) != (int(full_output_weight.shape[0]),):
                raise ValueError(
                    f"bias must have shape ({full_output_weight.shape[0]},), "
                    f"got {tuple(bias.shape)}"
                )
            self.bias = nn.Parameter(bias.detach().contiguous(), requires_grad=False)
        self.runtime_breakdown = RuntimeBreakdownRecorder()

    @property
    def world_size(self) -> int:
        return _group_world_size(self.process_group)

    @property
    def local_in_features(self) -> int:
        return self._local_in_features

    @property
    def out_features(self) -> int:
        return int(self.full_output_weight.shape[0])

    @property
    def kept(self) -> int:
        return self._kept

    @property
    def bitmap_bytes(self) -> int:
        return math.ceil(self.local_in_features / 8)

    @property
    def raw_packet_bytes(self) -> int:
        return fixed_topk_packet_bytes(
            self.local_in_features,
            self.kept,
            self.communication_dtype,
        )

    @property
    def packet_bytes(self) -> int:
        element_size = torch.empty((), dtype=self.communication_dtype).element_size()
        return math.ceil(self.raw_packet_bytes / element_size) * element_size

    @property
    def packet_elements(self) -> int:
        element_size = torch.empty((), dtype=self.communication_dtype).element_size()
        return self.packet_bytes // element_size

    @property
    def dense_source_bytes(self) -> int:
        return self.local_in_features * torch.empty(
            (), dtype=self.communication_dtype
        ).element_size()

    def enable_runtime_breakdown(self, enabled: bool = True) -> None:
        self.runtime_breakdown.enabled = enabled

    def reset_runtime_breakdown(self) -> None:
        self.runtime_breakdown.reset()

    def runtime_breakdown_summary(self) -> dict[str, object]:
        summary = self.runtime_breakdown.summary()
        summary.update(
            {
                "module_type": "fixed_topk_all_gather",
                "collective": "single_value_dtype_carrier_all_gather",
                "local_in_features": self.local_in_features,
                "out_features": self.out_features,
                "world_size": self.world_size,
                "keep_ratio": self.keep_ratio,
                "kept": self.kept,
                "bitmap_bytes": self.bitmap_bytes,
                "raw_packet_bytes_per_source_token": self.raw_packet_bytes,
                "packet_bytes_per_source_token": self.packet_bytes,
                "dense_bytes_per_source_token": self.dense_source_bytes,
                "payload_reduction": self.dense_source_bytes / self.packet_bytes,
                "communication_dtype": str(self.communication_dtype),
                "decoder": "dense_zero_fill_then_linear",
                "debug_sync": self.debug_sync,
            }
        )
        return summary

    def _debug_synchronize(self, tensor: Tensor) -> None:
        if self.debug_sync and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def _pack(self, local_hidden_states: Tensor) -> Tensor:
        byte_packet, _, _ = pack_fixed_topk(
            local_hidden_states,
            kept=self.kept,
            value_dtype=self.communication_dtype,
        )
        if self.packet_bytes != self.raw_packet_bytes:
            byte_packet = F.pad(
                byte_packet,
                (0, self.packet_bytes - self.raw_packet_bytes),
                value=0,
            )
        return byte_packet.contiguous().view(self.communication_dtype).reshape(
            *local_hidden_states.shape[:-1],
            self.packet_elements,
        )

    def _unpack_gathered(self, gathered_packet: Tensor) -> Tensor:
        prefix = gathered_packet.shape[:-1]
        carriers = gathered_packet.reshape(
            -1,
            self.world_size,
            self.packet_elements,
        ).contiguous()
        packets = carriers.view(torch.uint8).reshape(
            -1,
            self.world_size,
            self.packet_bytes,
        )[..., : self.raw_packet_bytes]
        dense_sources = unpack_fixed_topk(
            packets,
            width=self.local_in_features,
            kept=self.kept,
            value_dtype=self.communication_dtype,
        )
        return dense_sources.reshape(
            *prefix,
            self.world_size * self.local_in_features,
        )

    def forward(
        self,
        local_hidden_states: Tensor,
        *,
        return_gathered_input: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if torch.is_grad_enabled() and local_hidden_states.requires_grad:
            raise RuntimeError("FixedTopKAllGatherOutput is an inference-only prototype")
        if int(local_hidden_states.shape[-1]) != self.local_in_features:
            raise ValueError(
                f"expected local hidden width {self.local_in_features}, "
                f"got {local_hidden_states.shape[-1]}"
            )

        recorder = self.runtime_breakdown
        device = local_hidden_states.device

        def body() -> tuple[Tensor, Tensor]:
            packet = recorder.record_segment(
                "topk_pack",
                lambda: self._pack(local_hidden_states),
                device=device,
            )
            self._debug_synchronize(packet)
            tokens = int(local_hidden_states.numel() // self.local_in_features)
            recorder.record_call(tokens)
            recorder.record_payload(
                dense_payload_elements=tokens * self.dense_source_bytes,
                compressed_payload_elements=tokens * self.packet_bytes,
                element_size_bytes=1,
            )
            gathered_packet = recorder.record_segment(
                "all_gather",
                lambda: _all_gather_last_dim(packet, self.process_group),
                device=device,
            )
            self._debug_synchronize(gathered_packet)
            gathered_input = recorder.record_segment(
                "unpack_zero_fill",
                lambda: self._unpack_gathered(gathered_packet),
                device=device,
            )
            self._debug_synchronize(gathered_input)
            if gathered_input.dtype != self.full_output_weight.dtype:
                gathered_input = recorder.record_segment(
                    "cast_from_comm",
                    lambda: gathered_input.to(self.full_output_weight.dtype),
                    device=device,
                )
            output = recorder.record_segment(
                "projection",
                lambda: F.linear(gathered_input, self.full_output_weight, self.bias),
                device=device,
            )
            self._debug_synchronize(output)
            return output, gathered_input

        output, gathered_input = recorder.record_segment("total", body, device=device)
        if return_gathered_input:
            return output, gathered_input
        return output

    def extra_repr(self) -> str:
        return (
            f"local_in_features={self.local_in_features}, "
            f"out_features={self.out_features}, world_size={self.world_size}, "
            f"kept={self.kept}, packet_bytes={self.packet_bytes}, "
            f"carrier_dtype={self.communication_dtype}"
        )


__all__ = [
    "FixedTopKAllGatherOutput",
    "fixed_topk_count",
    "fixed_topk_packet_bytes",
    "pack_fixed_topk",
    "unpack_fixed_topk",
]
