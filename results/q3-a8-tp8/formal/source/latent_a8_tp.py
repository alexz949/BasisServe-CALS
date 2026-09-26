"""Prepared exact-width latent transport and BF16/FP8 decoder boundary."""

import torch

from basisserve.kernels.fp8_wire import (
    quantize_e4m3_static, quantize_e4m3_tensorwise_col_major, scaled_mm_e4m3_static,
)
from basisserve.kernels.latent_a8_pack import pack_latent_a8, unpack_latent_a8
from basisserve.kernels.ragged_allgather import StaticRaggedPlan


class LatentTPBoundary:
    """Construct on the stream that will execute or capture this boundary."""

    def __init__(self, communicator, widths, rows, decoder, scale, mode):
        assert mode in ("bf16", "a8", "w8a8")
        self.communicator, self.mode = communicator, mode
        self.plan = StaticRaggedPlan.from_source_widths(widths)
        self.decoder, self.scale = decoder, scale
        assert decoder.shape[0] == sum(widths) and decoder.dtype == torch.bfloat16
        dtype = torch.bfloat16 if mode == "bf16" else torch.uint8
        self.prepared = None
        if len(set(widths)) == 1:
            self.prepared = communicator.prepare_uniform(self.plan, tokens=rows,
                dtype=dtype, backend="uniform_nccl")
            self.local = self.prepared.local_feature_major_view_fast()
        else:
            self.local = communicator.direct_local_feature_major_view(self.plan, tokens=rows, dtype=dtype)
        self.codes_w, self.scale_w = quantize_e4m3_tensorwise_col_major(decoder)
        output_dtype = torch.float8_e4m3fn if mode == "w8a8" else torch.bfloat16
        self.token_major = torch.empty((rows, sum(widths)), device=decoder.device, dtype=output_dtype)

    def pack(self, value, fused=True):
        if self.mode == "bf16":
            self.local.copy_(value.T)
        elif fused:
            pack_latent_a8(value, self.scale, self.local)
        else:
            self.local.copy_(quantize_e4m3_static(value, self.scale).view(torch.uint8).T)

    def gather(self):
        if self.prepared is not None:
            return self.prepared.gather_inplace_fast()
        return self.communicator.gather(self.local, self.plan,
            backend="feature_direct", local_is_feature_major=True)

    def decode(self, arena, fused=True):
        if self.mode == "bf16":
            return torch.mm(arena.T, self.decoder)
        if fused:
            unpack_latent_a8(arena, self.scale, self.token_major)
        elif self.mode == "a8":
            self.token_major.copy_((arena.view(torch.float8_e4m3fn).T.float() * self.scale).to(torch.bfloat16))
        else:
            self.token_major.view(torch.uint8).copy_(arena.T)
        if self.mode == "a8":
            return torch.mm(self.token_major, self.decoder)
        return scaled_mm_e4m3_static(self.token_major, self.codes_w,
            left_scale=self.scale, right_scale=self.scale_w, out_dtype=torch.bfloat16)

    def __call__(self, value, fused=True):
        self.pack(value, fused)
        return self.decode(self.gather(), fused)
