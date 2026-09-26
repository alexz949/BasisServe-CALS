"""Prepared global V statistics and A8 decoder transport for sequential layers."""

import torch
import triton

from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.ragged_allgather import StaticRaggedPlan
from basisserve.kernels.nuq4_cache import nuq4_value_stats
from basisserve.kernels.latent_a8_pack import pack_latent_a8, unpack_latent_a8
from basisserve.kernels.fp8_wire import scaled_mm_e4m3_static
from basisserve.kernels.nuq4_decode import decode_workspace


class NUQ4Boundary:
    def __init__(self, group, device, max_tokens):
        self.stats_comm = FeatureRaggedCommunicator.from_distributed(group, device=device)
        self.wire_comm = FeatureRaggedCommunicator.from_distributed(group, device=device)
        assert self.stats_comm.world_size == self.wire_comm.world_size == 8
        self.stats_comm.configure_direct_workspace(tokens=max_tokens, max_total_width=1024, dtype=torch.bfloat16)
        self.wire_comm.configure_direct_workspace(tokens=max_tokens, max_total_width=4096, dtype=torch.uint8)
        self.stats = torch.empty((max_tokens, 4), device=device, dtype=torch.float32)
        self.token_major = torch.empty(max_tokens * 4096, device=device, dtype=torch.float8_e4m3fn)
        self.local = torch.empty(max_tokens * 512, device=device, dtype=torch.bfloat16)
        self.prepared = {}
        self.decode_scratch = {}
        self.capture_calls = self.eager_calls = 0

    def _plan(self, width, rows, stats):
        stream = torch.cuda.current_stream(self.local.device).cuda_stream
        key = (width, rows, stats, stream)
        prepared = self.prepared.get(key)
        if prepared is None:
            assert not torch.cuda.is_current_stream_capturing(), "Warm communication shapes on the capture stream"
            communicator = self.stats_comm if stats else self.wire_comm
            prepared = communicator.prepare_uniform(StaticRaggedPlan.from_source_widths([width]*8),
                tokens=rows, dtype=torch.bfloat16 if stats else torch.uint8, backend="uniform_nccl")
            self.prepared[key] = prepared
        return prepared

    def value_stats(self, value):
        rows, width = value.shape
        prepared = self._plan(width, rows, True)
        prepared.local_feature_major_view_fast().copy_(value.T)
        arena = prepared.gather_inplace_fast()
        return nuq4_value_stats(arena.T, self.stats[:rows])

    def attention_output(self, rows, width):
        return self.local[:rows*4*width].view(rows, 4, width).zero_()

    def decode_workspace(self, sequences, width):
        capacity = triton.next_power_of_2(sequences)
        key = (capacity, width)
        workspace = self.decode_scratch.get(key)
        if workspace is None:
            assert not torch.cuda.is_current_stream_capturing(), "Warm decode scratch before capture"
            workspace = decode_workspace(capacity, width, self.local.device)
            self.decode_scratch[key] = workspace
        return workspace

    def decode(self, latent, layer):
        rows, width = latent.shape
        prepared = self._plan(width, rows, False)
        pack_latent_a8(latent, layer.a8_scale, prepared.local_feature_major_view_fast())
        arena = prepared.gather_inplace_fast()
        token_major = self.token_major[:rows*width*8].view(rows, width*8)
        unpack_latent_a8(arena, layer.a8_scale, token_major)
        if torch.cuda.is_current_stream_capturing():
            self.capture_calls += 1
        else:
            self.eager_calls += 1
        return scaled_mm_e4m3_static(token_major, layer.decoder_fp8,
            left_scale=layer.a8_scale, right_scale=layer.decoder_scale, out_dtype=torch.bfloat16)
