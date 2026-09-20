"""Stream-bound prepared NCCL AllGather and one replicated decoder GEMM."""

from __future__ import annotations

import torch
from torch import Tensor

from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.ragged_allgather import StaticRaggedPlan
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op


class PreparedC1Output:
    """One communicator shared by sequential attention layers.

    Plans are warmed on the actual capture stream before capture begins.
    Returning the GEMM result instead of the mutable arena also gives the
    custom operator an ordinary, non-aliasing output contract for compilation.
    """

    def __init__(self, group, device, local_width: int, max_tokens: int):
        self.communicator = FeatureRaggedCommunicator.from_distributed(group, device=device)
        self.plan = StaticRaggedPlan.from_source_widths((local_width,) * self.communicator.world_size)
        self.communicator.configure_direct_workspace(
            tokens=max_tokens, max_total_width=self.plan.total_width, dtype=torch.bfloat16,
        )
        self.prepared = {}
        self.capture_calls = 0
        self.eager_calls = 0

    def __call__(self, local: Tensor, decoder: Tensor) -> Tensor:
        stream = torch.cuda.current_stream(local.device)
        key = (local.shape[0], stream.cuda_stream)
        capturing = torch.cuda.is_current_stream_capturing()
        prepared = self.prepared.get(key)
        if prepared is None:
            assert not capturing, "Warm this token count on the capture stream first"
            prepared = self.communicator.prepare_uniform(
                self.plan, tokens=local.shape[0], dtype=local.dtype, backend="uniform_nccl",
            )
            self.prepared[key] = prepared
        if capturing:
            self.capture_calls += 1
        else:
            self.eager_calls += 1
        # Native vLLM attention emits token-major coordinates. The source-local
        # pack is currently required; the gathered arena needs no global pack.
        prepared.local_feature_major_view_fast().copy_(local.T)
        arena = prepared.gather_inplace_fast()
        return torch.mm(arena.T, decoder)

    def statistics(self):
        return dict(backend="uniform_nccl", decoder_gemms=1,
                    capture_calls=self.capture_calls, eager_calls=self.eager_calls,
                    prepared_shapes=sorted({int(key[0]) for key in self.prepared}))


def prepared_c1_output(local: Tensor, decoder: Tensor, layer_name: str) -> Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer._basisserve_c1_output(local, decoder)


def prepared_c1_output_fake(local: Tensor, decoder: Tensor, layer_name: str) -> Tensor:
    return local.new_empty((local.shape[0], decoder.shape[1]))


direct_register_custom_op(
    op_name="basisserve_prepared_c1_output", op_func=prepared_c1_output,
    fake_impl=prepared_c1_output_fake,
)
