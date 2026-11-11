"""Stream-bound C1 AllGather and one replicated decoder GEMM.

On fully NVLink-connected TP groups (where vLLM enables its custom all-reduce) decode gathers run in place on one
fixed NCCL symmetric-window arena (ncclMemAlloc + NCCL_WIN_COLL_SYMMETRIC through vLLM's TP PyNccl communicator), so
NCCL >= 2.27 uses its one-shot NVLink kernels: 8-26 us instead of 17-34 us for the ring at 1-256 tokens on 8 x A100
(`benchmarks/a100/bench_allgather.py`). Prefill and PCIe-only groups use the prepared NCCL plan.
"""

from __future__ import annotations

import torch
from torch import Tensor
from basisserve.core.tp4_c1_factors import local_decoder

from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.ragged_allgather import StaticRaggedPlan
from vllm.distributed.device_communicators.pynccl_allocator import get_nccl_mem_pool
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.torch_utils import direct_register_custom_op


class PreparedC1Output:
    """One communicator shared by sequential attention layers.

    Plans are warmed on the actual capture stream before capture begins.
    Returning the GEMM result instead of the mutable arena also gives the
    custom operator an ordinary, non-aliasing output contract for compilation.
    """

    def __init__(self, group, device, local_width: int, max_tokens: int, *, output_mode="ag", audit_routes=False):
        assert output_mode in ("ag", "hybrid", "ar")
        self.output_mode = output_mode
        self.audit_routes = audit_routes
        self.route_counts = torch.zeros(6, dtype=torch.int64, device=device)
        self.communicator = FeatureRaggedCommunicator.from_distributed(group, device=device)
        self.plan = StaticRaggedPlan.from_source_widths(
            (local_width,) * self.communicator.world_size
        )
        self.communicator.configure_direct_workspace(
            tokens=max_tokens,
            max_total_width=self.plan.total_width,
            dtype=torch.bfloat16,
        )
        self.prepared = {}
        self.capture_calls = 0
        self.eager_calls = 0
        self.direct_decode_calls = 0
        # A100 (SM80) cuBLAS picks align-2 kernels for the transposed feature-major operand at small token counts
        # (2.3-3x slower at 2/4/16 tokens; cuBLASLt likewise); a token-major copy restores the fast path there.
        self.token_major_small = torch.cuda.get_device_capability(device) == (8, 0)
        self.local_width = local_width
        self.symmetric = None
        tp = get_tp_group().device_communicator
        nvlink = tp.ca_comm is not None and not tp.ca_comm.disabled
        if nvlink and tp.pynccl_comm is not None and tp.pynccl_comm.nccl_version >= 22703:
            pool = get_nccl_mem_pool()
            assert pool is not None, "NCCL symmetric memory allocator unavailable"
            with torch.cuda.use_mem_pool(pool):
                arena = torch.empty(self.plan.total_width * max_tokens, dtype=torch.bfloat16, device=device)
            # Collective: every rank registers the same-sized arena once, in the same order.
            tp.pynccl_comm.register_comm_window(arena)
            # NCCL places blocks by PyNccl rank; decoder rows follow the communicator (TP group) rank.
            assert tp.pynccl_comm.rank == self.communicator.rank
            self.symmetric = (tp.pynccl_comm, arena, self.communicator.rank)

    def _prepared(
        self,
        tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        stream = torch.cuda.current_stream(device)
        key = (tokens, stream.cuda_stream)
        capturing = torch.cuda.is_current_stream_capturing()
        prepared = self.prepared.get(key)
        if prepared is None:
            assert not capturing, "Warm this token count on the capture stream first"
            prepared = self.communicator.prepare_uniform(
                self.plan,
                tokens=tokens,
                dtype=dtype,
                backend="uniform_nccl",
            )
            self.prepared[key] = prepared
        if capturing:
            self.capture_calls += 1
        else:
            self.eager_calls += 1
        return prepared

    def _decode(self, arena: Tensor, decoder: Tensor) -> Tensor:
        """One replicated decoder GEMM over the gathered feature-major arena [total_width, tokens]."""
        tokens = arena.T
        if self.token_major_small and arena.shape[1] <= 16:
            tokens = tokens.contiguous()
        return torch.mm(tokens, decoder)

    def __call__(self, local: Tensor, decoder: Tensor) -> Tensor:
        prepared = self._prepared(
            int(local.shape[0]), local.dtype, local.device,
        )
        # Native vLLM attention emits token-major coordinates. The source-local
        # pack is currently required; the gathered arena needs no global pack.
        prepared.local_feature_major_view_fast().copy_(local.T)
        return self._decode(prepared.gather_inplace_fast(), decoder)

    def local_reduce(self, local: Tensor, decoder: Tensor) -> Tensor:
        """Sum local Z_p D_p through vLLM's production TP AllReduce dispatcher."""
        weights = local_decoder(decoder, self.communicator.rank, self.local_width)
        return get_tp_group().all_reduce(torch.mm(local, weights))

    def attention(
        self,
        attn_layer,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kv_cache: Tensor,
        attn_metadata,
        decoder: Tensor,
    ) -> Tensor:
        tokens = int(query.shape[0])
        if self.audit_routes and attn_metadata is not None:
            step = attn_metadata.basisserve_step
            assert step is not None
            if step.kind != 'empty':
                index = ('prefill', 'mixed', 'decode').index(step.kind) * 2
                index += int(step.collective(self.output_mode) == 'ar')
                # Captured GPU increments also execute on every graph replay.
                self.route_counts[index].add_(1)
        if self.output_mode != "ag" and attn_metadata is not None:
            step = attn_metadata.basisserve_step
            assert step is not None, "Hybrid routing requires scheduler prefill metadata"
            if step.collective(self.output_mode) == "ar":
                output = torch.empty(
                    (tokens, attn_layer.impl.num_heads, attn_layer.head_size_v),
                    dtype=query.dtype, device=query.device,
                )
                attn_layer.impl.forward(attn_layer, query, key, value, kv_cache, attn_metadata, output)
                return self.local_reduce(output.flatten(1), decoder)
        if (
            attn_metadata is not None
            and attn_metadata.max_query_len <= 1
        ):
            if self.symmetric is not None:
                nccl, buffer, rank = self.symmetric
                arena = buffer[: self.plan.total_width * tokens].view(self.plan.total_width, tokens)
                local = arena[rank * self.local_width:(rank + 1) * self.local_width]
                if torch.cuda.is_current_stream_capturing():
                    self.capture_calls += 1
                else:
                    self.eager_calls += 1
            else:
                prepared = self._prepared(tokens, query.dtype, query.device)
                local = prepared.local_feature_major_view_fast()
            output = local.T.view(
                tokens, attn_layer.impl.num_heads, attn_layer.head_size_v
            )
            attn_layer.impl.forward(
                attn_layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
            )
            self.direct_decode_calls += 1
            if self.symmetric is not None:
                nccl.all_gather(arena.view(-1), local.reshape(-1))  # in place: local is this rank's block
                return self._decode(arena, decoder)
            return self._decode(prepared.gather_inplace_fast(), decoder)

        output = torch.empty(
            (tokens, attn_layer.impl.num_heads, attn_layer.head_size_v),
            dtype=query.dtype,
            device=query.device,
        )
        attn_layer.impl.forward(
            attn_layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
        )
        return self(output.flatten(1), decoder)

    def statistics(self):
        return dict(backend="uniform_nccl", decoder_gemms=1, output_mode=self.output_mode,
                    allreduce_dispatch="vllm_tp_group.all_reduce",
                    route_counts=dict(zip(('prefill_ag', 'prefill_ar', 'mixed_ag', 'mixed_ar',
                                           'decode_ag', 'decode_ar'), self.route_counts.tolist())),
                    decode_gather="nccl_symmetric_inplace" if self.symmetric is not None else "uniform_nccl",
                    token_major_small=self.token_major_small,
                    capture_calls=self.capture_calls, eager_calls=self.eager_calls,
                    direct_decode_calls=self.direct_decode_calls,
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


def c1_attention_output(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decoder: Tensor,
    layer_name: str,
) -> Tensor:
    attn_metadata, attn_layer, kv_cache, slot_mapping = get_attention_context(
        layer_name
    )
    query = query.view(-1, attn_layer.num_heads, attn_layer.head_size)
    key = key.view(-1, attn_layer.num_kv_heads, attn_layer.head_size)
    value = value.view(-1, attn_layer.num_kv_heads, attn_layer.head_size_v)
    if slot_mapping is not None:
        attn_layer.impl.do_kv_cache_update(
            attn_layer,
            key,
            value,
            kv_cache,
            slot_mapping,
        )
    boundary = attn_layer._basisserve_c1_output
    return boundary.attention(
        attn_layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        decoder,
    )


def c1_attention_output_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decoder: Tensor,
    layer_name: str,
) -> Tensor:
    return query.new_empty((query.shape[0], decoder.shape[1]))


direct_register_custom_op(
    op_name="basisserve_c1_attention_output",
    op_func=c1_attention_output,
    fake_impl=c1_attention_output_fake,
)
