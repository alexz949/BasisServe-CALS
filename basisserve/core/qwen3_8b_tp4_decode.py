"""Qwen3-8B TP4 static-cache runtime for dense and C1 attention.

The runtime supports an initial full causal prefill followed by one-token
decode, uses four tensor-parallel processes, and is specialized to the
Qwen3-8B geometry with eight physical KV heads. Every process owns two
consecutive KV heads and the corresponding eight query heads.

Dense attention retains the Transformers row-parallel output projection.  C1
folds each physical-head encoder into the local Value projection, stores the
compact Value cache, gathers the four process-local coordinate blocks, and
applies the checkpoint's complete decoder on every process.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.kernels.compressed_v_decode_attention import (
    CompressedVDecodeWorkspace,
    compressed_v_decode_attention_cuda_tuned,
    compressed_v_decode_attention_triton,
    compressed_v_prefill_attention,
)
from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.fp8_wire import (
    FP8_E4M3_MAX,
    decode_e4m3_bytes,
    quantize_e4m3_static,
)
from basisserve.kernels.ragged_allgather import StaticRaggedPlan

TP_SIZE = 4
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 4096
QUERY_HEADS_PER_KV_HEAD = NUM_QUERY_HEADS // NUM_KV_HEADS
KV_HEADS_PER_PROCESS = NUM_KV_HEADS // TP_SIZE
QUERY_HEADS_PER_PROCESS = NUM_QUERY_HEADS // TP_SIZE


def load_qwen3_tp4_fp8_wire_scales(
    path: str | Path,
    *,
    num_layers: int,
) -> Tensor:
    """Load one static E4M3 dequantization scale per layer and TP source."""

    selected = Path(path).expanduser().resolve()
    payload = load_file(str(selected), device="cpu")
    if set(payload) != {"wire_scales", "wire_amax"}:
        raise ValueError(
            f"{selected} must contain exactly wire_scales and wire_amax"
        )
    scales = payload["wire_scales"]
    amax = payload["wire_amax"]
    expected = (int(num_layers), TP_SIZE)
    if tuple(scales.shape) != expected or tuple(amax.shape) != expected:
        raise ValueError(
            f"FP8 wire scale tensors must have shape {expected}, got "
            f"{tuple(scales.shape)} and {tuple(amax.shape)}"
        )
    scales = scales.float().contiguous()
    amax = amax.float().contiguous()
    if not torch.isfinite(scales).all() or bool((scales <= 0).any()):
        raise ValueError("FP8 wire scales must be finite and positive")
    if not torch.isfinite(amax).all() or bool((amax < 0).any()):
        raise ValueError("FP8 wire amax values must be finite and nonnegative")
    expected_scales = (amax / FP8_E4M3_MAX).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    torch.testing.assert_close(scales, expected_scales, rtol=1.0e-6, atol=0.0)
    return scales


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Qwen3TP4C1FactorLayer:
    layer_index: int
    source_rank: int
    encoders: Tensor
    decoders: Tensor
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        rank = int(self.source_rank)
        if not 0 < rank <= HEAD_DIM:
            raise ValueError(f"invalid C1 source rank {rank}")
        if tuple(self.encoders.shape) != (NUM_KV_HEADS, HEAD_DIM, rank):
            raise ValueError(
                "C1 encoders must have shape "
                f"{(NUM_KV_HEADS, HEAD_DIM, rank)}, got {tuple(self.encoders.shape)}"
            )
        if tuple(self.decoders.shape) != (
            NUM_QUERY_HEADS,
            rank,
            HIDDEN_SIZE,
        ):
            raise ValueError(
                "C1 decoders must have shape "
                f"{(NUM_QUERY_HEADS, rank, HIDDEN_SIZE)}, "
                f"got {tuple(self.decoders.shape)}"
            )
        if self.encoders.dtype != self.decoders.dtype:
            raise TypeError("C1 encoders and decoders must share a dtype")


def _factor_path(factor_dir: str | Path, layer_index: int) -> Path:
    root = Path(factor_dir).expanduser().resolve()
    selected = root / "selected_factors" / f"layer_{layer_index:03d}.safetensors"
    uniform = root / f"layer_{layer_index:03d}.safetensors"
    if selected.is_file():
        return selected
    if uniform.is_file():
        return uniform
    raise FileNotFoundError(
        f"no C1 layer {layer_index} factor under {root}"
    )


def load_qwen3_tp4_c1_factor_layer(
    factor_dir: str | Path,
    layer_index: int,
) -> Qwen3TP4C1FactorLayer:
    """Load one uniform-within-layer Qwen3-8B C1 factor artifact."""

    layer = int(layer_index)
    path = _factor_path(factor_dir, layer)
    payload = load_file(str(path), device="cpu")
    required = {"value_coordinate_encoders", "head_output_decoders"}
    if not required.issubset(payload):
        raise ValueError(f"{path} is missing C1 tensors {sorted(required - set(payload))}")
    encoders = payload["value_coordinate_encoders"]
    decoders = payload["head_output_decoders"]
    if encoders.ndim != 3 or decoders.ndim != 3:
        raise ValueError(f"{path} contains non-rank-3 C1 tensors")
    maximum_rank = int(encoders.shape[-1])
    if "source_ranks" in payload:
        raw_ranks = payload["source_ranks"]
        if raw_ranks.dtype != torch.int32 or tuple(raw_ranks.shape) != (NUM_KV_HEADS,):
            raise ValueError(f"{path} has an invalid source_ranks tensor")
        source_ranks = tuple(map(int, raw_ranks.tolist()))
        if len(set(source_ranks)) != 1:
            raise ValueError(
                "TP4 benchmark requires one rank for all eight physical sources "
                f"within a layer, got {source_ranks} at layer {layer}"
            )
        rank = source_ranks[0]
    else:
        rank = maximum_rank
    if rank > maximum_rank or int(decoders.shape[1]) < rank:
        raise ValueError(f"{path} does not contain the recorded source rank {rank}")
    return Qwen3TP4C1FactorLayer(
        layer_index=layer,
        source_rank=rank,
        encoders=encoders[:, :, :rank].contiguous(),
        decoders=decoders[:, :rank, :].contiguous(),
        path=path,
        sha256=file_sha256(path),
    )


def fold_local_c1_value_projection(
    dense_weight: Tensor,
    encoders: Tensor,
    dense_bias: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Fold two process-local physical-head encoders into a TP-local V shard."""

    if tuple(dense_weight.shape[:1]) != (KV_HEADS_PER_PROCESS * HEAD_DIM,):
        raise ValueError(
            "TP4 dense V weight must have output width "
            f"{KV_HEADS_PER_PROCESS * HEAD_DIM}, got {tuple(dense_weight.shape)}"
        )
    if dense_weight.ndim != 2 or int(dense_weight.shape[1]) != HIDDEN_SIZE:
        raise ValueError(f"invalid TP4 dense V weight shape {tuple(dense_weight.shape)}")
    if encoders.ndim != 3 or tuple(encoders.shape[:2]) != (
        KV_HEADS_PER_PROCESS,
        HEAD_DIM,
    ):
        raise ValueError(
            "local C1 encoders must have shape [2, 128, rank], got "
            f"{tuple(encoders.shape)}"
        )
    rank = int(encoders.shape[2])
    target_dtype = dense_weight.dtype
    dense = dense_weight.detach().reshape(
        KV_HEADS_PER_PROCESS,
        HEAD_DIM,
        HIDDEN_SIZE,
    ).float()
    folded = torch.bmm(
        encoders.detach().to(device=dense.device, dtype=torch.float32).transpose(1, 2),
        dense,
    ).reshape(KV_HEADS_PER_PROCESS * rank, HIDDEN_SIZE)
    compact_bias = None
    if dense_bias is not None:
        if tuple(dense_bias.shape) != (KV_HEADS_PER_PROCESS * HEAD_DIM,):
            raise ValueError(f"invalid TP4 dense V bias shape {tuple(dense_bias.shape)}")
        compact_bias = torch.bmm(
            encoders.detach().to(device=dense.device, dtype=torch.float32).transpose(1, 2),
            dense_bias.detach().reshape(KV_HEADS_PER_PROCESS, HEAD_DIM, 1).float(),
        ).reshape(KV_HEADS_PER_PROCESS * rank)
    return (
        folded.to(dtype=target_dtype).contiguous(),
        None if compact_bias is None else compact_bias.to(dtype=target_dtype).contiguous(),
    )


class _Qwen3TP4StaticDecodeAttention(nn.Module):
    def __init__(self, base_attention: nn.Module) -> None:
        super().__init__()
        if not dist.is_initialized() or dist.get_world_size() != TP_SIZE:
            raise RuntimeError("Qwen3-8B TP4 attention requires an initialized TP4 group")
        config = base_attention.config
        observed = (
            int(config.num_attention_heads),
            int(config.num_key_value_heads),
            int(base_attention.head_dim),
            int(config.hidden_size),
        )
        expected = (NUM_QUERY_HEADS, NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE)
        if observed != expected:
            raise ValueError(f"Qwen3-8B geometry mismatch: {observed} != {expected}")
        self.config = config
        self.layer_idx = int(base_attention.layer_idx)
        self.head_dim = HEAD_DIM
        self.num_key_value_groups = QUERY_HEADS_PER_KV_HEAD
        self.scaling = float(base_attention.scaling)
        self.attention_dropout = 0.0
        self.is_causal = True
        self.sliding_window = None
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.register_buffer("key_cache", None, persistent=False)
        self.register_buffer("value_cache", None, persistent=False)
        self._cache_length = 0

    @property
    def value_head_dim(self) -> int:
        raise NotImplementedError

    def _project_values(self, hidden_states: Tensor) -> Tensor:
        raise NotImplementedError

    def _project_output(self, local_output: Tensor) -> Tensor:
        raise NotImplementedError

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        is_prefill: bool,
    ) -> Tensor:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=is_prefill,
            scale=self.scaling,
            enable_gqa=True,
        )

    def configure_cache(
        self,
        *,
        batch_size: int,
        capacity: int,
        max_forward_tokens: int = 1,
    ) -> None:
        batch = int(batch_size)
        length = int(capacity)
        forward_tokens = int(max_forward_tokens)
        if batch <= 0 or length <= 0 or forward_tokens <= 0:
            raise ValueError("cache batch, capacity, and forward width must be positive")
        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        self.key_cache = torch.empty(
            batch,
            KV_HEADS_PER_PROCESS,
            length,
            HEAD_DIM,
            device=device,
            dtype=dtype,
        )
        self.value_cache = torch.empty(
            batch,
            KV_HEADS_PER_PROCESS,
            length,
            self.value_head_dim,
            device=device,
            dtype=dtype,
        )
        self._cache_length = 0

    def clear_cache(self) -> None:
        self.key_cache = None
        self.value_cache = None
        self._cache_length = 0

    def reset_cache(self) -> None:
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("configure_cache must be called before reset_cache")
        self._cache_length = 0

    @property
    def cache_bytes(self) -> int:
        tensors = (self.key_cache, self.value_cache)
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
            if tensor is not None
        )

    def _append_cache(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("configure_cache must be called before inference")
        tokens = int(key.shape[2])
        if tokens <= 0 or int(value.shape[2]) != tokens:
            raise ValueError("TP4 cache append requires matching nonempty K/V tokens")
        if tokens > 1 and self._cache_length != 0:
            raise ValueError("TP4 runtime supports full initial prefill, not chunked prefill")
        if int(key.shape[0]) != int(self.key_cache.shape[0]):
            raise ValueError("inference batch differs from configured cache batch")
        start = self._cache_length
        stop = start + tokens
        if stop > int(self.key_cache.shape[2]):
            raise RuntimeError("static KV cache capacity exceeded")
        self.key_cache[:, :, start:stop].copy_(key)
        self.value_cache[:, :, start:stop].copy_(value)
        self._cache_length = stop
        return (
            self.key_cache[:, :, : self._cache_length],
            self.value_cache[:, :, : self._cache_length],
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, None]:
        if past_key_values is not None:
            raise ValueError("the TP4 benchmark owns its static cache internally")
        if attention_mask is not None:
            raise ValueError("the TP4 benchmark expects an unpadded prompt without a mask")
        if hidden_states.ndim != 3 or int(hidden_states.shape[1]) <= 0:
            raise ValueError("the TP4 benchmark expects [batch, tokens, hidden] inputs")
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        batch = int(hidden_states.shape[0])
        tokens = int(hidden_states.shape[1])
        query = self.q_norm(
            self.q_proj(hidden_states).view(
                batch,
                tokens,
                QUERY_HEADS_PER_PROCESS,
                HEAD_DIM,
            )
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch,
                tokens,
                KV_HEADS_PER_PROCESS,
                HEAD_DIM,
            )
        ).transpose(1, 2)
        value = self._project_values(hidden_states).view(
            batch,
            tokens,
            KV_HEADS_PER_PROCESS,
            self.value_head_dim,
        ).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        cached_key, cached_value = self._append_cache(key, value)
        local_output = self._attention(
            query,
            cached_key,
            cached_value,
            is_prefill=tokens > 1,
        )
        return self._project_output(local_output), None


class Qwen3TP4DenseDecodeAttention(_Qwen3TP4StaticDecodeAttention):
    """Dense TP4 attention with a preallocated local KV cache."""

    def __init__(self, base_attention: nn.Module) -> None:
        super().__init__(base_attention)
        self.v_proj = base_attention.v_proj
        self.o_proj = base_attention.o_proj

    @property
    def value_head_dim(self) -> int:
        return HEAD_DIM

    def _project_values(self, hidden_states: Tensor) -> Tensor:
        return self.v_proj(hidden_states)

    def _project_output(self, local_output: Tensor) -> Tensor:
        batch = int(local_output.shape[0])
        tokens = int(local_output.shape[2])
        token_major = local_output.transpose(1, 2).reshape(batch, tokens, -1)
        return self.o_proj(token_major)


class Qwen3TP4C1DecodeAttention(_Qwen3TP4StaticDecodeAttention):
    """Compact-Value attention followed by packed feature-major AllGather."""

    def __init__(
        self,
        base_attention: nn.Module,
        factors: Qwen3TP4C1FactorLayer,
        communicator: FeatureRaggedCommunicator,
        decode_attention_backend: str,
        wire_dtype: str,
        fp8_wire_source_scales: Tensor | None,
    ) -> None:
        super().__init__(base_attention)
        if factors.layer_index != self.layer_idx:
            raise ValueError("C1 factors belong to another transformer layer")
        process_rank = dist.get_rank()
        kv_start = process_rank * KV_HEADS_PER_PROCESS
        kv_stop = kv_start + KV_HEADS_PER_PROCESS
        local_encoders = factors.encoders[kv_start:kv_stop].to(
            device=base_attention.v_proj.weight.device,
            dtype=base_attention.v_proj.weight.dtype,
        )
        compact_weight, compact_bias = fold_local_c1_value_projection(
            base_attention.v_proj.weight,
            local_encoders,
            base_attention.v_proj.bias,
        )
        self.source_rank = int(factors.source_rank)
        self.local_wire_width = QUERY_HEADS_PER_PROCESS * self.source_rank
        if communicator.world_size != TP_SIZE:
            raise ValueError("C1 packed communicator must use TP4")
        self.communicator = communicator
        self.plan = StaticRaggedPlan.from_source_widths(
            (self.local_wire_width,) * TP_SIZE
        )
        global_decoder = factors.decoders.reshape(
            NUM_QUERY_HEADS * self.source_rank,
            HIDDEN_SIZE,
        ).to(
            device=base_attention.q_proj.weight.device,
            dtype=base_attention.q_proj.weight.dtype,
        ).contiguous()
        self.register_buffer(
            "wire_amax",
            torch.zeros(
                (),
                device=base_attention.q_proj.weight.device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self._observe_wire_amax = False
        self.register_buffer("compact_v_weight", compact_weight)
        self.register_buffer("compact_v_bias", compact_bias)
        if wire_dtype == "bfloat16":
            if fp8_wire_source_scales is not None:
                raise ValueError("BF16 C1 wire must not receive FP8 scales")
            selected_decoder = global_decoder
            self.register_buffer("wire_scale", None, persistent=True)
        elif wire_dtype == "float8_e4m3fn":
            if fp8_wire_source_scales is None or tuple(
                fp8_wire_source_scales.shape
            ) != (TP_SIZE,):
                raise ValueError(
                    "FP8 C1 requires one calibrated wire scale per TP source"
                )
            source_scales = fp8_wire_source_scales.detach().float().contiguous()
            if not torch.isfinite(source_scales).all() or bool(
                (source_scales <= 0).any()
            ):
                raise ValueError("FP8 wire source scales must be finite and positive")
            decoder_row_scales = source_scales.repeat_interleave(
                self.local_wire_width
            ).to(device=global_decoder.device, dtype=torch.float32)
            selected_decoder = (
                global_decoder.float() * decoder_row_scales[:, None]
            ).to(dtype=global_decoder.dtype)
            self.register_buffer(
                "wire_scale",
                source_scales[process_rank].to(
                    device=global_decoder.device,
                    dtype=torch.float32,
                ),
                persistent=True,
            )
        else:
            raise ValueError(
                "C1 wire dtype must be 'bfloat16' or 'float8_e4m3fn'"
            )
        self.register_buffer("global_decoder", selected_decoder.contiguous())
        self.wire_dtype = wire_dtype
        self.factor_path = str(factors.path)
        self.factor_sha256 = factors.sha256
        if decode_attention_backend not in ("cuda", "triton"):
            raise ValueError("C1 decode attention backend must be 'cuda' or 'triton'")
        self.decode_attention_backend = decode_attention_backend
        self._decode_workspace: CompressedVDecodeWorkspace | None = None

    @property
    def value_head_dim(self) -> int:
        return self.source_rank

    def _project_values(self, hidden_states: Tensor) -> Tensor:
        return F.linear(hidden_states, self.compact_v_weight, self.compact_v_bias)

    def _decode_gathered(self, arena: Tensor) -> Tensor:
        if self.wire_dtype == "float8_e4m3fn":
            arena = decode_e4m3_bytes(arena, dtype=self.global_decoder.dtype)
        from basisserve.kernels.feature_ragged_allgather import decode_feature_major

        return decode_feature_major(arena, self.global_decoder)

    def set_wire_amax_observation(self, enabled: bool) -> None:
        """Enable or disable calibration-only observation of local wire output."""

        self._observe_wire_amax = bool(enabled)

    def reset_wire_amax(self) -> None:
        self.wire_amax.zero_()

    def calibrated_wire_scale(self) -> Tensor:
        if not self._observe_wire_amax:
            raise RuntimeError("wire amax observation is not enabled")
        return (self.wire_amax / FP8_E4M3_MAX).clamp_min(
            torch.finfo(torch.float32).tiny
        )

    def _update_wire_amax(self, local_output: Tensor) -> None:
        if not self._observe_wire_amax:
            return
        observed = local_output.detach().float().abs().amax()
        torch.maximum(self.wire_amax, observed, out=self.wire_amax)

    def configure_decode_workspace(
        self,
        workspace: CompressedVDecodeWorkspace,
    ) -> None:
        if workspace.query_heads != QUERY_HEADS_PER_PROCESS:
            raise ValueError("C1 decode workspace has the wrong query-head count")
        if workspace.max_value_dim < self.source_rank:
            raise ValueError("C1 decode workspace is narrower than this layer")
        self._decode_workspace = workspace

    def clear_cache(self) -> None:
        super().clear_cache()
        self._decode_workspace = None

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        is_prefill: bool,
    ) -> Tensor:
        if is_prefill:
            return compressed_v_prefill_attention(
                query,
                key,
                value,
                scale=self.scaling,
            )
        if self.decode_attention_backend == "triton":
            return compressed_v_decode_attention_triton(
                query,
                key,
                value,
                scale=self.scaling,
            )
        if self._decode_workspace is None:
            raise RuntimeError("configure_cache must allocate the CUDA decode workspace")
        feature_major_output = None
        if self.wire_dtype == "bfloat16":
            feature_major_output = self.communicator.direct_local_feature_major_view(
                self.plan,
                tokens=int(query.shape[0]),
                dtype=query.dtype,
            )
        return compressed_v_decode_attention_cuda_tuned(
            query,
            key,
            value,
            workspace=self._decode_workspace,
            scale=self.scaling,
            feature_major_output=feature_major_output,
        )

    def _project_output(self, local_output: Tensor) -> Tensor:
        self._update_wire_amax(local_output)
        if local_output.ndim == 2:
            batch = int(local_output.shape[1])
            if self.wire_dtype == "float8_e4m3fn":
                if self.wire_scale is None:
                    raise AssertionError("FP8 wire scale is unavailable")
                quantized_local = quantize_e4m3_static(
                    local_output,
                    self.wire_scale,
                ).view(torch.uint8)
                arena = self.communicator.gather(
                    quantized_local,
                    self.plan,
                    backend="feature_direct",
                    local_is_feature_major=True,
                )
                decoded = self._decode_gathered(arena)
                return decoded.reshape(batch, 1, HIDDEN_SIZE)
            arena = self.communicator.gather(
                local_output,
                self.plan,
                backend="feature_direct",
                local_is_feature_major=True,
            )
            decoded = self._decode_gathered(arena)
            return decoded.reshape(batch, 1, HIDDEN_SIZE)

        batch = int(local_output.shape[0])
        tokens = int(local_output.shape[2])
        rows = batch * tokens
        local_coordinates = (
            local_output.transpose(1, 2)
            .reshape(rows, self.local_wire_width)
            .contiguous()
        )
        if self.wire_dtype == "float8_e4m3fn":
            if self.wire_scale is None:
                raise AssertionError("FP8 wire scale is unavailable")
            local_coordinates = quantize_e4m3_static(
                local_coordinates,
                self.wire_scale,
            ).view(torch.uint8)
        arena = self.communicator.gather(
            local_coordinates,
            self.plan,
            backend="feature_direct",
        )
        decoded = self._decode_gathered(arena)
        return decoded.reshape(batch, tokens, HIDDEN_SIZE)


def configure_qwen3_tp4_caches(
    modules: Sequence[_Qwen3TP4StaticDecodeAttention],
    *,
    batch_size: int,
    capacity: int,
    max_forward_tokens: int = 1,
) -> int:
    """Allocate per-layer KV caches and shared C1 communication/attention arenas."""

    batch = int(batch_size)
    forward_tokens = int(max_forward_tokens)
    for module in modules:
        module.configure_cache(
            batch_size=batch,
            capacity=int(capacity),
            max_forward_tokens=forward_tokens,
        )
    c1_modules = tuple(
        module for module in modules if isinstance(module, Qwen3TP4C1DecodeAttention)
    )
    decode_workspace: CompressedVDecodeWorkspace | None = None
    if c1_modules:
        rows = batch * forward_tokens
        reference = c1_modules[0].q_proj.weight
        cuda_modules = tuple(
            module
            for module in c1_modules
            if module.decode_attention_backend == "cuda"
        )
        if cuda_modules:
            decode_workspace = CompressedVDecodeWorkspace.allocate(
                batch=batch,
                query_heads=QUERY_HEADS_PER_PROCESS,
                max_value_dim=max(module.source_rank for module in cuda_modules),
                dtype=reference.dtype,
                device=reference.device,
            )
            for module in cuda_modules:
                module.configure_decode_workspace(decode_workspace)
        communicators = {id(module.communicator): module.communicator for module in c1_modules}
        if len(communicators) != 1:
            raise RuntimeError("Qwen3 TP4 C1 layers must share one packed communicator")
        communicator = next(iter(communicators.values()))
        wire_dtypes = {module.wire_dtype for module in c1_modules}
        if len(wire_dtypes) != 1:
            raise RuntimeError("Qwen3 TP4 C1 layers must share one wire dtype")
        wire_dtype = (
            torch.uint8
            if next(iter(wire_dtypes)) == "float8_e4m3fn"
            else reference.dtype
        )
        communicator.configure_direct_workspace(
            tokens=rows,
            max_total_width=max(module.plan.total_width for module in c1_modules),
            dtype=wire_dtype,
        )
    return sum(module.cache_bytes for module in modules) + (
        0 if decode_workspace is None else decode_workspace.nbytes
    )


def install_qwen3_tp4_decode_attention(
    model: nn.Module,
    *,
    factor_dir: str | Path | None = None,
    c1_decode_attention_backend: str | None = None,
    c1_wire_dtype: str = "bfloat16",
    c1_fp8_wire_scales: str | Path | None = None,
) -> tuple[_Qwen3TP4StaticDecodeAttention, ...]:
    """Replace every Qwen3 layer with the dense or C1 static decode path."""

    if factor_dir is None:
        if c1_decode_attention_backend is not None:
            raise ValueError("dense attention must not select a C1 decode backend")
        if c1_wire_dtype != "bfloat16":
            raise ValueError("dense attention must use a BF16 wire")
        if c1_fp8_wire_scales is not None:
            raise ValueError("dense attention must not receive FP8 wire scales")
    elif c1_decode_attention_backend not in ("cuda", "triton"):
        raise ValueError("C1 attention requires an explicit 'cuda' or 'triton' backend")
    elif c1_wire_dtype not in ("bfloat16", "float8_e4m3fn"):
        raise ValueError("unsupported C1 wire dtype")
    elif c1_wire_dtype == "float8_e4m3fn" and c1_fp8_wire_scales is None:
        raise ValueError("FP8 C1 requires calibrated --c1-fp8-wire-scales")
    elif c1_wire_dtype == "bfloat16" and c1_fp8_wire_scales is not None:
        raise ValueError("BF16 C1 wire must not receive FP8 scales")

    layers = model.model.layers
    wire_scales = (
        None
        if c1_fp8_wire_scales is None
        else load_qwen3_tp4_fp8_wire_scales(
            c1_fp8_wire_scales,
            num_layers=len(layers),
        )
    )
    communicator = (
        None
        if factor_dir is None
        else FeatureRaggedCommunicator.from_distributed(
            device=model.model.embed_tokens.weight.device,
        )
    )
    installed: list[_Qwen3TP4StaticDecodeAttention] = []
    for layer_index, layer in enumerate(layers):
        base_attention = layer.self_attn
        if factor_dir is None:
            replacement: _Qwen3TP4StaticDecodeAttention = (
                Qwen3TP4DenseDecodeAttention(base_attention)
            )
        else:
            factors = load_qwen3_tp4_c1_factor_layer(factor_dir, layer_index)
            replacement = Qwen3TP4C1DecodeAttention(
                base_attention,
                factors,
                communicator,
                c1_decode_attention_backend,
                c1_wire_dtype,
                None if wire_scales is None else wire_scales[layer_index],
            )
        replacement.eval()
        layer.self_attn = replacement
        installed.append(replacement)
    return tuple(installed)


def close_qwen3_tp4_packed_communicator(
    modules: Sequence[_Qwen3TP4StaticDecodeAttention],
) -> None:
    """Collectively close the one communicator shared by all C1 layers."""

    communicators = {
        id(module.communicator): module.communicator
        for module in modules
        if isinstance(module, Qwen3TP4C1DecodeAttention)
    }
    for communicator in communicators.values():
        communicator.close()


__all__ = [
    "HEAD_DIM",
    "HIDDEN_SIZE",
    "KV_HEADS_PER_PROCESS",
    "NUM_KV_HEADS",
    "NUM_QUERY_HEADS",
    "QUERY_HEADS_PER_PROCESS",
    "Qwen3TP4C1DecodeAttention",
    "Qwen3TP4C1FactorLayer",
    "Qwen3TP4DenseDecodeAttention",
    "TP_SIZE",
    "close_qwen3_tp4_packed_communicator",
    "configure_qwen3_tp4_caches",
    "file_sha256",
    "fold_local_c1_value_projection",
    "install_qwen3_tp4_decode_attention",
    "load_qwen3_tp4_c1_factor_layer",
    "load_qwen3_tp4_fp8_wire_scales",
]
