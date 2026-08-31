"""Qwen3-32B TP4 static-cache runtime for dense and ragged C1 attention.

Each TP4 process owns two consecutive physical KV heads and sixteen query
heads.  A ragged C1 checkpoint may assign a different Value rank to every KV
source.  The runtime therefore keeps the two local compressed Value caches in
one packed tensor, evaluates the two source groups independently, gathers one
packed process-local coordinate block, and applies the complete ragged decoder
on every process.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from basisserve.kernels.feature_ragged_allgather import (
    FeatureRaggedCommunicator,
    PreparedUniformAllGather,
    decode_feature_major,
    decode_feature_major_e4m3,
)
from basisserve.kernels.fp8_wire import (
    FP8_E4M3_MAX,
    quantize_e4m3_static,
    quantize_e4m3_tensorwise_col_major,
)
from basisserve.kernels.ragged_allgather import StaticRaggedPlan


TP_SIZE = 4
NUM_QUERY_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 5120
NUM_LAYERS = 64
QUERY_HEADS_PER_KV_HEAD = NUM_QUERY_HEADS // NUM_KV_HEADS
KV_HEADS_PER_PROCESS = NUM_KV_HEADS // TP_SIZE
QUERY_HEADS_PER_PROCESS = NUM_QUERY_HEADS // TP_SIZE
FACTOR_FORMAT = "basisserve.qwen3_32b.gqa_c1.ragged_schedule_als.v1"
CUDA_VALUE_RANKS = frozenset((32, 48, 64, 80, 96, 112))


def load_qwen3_32b_tp4_fp8_wire_scales(path: str | Path) -> Tensor:
    """Load one static E4M3 dequantization scale per layer and TP source."""

    selected = Path(path).expanduser().resolve()
    payload = load_file(str(selected), device="cpu")
    if set(payload) != {"wire_scales", "wire_amax"}:
        raise ValueError(
            f"{selected} must contain exactly wire_scales and wire_amax"
        )
    scales = payload["wire_scales"]
    amax = payload["wire_amax"]
    expected = (NUM_LAYERS, TP_SIZE)
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
class Qwen3_32BTP4C1FactorLayer:
    layer_index: int
    source_ranks: tuple[int, ...]
    encoders: Tensor
    decoders: Tensor
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if len(self.source_ranks) != NUM_KV_HEADS:
            raise ValueError("C1 factors must record eight source ranks")
        if any(not 0 < rank <= HEAD_DIM for rank in self.source_ranks):
            raise ValueError(f"invalid C1 source ranks {self.source_ranks}")
        maximum_rank = max(self.source_ranks)
        expected_encoders = (NUM_KV_HEADS, HEAD_DIM, maximum_rank)
        expected_decoders = (NUM_QUERY_HEADS, maximum_rank, HIDDEN_SIZE)
        if tuple(self.encoders.shape) != expected_encoders:
            raise ValueError(
                f"C1 encoders must have shape {expected_encoders}, got "
                f"{tuple(self.encoders.shape)}"
            )
        if tuple(self.decoders.shape) != expected_decoders:
            raise ValueError(
                f"C1 decoders must have shape {expected_decoders}, got "
                f"{tuple(self.decoders.shape)}"
            )
        if self.encoders.dtype != self.decoders.dtype:
            raise TypeError("C1 encoders and decoders must share a dtype")

    @property
    def process_wire_widths(self) -> tuple[int, ...]:
        return tuple(
            QUERY_HEADS_PER_KV_HEAD
            * sum(
                self.source_ranks[
                    process * KV_HEADS_PER_PROCESS : (process + 1)
                    * KV_HEADS_PER_PROCESS
                ]
            )
            for process in range(TP_SIZE)
        )


def load_qwen3_32b_tp4_c1_manifest(
    factor_dir: str | Path,
) -> Mapping[str, Any]:
    """Load and validate the layer inventory of a Qwen3-32B C1 checkpoint."""

    root = Path(factor_dir).expanduser().resolve()
    path = root / "result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("format") != FACTOR_FORMAT or result.get("status") != "complete":
        raise ValueError(f"{path} is not a completed Qwen3-32B ragged ALS result")
    schedule = result.get("selection", {}).get("selected_schedule")
    if not isinstance(schedule, list) or len(schedule) != NUM_LAYERS:
        raise ValueError("ragged checkpoint schedule does not cover all 64 layers")
    return result


def load_qwen3_32b_tp4_c1_factor_layer(
    factor_dir: str | Path,
    layer_index: int,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> Qwen3_32BTP4C1FactorLayer:
    """Load and hash-check one source-ragged C1 layer."""

    root = Path(factor_dir).expanduser().resolve()
    result = (
        load_qwen3_32b_tp4_c1_manifest(root)
        if manifest is None
        else manifest
    )
    layer = int(layer_index)
    if not 0 <= layer < NUM_LAYERS:
        raise ValueError(f"layer index out of range: {layer}")
    artifact = result.get("artifacts", {}).get(str(layer))
    if not isinstance(artifact, dict):
        raise ValueError(f"checkpoint manifest has no artifact for layer {layer}")
    path = root / str(artifact.get("file"))
    observed_hash = file_sha256(path)
    expected_hash = str(artifact.get("sha256"))
    if observed_hash != expected_hash:
        raise ValueError(f"factor hash mismatch at layer {layer}")
    payload = load_file(str(path), device="cpu")
    required = {
        "value_coordinate_encoders",
        "head_output_decoders",
        "source_ranks",
    }
    if set(payload) != required:
        raise ValueError(f"unexpected tensors in {path}: {sorted(payload)}")
    raw_ranks = payload["source_ranks"]
    if raw_ranks.dtype != torch.int32 or tuple(raw_ranks.shape) != (NUM_KV_HEADS,):
        raise ValueError(f"{path} has an invalid source_ranks tensor")
    source_ranks = tuple(map(int, raw_ranks.tolist()))
    selected = tuple(
        map(int, result["selection"]["selected_schedule"][layer])
    )
    if source_ranks != selected:
        raise ValueError(
            f"factor ranks differ from selected schedule at layer {layer}: "
            f"{source_ranks} != {selected}"
        )
    return Qwen3_32BTP4C1FactorLayer(
        layer_index=layer,
        source_ranks=source_ranks,
        encoders=payload["value_coordinate_encoders"].contiguous(),
        decoders=payload["head_output_decoders"].contiguous(),
        path=path,
        sha256=observed_hash,
    )


def fold_ragged_local_c1_value_projection(
    dense_weight: Tensor,
    encoders: Sequence[Tensor],
    dense_bias: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Fold two differently sized source encoders into one packed V shard."""

    expected_width = KV_HEADS_PER_PROCESS * HEAD_DIM
    if dense_weight.ndim != 2 or tuple(dense_weight.shape) != (
        expected_width,
        HIDDEN_SIZE,
    ):
        raise ValueError(f"invalid TP4 dense V weight shape {tuple(dense_weight.shape)}")
    if len(encoders) != KV_HEADS_PER_PROCESS:
        raise ValueError("TP4 requires two process-local C1 encoders")
    if dense_bias is not None and tuple(dense_bias.shape) != (expected_width,):
        raise ValueError(f"invalid TP4 dense V bias shape {tuple(dense_bias.shape)}")
    tensors = (dense_weight, *encoders) + (
        () if dense_bias is None else (dense_bias,)
    )
    if any(tensor.device != dense_weight.device for tensor in tensors):
        raise ValueError("dense V weight, encoders, and bias must share one device")
    work_dtype = (
        torch.float64
        if any(tensor.dtype == torch.float64 for tensor in tensors)
        else torch.float32
    )
    folded_weights: list[Tensor] = []
    folded_biases: list[Tensor] = []
    for local_source, encoder in enumerate(encoders):
        if encoder.ndim != 2 or int(encoder.shape[0]) != HEAD_DIM:
            raise ValueError("each local encoder must have shape [128, source_rank]")
        start = local_source * HEAD_DIM
        stop = start + HEAD_DIM
        folded_weights.append(
            encoder.to(work_dtype).T
            @ dense_weight[start:stop].detach().to(work_dtype)
        )
        if dense_bias is not None:
            folded_biases.append(
                encoder.to(work_dtype).T
                @ dense_bias[start:stop].detach().to(work_dtype)
            )
    target_dtype = dense_weight.dtype
    return (
        torch.cat(folded_weights, dim=0).to(dtype=target_dtype).contiguous(),
        None
        if dense_bias is None
        else torch.cat(folded_biases, dim=0).to(dtype=target_dtype).contiguous(),
    )


def view_uniform_local_c1_value_heads(
    packed_values: Tensor,
    source_rank: int,
) -> Tensor:
    """View packed two-source coordinates as ``[B, 2, S, rank]``.

    The folded Value projection stores source 0 followed by source 1 in the
    last dimension. Splitting that dimension and permuting only metadata lets
    the fused decode kernel consume both local KV sources without a copy.
    """

    rank = int(source_rank)
    if rank <= 0:
        raise ValueError("source rank must be positive")
    if packed_values.ndim != 3:
        raise ValueError("packed local Values must have shape [batch, sequence, 2*rank]")
    if int(packed_values.shape[-1]) != KV_HEADS_PER_PROCESS * rank:
        raise ValueError(
            "packed local Value width differs from two uniform source ranks"
        )
    if packed_values.stride(-1) != 1:
        raise ValueError("packed local Value coordinates must be contiguous")
    batch, sequence = map(int, packed_values.shape[:2])
    return packed_values.view(
        batch,
        sequence,
        KV_HEADS_PER_PROCESS,
        rank,
    ).permute(0, 2, 1, 3)


class _Qwen3_32BTP4AttentionBase(nn.Module):
    def __init__(self, base_attention: nn.Module) -> None:
        super().__init__()
        if not dist.is_initialized() or dist.get_world_size() != TP_SIZE:
            raise RuntimeError("Qwen3-32B attention requires an initialized TP4 group")
        config = base_attention.config
        observed = (
            int(config.num_attention_heads),
            int(config.num_key_value_heads),
            int(base_attention.head_dim),
            int(config.hidden_size),
        )
        expected = (NUM_QUERY_HEADS, NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE)
        if observed != expected:
            raise ValueError(f"Qwen3-32B geometry mismatch: {observed} != {expected}")
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
        self.register_buffer("graph_decode_position", None, persistent=False)
        self._cache_length = 0

    def _project_and_rotate(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        batch, tokens = map(int, hidden_states.shape[:2])
        query = self.q_norm(
            self.q_proj(hidden_states).view(
                batch, tokens, QUERY_HEADS_PER_PROCESS, HEAD_DIM
            )
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch, tokens, KV_HEADS_PER_PROCESS, HEAD_DIM
            )
        ).transpose(1, 2)
        cos, sin = position_embeddings
        return apply_rotary_pos_emb(query, key, cos, sin)

    def _validate_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        past_key_values: Any | None,
    ) -> None:
        if past_key_values is not None:
            raise ValueError("the TP4 benchmark owns its static cache internally")
        if attention_mask is not None:
            raise ValueError("the TP4 benchmark expects an unpadded prompt")
        if (
            hidden_states.ndim != 3
            or int(hidden_states.shape[1]) <= 0
            or int(hidden_states.shape[2]) != HIDDEN_SIZE
        ):
            raise ValueError("expected [batch, nonempty_tokens, 5120] hidden states")

    def clear_cache(self) -> None:
        self.key_cache = None
        self.graph_decode_position = None
        self._cache_length = 0

    def reset_cache(self) -> None:
        if self.key_cache is None:
            raise RuntimeError("configure_cache must be called before reset_cache")
        self._cache_length = 0

    def set_cache_length(self, length: int) -> None:
        """Restore a valid static-cache prefix for fixed-context benchmarks."""

        if self.key_cache is None:
            raise RuntimeError("configure_cache must be called before setting cache length")
        selected = int(length)
        capacity = int(self.key_cache.shape[2])
        if not 0 <= selected <= capacity:
            raise ValueError(
                f"cache length must be between 0 and {capacity}, got {selected}"
            )
        self._cache_length = selected

    def configure_graph_decode(self, position: Tensor) -> None:
        """Use one mutable device position with the full static cache shape."""

        if self.key_cache is None:
            raise RuntimeError("configure_cache must run before graph decode")
        if (
            position.ndim != 0
            or position.dtype != torch.int64
            or not position.is_cuda
            or position.device != self.key_cache.device
        ):
            raise ValueError(
                "graph decode position must be a CUDA int64 scalar on the cache device"
            )
        self.graph_decode_position = position

    def _uses_graph_decode(self, tokens: int) -> bool:
        return int(tokens) == 1 and self.graph_decode_position is not None

    @property
    def cache_length(self) -> int:
        return self._cache_length

    @property
    def cache_bytes(self) -> int:
        raise NotImplementedError


class Qwen3_32BTP4DenseAttention(_Qwen3_32BTP4AttentionBase):
    def __init__(self, base_attention: nn.Module) -> None:
        super().__init__(base_attention)
        self.v_proj = base_attention.v_proj
        self.o_proj = base_attention.o_proj
        self.register_buffer("value_cache", None, persistent=False)

    def configure_cache(self, *, batch_size: int, capacity: int) -> None:
        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        shape = (int(batch_size), KV_HEADS_PER_PROCESS, int(capacity))
        if min(shape) <= 0:
            raise ValueError("cache dimensions must be positive")
        self.key_cache = torch.empty(*shape, HEAD_DIM, device=device, dtype=dtype)
        self.value_cache = torch.empty(*shape, HEAD_DIM, device=device, dtype=dtype)
        self._cache_length = 0

    def clear_cache(self) -> None:
        super().clear_cache()
        self.value_cache = None

    @property
    def cache_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.key_cache, self.value_cache)
            if tensor is not None
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, None]:
        self._validate_forward(hidden_states, attention_mask, past_key_values)
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("configure_cache must be called before inference")
        batch, tokens = map(int, hidden_states.shape[:2])
        query, key = self._project_and_rotate(hidden_states, position_embeddings)
        value = self.v_proj(hidden_states).view(
            batch, tokens, KV_HEADS_PER_PROCESS, HEAD_DIM
        ).transpose(1, 2)
        if self._uses_graph_decode(tokens):
            if self.graph_decode_position is None:
                raise AssertionError("graph decode position is unavailable")
            index = self.graph_decode_position.view(1)
            self.key_cache.index_copy_(2, index, key)
            self.value_cache.index_copy_(2, index, value)
            local_output = compressed_v_decode_attention_triton(
                query,
                self.key_cache,
                self.value_cache,
                scale=self.scaling,
                valid_sequence_length=self.graph_decode_position + 1,
            )
        else:
            start = self._cache_length
            stop = start + tokens
            if tokens > 1 and start != 0:
                raise ValueError("chunked prefill is not supported")
            if stop > int(self.key_cache.shape[2]):
                raise RuntimeError("static KV cache capacity exceeded")
            self.key_cache[:, :, start:stop].copy_(key)
            self.value_cache[:, :, start:stop].copy_(value)
            self._cache_length = stop
            selected_key = self.key_cache[:, :, :stop]
            selected_value = self.value_cache[:, :, :stop]
            local_output = F.scaled_dot_product_attention(
                query,
                selected_key,
                selected_value,
                dropout_p=0.0,
                is_causal=tokens > 1,
                scale=self.scaling,
                enable_gqa=True,
            )
        token_major = local_output.transpose(1, 2).reshape(batch, tokens, -1)
        return self.o_proj(token_major), None


class Qwen3_32BTP4RaggedC1Attention(_Qwen3_32BTP4AttentionBase):
    def __init__(
        self,
        base_attention: nn.Module,
        factors: Qwen3_32BTP4C1FactorLayer,
        communicator: FeatureRaggedCommunicator,
        decode_attention_backend: str,
        wire_dtype: str,
        fp8_wire_source_scales: Tensor | None,
        allgather_backend: str,
        ipc_algorithm: str,
        ipc_channels: int,
    ) -> None:
        super().__init__(base_attention)
        if factors.layer_index != self.layer_idx:
            raise ValueError("C1 factors belong to another transformer layer")
        process_rank = dist.get_rank()
        source_start = process_rank * KV_HEADS_PER_PROCESS
        source_stop = source_start + KV_HEADS_PER_PROCESS
        self.local_source_ranks = factors.source_ranks[source_start:source_stop]
        self.local_wire_width = QUERY_HEADS_PER_KV_HEAD * sum(self.local_source_ranks)
        self.process_wire_widths = factors.process_wire_widths
        self.plan = StaticRaggedPlan.from_source_widths(self.process_wire_widths)
        self.communicator = communicator
        if communicator.world_size != TP_SIZE:
            raise ValueError("C1 communicator must use TP4")

        local_encoders = tuple(
            factors.encoders[source, :, : factors.source_ranks[source]].to(
                device=base_attention.v_proj.weight.device,
                dtype=base_attention.v_proj.weight.dtype,
            )
            for source in range(source_start, source_stop)
        )
        compact_weight, compact_bias = fold_ragged_local_c1_value_projection(
            base_attention.v_proj.weight,
            local_encoders,
            base_attention.v_proj.bias,
        )
        decoder_blocks: list[Tensor] = []
        for source, rank in enumerate(factors.source_ranks):
            head_start = source * QUERY_HEADS_PER_KV_HEAD
            head_stop = head_start + QUERY_HEADS_PER_KV_HEAD
            decoder_blocks.append(
                factors.decoders[head_start:head_stop, :rank]
                .reshape(QUERY_HEADS_PER_KV_HEAD * rank, HIDDEN_SIZE)
            )
        global_decoder = torch.cat(decoder_blocks, dim=0).to(
            device=base_attention.q_proj.weight.device,
            dtype=base_attention.q_proj.weight.dtype,
        )
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
            self.register_buffer("global_decoder", global_decoder.contiguous())
            self.register_buffer("global_decoder_fp8", None, persistent=True)
            self.register_buffer("wire_scale", None, persistent=True)
            self.register_buffer("fp8_arena_scale", None, persistent=True)
            self.register_buffer("fp8_decoder_scale", None, persistent=True)
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
            decoder_row_scales = torch.cat(
                tuple(
                    source_scales[source].expand(self.process_wire_widths[source])
                    for source in range(TP_SIZE)
                )
            ).to(device=global_decoder.device, dtype=torch.float32)
            scaled_decoder = global_decoder.float() * decoder_row_scales[:, None]
            fp8_decoder, fp8_decoder_scale = (
                quantize_e4m3_tensorwise_col_major(scaled_decoder)
            )
            self.register_buffer("global_decoder", None, persistent=True)
            self.register_buffer("global_decoder_fp8", fp8_decoder, persistent=True)
            self.register_buffer(
                "wire_scale",
                source_scales[process_rank].to(
                    device=global_decoder.device,
                    dtype=torch.float32,
                ),
                persistent=True,
            )
            self.register_buffer(
                "fp8_arena_scale",
                torch.ones((), device=global_decoder.device, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "fp8_decoder_scale",
                fp8_decoder_scale.to(device=global_decoder.device),
                persistent=True,
            )
        else:
            raise ValueError(
                "C1 wire dtype must be 'bfloat16' or 'float8_e4m3fn'"
            )
        self.wire_dtype = wire_dtype
        self.register_buffer("value_cache", None, persistent=False)
        self.factor_path = str(factors.path)
        self.factor_sha256 = factors.sha256
        self.source_ranks = factors.source_ranks
        if decode_attention_backend not in ("cuda", "triton"):
            raise ValueError("C1 decode attention backend must be 'cuda' or 'triton'")
        if decode_attention_backend == "cuda":
            if len(set(self.local_source_ranks)) != 1:
                raise ValueError(
                    "fused CUDA decode requires the two local C1 sources to share one rank"
                )
            if self.local_source_ranks[0] not in CUDA_VALUE_RANKS:
                raise ValueError(
                    "fused CUDA decode requires local source rank in "
                    f"{sorted(CUDA_VALUE_RANKS)}"
                )
        if allgather_backend not in (
            "feature_direct",
            "uniform_nccl",
            "uniform_ipc",
        ):
            raise ValueError(
                "C1 AllGather backend must be feature_direct, uniform_nccl, or uniform_ipc"
            )
        if allgather_backend != "feature_direct" and len(
            set(self.process_wire_widths)
        ) != 1:
            raise ValueError(
                "prepared uniform AllGather requires one common TP process width; "
                f"got {self.process_wire_widths}"
            )
        if ipc_algorithm not in (
            "auto",
            "fanout",
            "fanout_warp",
            "recursive_doubling",
            "ring",
        ):
            raise ValueError("unsupported C1 IPC AllGather algorithm")
        if allgather_backend != "uniform_ipc" and ipc_algorithm != "auto":
            raise ValueError("an explicit IPC algorithm requires uniform_ipc")
        selected_channels = int(ipc_channels)
        if selected_channels not in (0, 1, 2, 4, 8):
            raise ValueError("C1 IPC channels must be 0 (auto), 1, 2, 4, or 8")
        if allgather_backend != "uniform_ipc" and selected_channels != 0:
            raise ValueError("explicit IPC channels require uniform_ipc")
        if selected_channels > 1 and ipc_algorithm not in ("fanout", "ring"):
            raise ValueError("multiple IPC channels require fanout or ring")
        self.decode_attention_backend = decode_attention_backend
        self.allgather_backend = allgather_backend
        self.ipc_algorithm = ipc_algorithm
        self.ipc_channels = selected_channels
        self._decode_workspace: CompressedVDecodeWorkspace | None = None
        self._uniform_allgather: dict[
            tuple[int, torch.dtype, int], PreparedUniformAllGather
        ] = {}

    def configure_cache(self, *, batch_size: int, capacity: int) -> None:
        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        batch = int(batch_size)
        length = int(capacity)
        if batch <= 0 or length <= 0:
            raise ValueError("cache dimensions must be positive")
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
            length,
            sum(self.local_source_ranks),
            device=device,
            dtype=dtype,
        )
        self._cache_length = 0

    def clear_cache(self) -> None:
        super().clear_cache()
        self.value_cache = None
        self._decode_workspace = None
        self._uniform_allgather.clear()

    def configure_decode_workspace(
        self,
        workspace: CompressedVDecodeWorkspace,
    ) -> None:
        if self.decode_attention_backend != "cuda":
            raise ValueError("only CUDA C1 layers use a decode workspace")
        if workspace.query_heads != QUERY_HEADS_PER_PROCESS:
            raise ValueError("C1 decode workspace has the wrong query-head count")
        if workspace.max_value_dim < self.local_source_ranks[0]:
            raise ValueError("C1 decode workspace is narrower than this layer")
        self._decode_workspace = workspace

    def clear_uniform_allgather_plans(self) -> None:
        self._uniform_allgather.clear()

    def configure_uniform_allgather(
        self,
        *,
        tokens: int,
        dtype: torch.dtype,
    ) -> None:
        if self.allgather_backend == "feature_direct":
            return
        selected_tokens = int(tokens)
        stream = int(
            torch.cuda.current_stream(self.q_proj.weight.device).cuda_stream
        )
        key = (selected_tokens, dtype, stream)
        self._uniform_allgather[key] = self.communicator.prepare_uniform(
            self.plan,
            tokens=selected_tokens,
            dtype=dtype,
            backend=self.allgather_backend,
            ipc_algorithm=self.ipc_algorithm,
            ipc_channels=self.ipc_channels,
        )

    def _uniform_plan(
        self,
        *,
        tokens: int,
        dtype: torch.dtype,
    ) -> PreparedUniformAllGather | None:
        stream = int(
            torch.cuda.current_stream(self.q_proj.weight.device).cuda_stream
        )
        return self._uniform_allgather.get((int(tokens), dtype, stream))

    def _gather_coordinates(
        self,
        local: Tensor,
        *,
        tokens: int,
        local_is_feature_major: bool,
    ) -> Tensor:
        prepared = self._uniform_plan(tokens=tokens, dtype=local.dtype)
        if prepared is None:
            return self.communicator.gather(
                local,
                self.plan,
                backend="feature_direct",
                local_is_feature_major=local_is_feature_major,
            )
        if local_is_feature_major:
            if (
                self.wire_dtype == "bfloat16"
                and self.decode_attention_backend == "cuda"
                and local.ndim == 2
            ):
                return prepared.gather_inplace_fast()
            destination = prepared.local_feature_major_view()
            if local.data_ptr() != destination.data_ptr():
                destination.copy_(local, non_blocking=True)
            return prepared.gather_inplace_fast()
        return prepared.gather(local, local_is_feature_major=False)

    def _decode_gathered(self, arena: Tensor) -> Tensor:
        if self.wire_dtype == "bfloat16":
            if self.global_decoder is None:
                raise AssertionError("BF16 decoder is unavailable")
            return decode_feature_major(arena, self.global_decoder)
        if (
            self.global_decoder_fp8 is None
            or self.fp8_arena_scale is None
            or self.fp8_decoder_scale is None
        ):
            raise AssertionError("FP8 decoder state is unavailable")
        return decode_feature_major_e4m3(
            arena,
            self.global_decoder_fp8,
            arena_scale=self.fp8_arena_scale,
            decoder_scale=self.fp8_decoder_scale,
            out_dtype=self.q_proj.weight.dtype,
        )

    def set_wire_amax_observation(self, enabled: bool) -> None:
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

    def _fused_uniform_attention(
        self,
        query: Tensor,
        *,
        sequence_length: int,
        is_prefill: bool,
        valid_sequence_length: Tensor | None = None,
    ) -> Tensor:
        if self.value_cache is None or self.key_cache is None:
            raise RuntimeError("C1 cache is unavailable")
        rank = self.local_source_ranks[0]
        values = view_uniform_local_c1_value_heads(
            self.value_cache[:, :sequence_length],
            rank,
        )
        keys = self.key_cache[:, :, :sequence_length]
        if is_prefill:
            return compressed_v_prefill_attention(
                query,
                keys,
                values,
                scale=self.scaling,
            )
        if self.decode_attention_backend == "triton":
            return compressed_v_decode_attention_triton(
                query,
                keys,
                values,
                scale=self.scaling,
                valid_sequence_length=valid_sequence_length,
            )
        if self._decode_workspace is None:
            raise RuntimeError("configure_cache must allocate the CUDA decode workspace")
        feature_major_output = None
        if self.wire_dtype == "bfloat16":
            prepared = self._uniform_plan(tokens=int(query.shape[0]), dtype=query.dtype)
            if prepared is not None:
                feature_major_output = prepared.local_feature_major_view_fast()
            else:
                feature_major_output = self.communicator.direct_local_feature_major_view(
                    self.plan,
                    tokens=int(query.shape[0]),
                    dtype=query.dtype,
                )
        return compressed_v_decode_attention_cuda_tuned(
            query,
            keys,
            values,
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
                local_output = quantize_e4m3_static(
                    local_output,
                    self.wire_scale,
                ).view(torch.uint8)
            arena = self._gather_coordinates(
                local_output,
                tokens=batch,
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
        arena = self._gather_coordinates(
            local_coordinates,
            tokens=rows,
            local_is_feature_major=False,
        )
        decoded = self._decode_gathered(arena)
        return decoded.reshape(batch, tokens, HIDDEN_SIZE)

    @property
    def cache_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.key_cache, self.value_cache)
            if tensor is not None
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, None]:
        self._validate_forward(hidden_states, attention_mask, past_key_values)
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("configure_cache must be called before inference")
        batch, tokens = map(int, hidden_states.shape[:2])
        query, key = self._project_and_rotate(hidden_states, position_embeddings)
        value = F.linear(
            hidden_states, self.compact_v_weight, self.compact_v_bias
        )
        valid_sequence_length: Tensor | None = None
        if self._uses_graph_decode(tokens):
            if self.graph_decode_position is None:
                raise AssertionError("graph decode position is unavailable")
            index = self.graph_decode_position.view(1)
            self.key_cache.index_copy_(2, index, key)
            self.value_cache.index_copy_(1, index, value)
            sequence_length = int(self.key_cache.shape[2])
            valid_sequence_length = self.graph_decode_position + 1
        else:
            start = self._cache_length
            stop = start + tokens
            if tokens > 1 and start != 0:
                raise ValueError("chunked prefill is not supported")
            if stop > int(self.key_cache.shape[2]):
                raise RuntimeError("static KV cache capacity exceeded")
            self.key_cache[:, :, start:stop].copy_(key)
            self.value_cache[:, start:stop].copy_(value)
            self._cache_length = stop
            sequence_length = stop

        if len(set(self.local_source_ranks)) == 1:
            local_output = self._fused_uniform_attention(
                query,
                sequence_length=sequence_length,
                is_prefill=tokens > 1,
                valid_sequence_length=valid_sequence_length,
            )
            return self._project_output(local_output), None

        source_outputs: list[Tensor] = []
        value_offset = 0
        for local_source, rank in enumerate(self.local_source_ranks):
            query_start = local_source * QUERY_HEADS_PER_KV_HEAD
            query_stop = query_start + QUERY_HEADS_PER_KV_HEAD
            source_query = query[:, query_start:query_stop]
            source_key = self.key_cache[
                :, local_source : local_source + 1, :sequence_length
            ]
            source_value = self.value_cache[
                :, :sequence_length, value_offset : value_offset + rank
            ].unsqueeze(1)
            if tokens > 1:
                source_output = compressed_v_prefill_attention(
                    source_query,
                    source_key,
                    source_value,
                    scale=self.scaling,
                )
            else:
                source_output = compressed_v_decode_attention_triton(
                    source_query,
                    source_key,
                    source_value,
                    scale=self.scaling,
                    valid_sequence_length=valid_sequence_length,
                )
            source_outputs.append(
                source_output.transpose(1, 2).reshape(
                    batch, tokens, QUERY_HEADS_PER_KV_HEAD * rank
                )
            )
            value_offset += rank
        local_output = torch.cat(source_outputs, dim=-1).view(
            batch,
            tokens,
            1,
            self.local_wire_width,
        ).transpose(1, 2)
        return self._project_output(local_output), None


Qwen3_32BTP4Attention = Qwen3_32BTP4DenseAttention | Qwen3_32BTP4RaggedC1Attention


def configure_qwen3_32b_tp4_caches(
    modules: Sequence[Qwen3_32BTP4Attention],
    *,
    batch_size: int,
    capacity: int,
    max_forward_tokens: int,
) -> int:
    """Allocate static KV storage and shared decode/collective arenas."""

    batch = int(batch_size)
    forward_tokens = int(max_forward_tokens)
    for module in modules:
        module.configure_cache(batch_size=batch, capacity=capacity)
    c1_modules = tuple(
        module
        for module in modules
        if isinstance(module, Qwen3_32BTP4RaggedC1Attention)
    )
    for module in c1_modules:
        module.clear_uniform_allgather_plans()
    decode_workspace: CompressedVDecodeWorkspace | None = None
    if c1_modules:
        communicators = {id(module.communicator): module.communicator for module in c1_modules}
        if len(communicators) != 1:
            raise RuntimeError("all C1 layers must share one communicator")
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
                max_value_dim=max(
                    module.local_source_ranks[0] for module in cuda_modules
                ),
                dtype=reference.dtype,
                device=reference.device,
            )
            for module in cuda_modules:
                module.configure_decode_workspace(decode_workspace)

        communicator = next(iter(communicators.values()))
        wire_dtypes = {module.wire_dtype for module in c1_modules}
        if len(wire_dtypes) != 1:
            raise RuntimeError("Qwen3-32B C1 layers must share one wire dtype")
        wire_dtype = (
            torch.uint8
            if next(iter(wire_dtypes)) == "float8_e4m3fn"
            else reference.dtype
        )
        allgather_backends = {module.allgather_backend for module in c1_modules}
        if len(allgather_backends) != 1:
            raise RuntimeError("Qwen3-32B C1 layers must share one AllGather backend")
        allgather_backend = next(iter(allgather_backends))
        ipc_algorithms = {module.ipc_algorithm for module in c1_modules}
        if len(ipc_algorithms) != 1:
            raise RuntimeError("Qwen3-32B C1 layers must share one IPC algorithm")
        ipc_channels = {module.ipc_channels for module in c1_modules}
        if len(ipc_channels) != 1:
            raise RuntimeError("Qwen3-32B C1 layers must share one IPC channel count")
        rows = batch * forward_tokens
        maximum_width = max(module.plan.total_width for module in c1_modules)
        if allgather_backend in ("feature_direct", "uniform_nccl"):
            communicator.configure_direct_workspace(
                tokens=rows,
                max_total_width=maximum_width,
                dtype=wire_dtype,
            )
        elif allgather_backend == "uniform_ipc":
            if forward_tokens != 1:
                raise RuntimeError(
                    "uniform_ipc v1 is decode-only and requires max_forward_tokens=1; "
                    "use uniform_nccl for prefill+decode serving"
                )
            communicator.prepare_ipc(
                tokens=rows,
                max_total_width=maximum_width,
                dtype=wire_dtype,
            )
        else:
            raise AssertionError(f"unreachable AllGather backend {allgather_backend}")

        if allgather_backend != "feature_direct":
            token_counts = {rows}
            if forward_tokens != 1:
                token_counts.add(batch)
            for module in c1_modules:
                for tokens in sorted(token_counts):
                    module.configure_uniform_allgather(
                        tokens=tokens,
                        dtype=wire_dtype,
                    )
    return sum(module.cache_bytes for module in modules) + (
        0 if decode_workspace is None else decode_workspace.nbytes
    )


def install_qwen3_32b_tp4_attention(
    model: nn.Module,
    *,
    factor_dir: str | Path | None,
    c1_decode_attention_backend: str | None = None,
    c1_wire_dtype: str = "bfloat16",
    c1_fp8_wire_scales: str | Path | None = None,
    c1_allgather_backend: str = "feature_direct",
    c1_ipc_algorithm: str = "auto",
    c1_ipc_channels: int = 0,
) -> tuple[Qwen3_32BTP4Attention, ...]:
    """Install dense or compact-Value C1 attention in all Qwen3-32B layers."""

    if factor_dir is None:
        if c1_decode_attention_backend is not None:
            raise ValueError("dense attention must not select a C1 decode backend")
        if c1_wire_dtype != "bfloat16":
            raise ValueError("dense attention must use a BF16 wire")
        if c1_fp8_wire_scales is not None:
            raise ValueError("dense attention must not receive FP8 wire scales")
        if c1_allgather_backend != "feature_direct":
            raise ValueError("dense attention must not select a C1 AllGather backend")
        if c1_ipc_algorithm != "auto" or int(c1_ipc_channels) != 0:
            raise ValueError("dense attention must not select CUDA-IPC controls")
    elif c1_decode_attention_backend not in ("cuda", "triton"):
        raise ValueError("C1 attention requires an explicit 'cuda' or 'triton' backend")
    elif c1_wire_dtype not in ("bfloat16", "float8_e4m3fn"):
        raise ValueError("unsupported C1 wire dtype")
    elif c1_wire_dtype == "float8_e4m3fn" and c1_fp8_wire_scales is None:
        raise ValueError("FP8 C1 requires calibrated --c1-fp8-wire-scales")
    elif c1_wire_dtype == "bfloat16" and c1_fp8_wire_scales is not None:
        raise ValueError("BF16 C1 wire must not receive FP8 scales")

    layers: Sequence[nn.Module] = model.model.layers
    if len(layers) != NUM_LAYERS:
        raise ValueError(f"expected {NUM_LAYERS} layers, got {len(layers)}")
    root = None if factor_dir is None else Path(factor_dir).expanduser().resolve()
    manifest = (
        None if root is None else load_qwen3_32b_tp4_c1_manifest(root)
    )
    wire_scales = (
        None
        if c1_fp8_wire_scales is None
        else load_qwen3_32b_tp4_fp8_wire_scales(c1_fp8_wire_scales)
    )
    communicator = (
        None
        if root is None
        else FeatureRaggedCommunicator.from_distributed(
            device=model.model.embed_tokens.weight.device
        )
    )
    installed: list[Qwen3_32BTP4Attention] = []
    for layer_index, layer in enumerate(layers):
        if root is None:
            replacement: Qwen3_32BTP4Attention = Qwen3_32BTP4DenseAttention(
                layer.self_attn
            )
        else:
            assert manifest is not None and communicator is not None
            factors = load_qwen3_32b_tp4_c1_factor_layer(
                root, layer_index, manifest=manifest
            )
            replacement = Qwen3_32BTP4RaggedC1Attention(
                layer.self_attn,
                factors,
                communicator,
                c1_decode_attention_backend,
                c1_wire_dtype,
                None if wire_scales is None else wire_scales[layer_index],
                c1_allgather_backend,
                c1_ipc_algorithm,
                c1_ipc_channels,
            )
        replacement.eval()
        layer.self_attn = replacement
        installed.append(replacement)
    return tuple(installed)


def close_qwen3_32b_tp4_communicator(
    modules: Sequence[Qwen3_32BTP4Attention],
) -> None:
    for module in modules:
        if isinstance(module, Qwen3_32BTP4RaggedC1Attention):
            module.clear_uniform_allgather_plans()
    communicators = {
        id(module.communicator): module.communicator
        for module in modules
        if isinstance(module, Qwen3_32BTP4RaggedC1Attention)
    }
    for communicator in communicators.values():
        communicator.close()


__all__ = [
    "HEAD_DIM",
    "HIDDEN_SIZE",
    "KV_HEADS_PER_PROCESS",
    "NUM_KV_HEADS",
    "NUM_LAYERS",
    "NUM_QUERY_HEADS",
    "QUERY_HEADS_PER_KV_HEAD",
    "QUERY_HEADS_PER_PROCESS",
    "Qwen3_32BTP4C1FactorLayer",
    "Qwen3_32BTP4DenseAttention",
    "Qwen3_32BTP4RaggedC1Attention",
    "TP_SIZE",
    "close_qwen3_32b_tp4_communicator",
    "configure_qwen3_32b_tp4_caches",
    "file_sha256",
    "fold_ragged_local_c1_value_projection",
    "install_qwen3_32b_tp4_attention",
    "load_qwen3_32b_tp4_c1_factor_layer",
    "load_qwen3_32b_tp4_c1_manifest",
    "load_qwen3_32b_tp4_fp8_wire_scales",
    "view_uniform_local_c1_value_heads",
]
