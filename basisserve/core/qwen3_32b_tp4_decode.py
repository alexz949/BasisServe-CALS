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
    compressed_v_decode_attention_triton,
    compressed_v_prefill_attention,
)
from basisserve.kernels.feature_ragged_allgather import (
    FeatureRaggedCommunicator,
    decode_feature_major,
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


def _load_manifest(root: Path) -> Mapping[str, Any]:
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
    result = _load_manifest(root) if manifest is None else manifest
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
    folded_weights: list[Tensor] = []
    folded_biases: list[Tensor] = []
    for local_source, encoder in enumerate(encoders):
        if encoder.ndim != 2 or int(encoder.shape[0]) != HEAD_DIM:
            raise ValueError("each local encoder must have shape [128, source_rank]")
        start = local_source * HEAD_DIM
        stop = start + HEAD_DIM
        folded_weights.append(
            encoder.float().T @ dense_weight[start:stop].detach().float()
        )
        if dense_bias is not None:
            folded_biases.append(
                encoder.float().T @ dense_bias[start:stop].detach().float()
            )
    target_dtype = dense_weight.dtype
    return (
        torch.cat(folded_weights, dim=0).to(dtype=target_dtype).contiguous(),
        None
        if dense_bias is None
        else torch.cat(folded_biases, dim=0).to(dtype=target_dtype).contiguous(),
    )


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
        self._cache_length = 0

    def reset_cache(self) -> None:
        if self.key_cache is None:
            raise RuntimeError("configure_cache must be called before reset_cache")
        self._cache_length = 0

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
        start = self._cache_length
        stop = start + tokens
        if tokens > 1 and start != 0:
            raise ValueError("chunked prefill is not supported")
        if stop > int(self.key_cache.shape[2]):
            raise RuntimeError("static KV cache capacity exceeded")
        self.key_cache[:, :, start:stop].copy_(key)
        self.value_cache[:, :, start:stop].copy_(value)
        self._cache_length = stop
        local_output = F.scaled_dot_product_attention(
            query,
            self.key_cache[:, :, :stop],
            self.value_cache[:, :, :stop],
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
        self.register_buffer("compact_v_weight", compact_weight)
        self.register_buffer("compact_v_bias", compact_bias)
        self.register_buffer("global_decoder", global_decoder.contiguous())
        self.register_buffer("value_cache", None, persistent=False)
        self.factor_path = str(factors.path)
        self.factor_sha256 = factors.sha256
        self.source_ranks = factors.source_ranks

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
        start = self._cache_length
        stop = start + tokens
        if tokens > 1 and start != 0:
            raise ValueError("chunked prefill is not supported")
        if stop > int(self.key_cache.shape[2]):
            raise RuntimeError("static KV cache capacity exceeded")
        self.key_cache[:, :, start:stop].copy_(key)
        self.value_cache[:, start:stop].copy_(value)
        self._cache_length = stop

        source_outputs: list[Tensor] = []
        value_offset = 0
        for local_source, rank in enumerate(self.local_source_ranks):
            query_start = local_source * QUERY_HEADS_PER_KV_HEAD
            query_stop = query_start + QUERY_HEADS_PER_KV_HEAD
            source_query = query[:, query_start:query_stop]
            source_key = self.key_cache[:, local_source : local_source + 1, :stop]
            source_value = self.value_cache[
                :, :stop, value_offset : value_offset + rank
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
                )
            source_outputs.append(
                source_output.transpose(1, 2).reshape(
                    batch, tokens, QUERY_HEADS_PER_KV_HEAD * rank
                )
            )
            value_offset += rank
        local_coordinates = torch.cat(source_outputs, dim=-1).reshape(
            batch * tokens, self.local_wire_width
        )
        arena = self.communicator.gather(
            local_coordinates,
            self.plan,
            backend="feature_direct",
        )
        decoded = decode_feature_major(arena, self.global_decoder)
        return decoded.reshape(batch, tokens, HIDDEN_SIZE), None


Qwen3_32BTP4Attention = Qwen3_32BTP4DenseAttention | Qwen3_32BTP4RaggedC1Attention


def configure_qwen3_32b_tp4_caches(
    modules: Sequence[Qwen3_32BTP4Attention],
    *,
    batch_size: int,
    capacity: int,
    max_forward_tokens: int,
) -> int:
    """Allocate static KV storage and the shared ragged receive arena."""

    for module in modules:
        module.configure_cache(batch_size=batch_size, capacity=capacity)
    c1_modules = tuple(
        module
        for module in modules
        if isinstance(module, Qwen3_32BTP4RaggedC1Attention)
    )
    if c1_modules:
        communicators = {id(module.communicator): module.communicator for module in c1_modules}
        if len(communicators) != 1:
            raise RuntimeError("all C1 layers must share one communicator")
        reference = c1_modules[0].q_proj.weight
        next(iter(communicators.values())).configure_direct_workspace(
            tokens=int(batch_size) * int(max_forward_tokens),
            max_total_width=max(module.plan.total_width for module in c1_modules),
            dtype=reference.dtype,
        )
    return sum(module.cache_bytes for module in modules)


def install_qwen3_32b_tp4_attention(
    model: nn.Module,
    *,
    factor_dir: str | Path | None,
) -> tuple[Qwen3_32BTP4Attention, ...]:
    """Install dense or BF16 ragged-C1 attention in all Qwen3-32B layers."""

    layers: Sequence[nn.Module] = model.model.layers
    if len(layers) != NUM_LAYERS:
        raise ValueError(f"expected {NUM_LAYERS} layers, got {len(layers)}")
    root = None if factor_dir is None else Path(factor_dir).expanduser().resolve()
    manifest = None if root is None else _load_manifest(root)
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
                layer.self_attn, factors, communicator
            )
        replacement.eval()
        layer.self_attn = replacement
        installed.append(replacement)
    return tuple(installed)


def close_qwen3_32b_tp4_communicator(
    modules: Sequence[Qwen3_32BTP4Attention],
) -> None:
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
]
