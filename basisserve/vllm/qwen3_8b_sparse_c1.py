"""Qwen3-8B C1-V80 with native vLLM sparse-Key decode.

The paged cache stores exact post-RoPE K128, C1 Value coordinates V80, and a
32-dimensional routing sidecar.  Its Value storage is padded with 16 zero
coordinates to the V128 ABI required by vLLM FlashAttention prefill.
Single-token decode selects exact-Key support with either the conditional
Base16+R8 Page32 router or the repository-faithful Loki R32 token router.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.vllm.qwen3_8b_sparse_c1_backend import (
    CACHE_PADDING_DIM,
    CACHE_VALUE_HEAD_SIZE,
    BasisServeQwen3_8BSparseC1Backend,
)

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader


MODEL_ARCHITECTURE = "BasisServeQwen3_8BSparseC1ForCausalLM"
VALUE_FORMAT = "basisserve.qwen3_8b.gqa_c1_joint.v1"
CONDITIONAL_FORMAT = "basisserve.qwen3_8b.v80_base16_r8_nonsink_page32.v1"
LOKI_FORMAT = "basisserve.qwen3_8b.loki_key_pca.v1"
NUM_LAYERS = 36
HIDDEN_SIZE = 4096
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
VALUE_RANK = 80
ROUTING_RANK = 32
BASE_RANK = 16
RESIDUAL_RANK = 8

_CONFIG_VALUE_DIR = "basisserve_value_factor_dir"
_CONFIG_VALUE_RESULT_SHA256 = "basisserve_value_result_sha256"
_CONFIG_ROUTER_MODE = "basisserve_router_mode"
_CONFIG_ROUTER_DIR = "basisserve_router_factor_dir"
_CONFIG_ROUTER_FINGERPRINT = "basisserve_router_fingerprint"
_CONFIG_TOKEN_BUDGET = "basisserve_token_budget"
_CONFIG_PINNED_PREFIX_PAGES = "basisserve_pinned_prefix_pages"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(config: Any, name: str) -> str:
    value = getattr(config, name, None)
    assert isinstance(value, str) and value.strip()
    return value.strip()


def _checkpoint_fingerprint(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def router_checkpoint_fingerprint(root: Path, mode: str) -> str:
    """Return the authenticated file-set fingerprint used by HF overrides."""

    resolved = root.expanduser().resolve()
    if mode == "conditional":
        paths = list(resolved.rglob("result.json")) + list(
            resolved.rglob("layer_*.safetensors")
        )
    else:
        paths = [resolved / "result.json", resolved / "factors.safetensors"]
    assert paths and all(path.is_file() for path in paths)
    return _checkpoint_fingerprint(paths, resolved)


def _load_value_layer(
    root: Path, manifest: dict[str, Any], layer: int
) -> dict[str, Tensor]:
    record = manifest["artifacts"][str(layer)]
    path = root / record["file"]
    assert path.is_file() and _file_sha256(path) == record["sha256"]
    tensors = load_file(str(path), device="cpu")
    assert set(tensors) == {"value_coordinate_encoders", "head_output_decoders"}
    assert tuple(tensors["value_coordinate_encoders"].shape) == (
        NUM_KV_HEADS,
        HEAD_DIM,
        VALUE_RANK,
    )
    assert tuple(tensors["head_output_decoders"].shape) == (
        NUM_QUERY_HEADS,
        VALUE_RANK,
        HIDDEN_SIZE,
    )
    return tensors


def _conditional_layer_paths(root: Path) -> tuple[Path, ...]:
    indexed = {
        int(path.stem.removeprefix("layer_")): path
        for path in root.rglob("layer_*.safetensors")
    }
    assert set(indexed) == set(range(NUM_LAYERS))
    return tuple(indexed[layer] for layer in range(NUM_LAYERS))


def fold_value_projection(dense_value: Tensor, encoders: Tensor) -> Tensor:
    """Fold eight dense V128 projection blocks into eight V80 blocks."""

    hidden_size = int(dense_value.shape[1])
    assert tuple(dense_value.shape) == (NUM_KV_HEADS * HEAD_DIM, hidden_size)
    assert tuple(encoders.shape) == (NUM_KV_HEADS, HEAD_DIM, VALUE_RANK)
    return torch.bmm(
        encoders.transpose(1, 2).float(),
        dense_value.reshape(NUM_KV_HEADS, HEAD_DIM, hidden_size).float(),
    )


class BasisServeSparseC1QKVParallelLinear(QKVParallelLinear):
    """Load dense Q/K and fold dense V into V80; routing aux is dynamic."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        value_encoders: Tensor,
        prefix: str,
    ) -> None:
        super().__init__(
            hidden_size,
            head_size,
            total_num_heads,
            total_num_kv_heads,
            bias=False,
            quant_config=None,
            prefix=prefix,
            v_head_size=CACHE_VALUE_HEAD_SIZE,
        )
        assert tuple(value_encoders.shape) == (
            NUM_KV_HEADS,
            HEAD_DIM,
            VALUE_RANK,
        )
        self.register_buffer(
            "value_encoders",
            value_encoders.detach().to(device=self.weight.device).contiguous(),
            persistent=False,
        )

    def _load_value_weight(self, param: nn.Parameter, loaded_weight: Tensor) -> None:
        assert param is self.weight and getattr(param, "output_dim", None) == 0
        assert self.tp_size == 1 and self.num_kv_heads == NUM_KV_HEADS
        assert tuple(loaded_weight.shape) == (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)
        compact = fold_value_projection(
            loaded_weight.to(device=self.value_encoders.device),
            self.value_encoders,
        )
        value_offset = (self.num_heads + self.num_kv_heads) * self.head_size
        destination = param.data.narrow(
            0,
            value_offset,
            self.num_kv_heads * self.v_head_size,
        ).view(NUM_KV_HEADS, CACHE_VALUE_HEAD_SIZE, HIDDEN_SIZE)
        destination.zero_()
        destination[:, :VALUE_RANK].copy_(compact.to(dtype=destination.dtype))

    def weight_loader_v2(
        self,
        param: nn.Parameter,
        loaded_weight: Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        assert loaded_shard_id is not None
        if loaded_shard_id == "v":
            self._load_value_weight(param, loaded_weight)
        else:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        assert loaded_shard_id is not None
        if loaded_shard_id == "v":
            self._load_value_weight(param, loaded_weight)
        else:
            super().weight_loader(param, loaded_weight, loaded_shard_id)


class BasisServeQwen3_8BSparseC1Attention(nn.Module):
    """C1-V80 projection/closure around one native vLLM sparse cache."""

    def __init__(
        self,
        base_attention: Qwen3Attention,
        *,
        value_factors: dict[str, Tensor],
        router_factors: dict[str, Tensor],
        router_mode: str,
        token_budget: int,
        pinned_prefix_pages: int,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__()
        assert base_attention.dual_chunk_attention_config is None
        assert base_attention.num_heads == NUM_QUERY_HEADS
        assert base_attention.num_kv_heads == NUM_KV_HEADS
        self.hidden_size = base_attention.hidden_size
        self.total_num_heads = base_attention.total_num_heads
        self.num_heads = base_attention.num_heads
        self.total_num_kv_heads = base_attention.total_num_kv_heads
        self.num_kv_heads = base_attention.num_kv_heads
        self.head_dim = base_attention.head_dim
        self.q_size = base_attention.q_size
        self.kv_size = base_attention.kv_size
        self.cache_value_size = NUM_KV_HEADS * CACHE_VALUE_HEAD_SIZE
        self.scaling = base_attention.scaling
        self.router_mode = router_mode

        attention_prefix = base_attention.attn.layer_name
        assert attention_prefix.endswith(".attn")
        self_attention_prefix = attention_prefix[: -len(".attn")]
        static_context = vllm_config.compilation_config.static_forward_context
        assert static_context.pop(attention_prefix, None) is base_attention.attn

        reference = base_attention.qkv_proj.weight
        dtype = reference.dtype
        device = reference.device
        value_encoders = value_factors["value_coordinate_encoders"].to(
            device=device,
            dtype=dtype,
        )
        decoders = value_factors["head_output_decoders"].to(
            device=device,
            dtype=dtype,
        )
        self.qkv_proj = BasisServeSparseC1QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            value_encoders=value_encoders,
            prefix=f"{self_attention_prefix}.qkv_proj",
        )
        self.rotary_emb = base_attention.rotary_emb
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=None,
            prefix=attention_prefix,
            attn_backend=BasisServeQwen3_8BSparseC1Backend,
            head_size_v=CACHE_VALUE_HEAD_SIZE,
        )
        self.register_buffer(
            "decoder_weight",
            decoders.permute(2, 0, 1).reshape(HIDDEN_SIZE, -1).contiguous(),
            persistent=False,
        )

        self.attn._basisserve_router_mode = router_mode
        self.attn._basisserve_token_budget = int(token_budget)
        self.attn._basisserve_pinned_prefix_pages = int(pinned_prefix_pages)
        object.__setattr__(
            self.attn,
            "_basisserve_rotary_emb",
            self.rotary_emb,
        )
        self.attn.register_buffer(
            "_basisserve_routing_queries",
            torch.zeros((), dtype=torch.int64, device=device),
            persistent=False,
        )
        self.attn.register_buffer(
            "_basisserve_physical_tokens",
            torch.zeros((), dtype=torch.int64, device=device),
            persistent=False,
        )
        self.attn.register_buffer(
            "_basisserve_logical_tokens",
            torch.zeros((), dtype=torch.int64, device=device),
            persistent=False,
        )

        if router_mode == "conditional":
            self.register_buffer(
                "base_left",
                router_factors["base_left_b16"].to(device=device, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                "residual_encoder",
                router_factors["residual_encoder_b16_r8"].to(
                    device=device,
                    dtype=dtype,
                ),
                persistent=False,
            )
            for name, key in (
                ("_basisserve_base_right", "base_right_b16"),
                ("_basisserve_base_bias", "base_bias_b16"),
                ("_basisserve_residual_query", "residual_query_b16_r8"),
            ):
                self.attn.register_buffer(
                    name,
                    router_factors[key].to(device=device, dtype=dtype),
                    persistent=False,
                )
        else:
            projector = router_factors["key_projector"].to(
                device=device,
                dtype=dtype,
            )
            assert tuple(projector.shape) == (
                NUM_KV_HEADS,
                HEAD_DIM,
                ROUTING_RANK,
            )
            self.register_buffer(
                "loki_key_projector",
                projector,
                persistent=False,
            )
            self.attn.register_buffer(
                "_basisserve_loki_query_projector",
                projector.repeat_interleave(
                    NUM_QUERY_HEADS // NUM_KV_HEADS,
                    dim=0,
                ).contiguous(),
                persistent=False,
            )

    def _conditional_aux(self, positions: Tensor, key: Tensor, value: Tensor) -> Tensor:
        tokens = int(value.shape[0])
        value_heads = value.view(tokens, NUM_KV_HEADS, CACHE_VALUE_HEAD_SIZE)[
            ..., :VALUE_RANK
        ]
        base_code = torch.einsum("thv,hvr->thr", value_heads, self.base_left)
        predicted_pre = torch.einsum(
            "thr,hrd->thd",
            base_code,
            self.attn._basisserve_base_right,
        ).add(self.attn._basisserve_base_bias)
        predicted_post, _ = self.rotary_emb(
            positions,
            predicted_pre.reshape(tokens, -1),
            None,
        )
        residual = key.view(tokens, NUM_KV_HEADS, HEAD_DIM) - predicted_post.view(
            tokens,
            NUM_KV_HEADS,
            HEAD_DIM,
        )
        residual_code = torch.einsum(
            "thd,hdr->thr",
            residual,
            self.residual_encoder,
        )
        padding = torch.zeros(
            tokens,
            NUM_KV_HEADS,
            ROUTING_RANK - BASE_RANK - RESIDUAL_RANK,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat((base_code, residual_code, padding), dim=-1)

    def _loki_aux(self, key: Tensor) -> Tensor:
        key_heads = key.view(-1, NUM_KV_HEADS, HEAD_DIM)
        return torch.einsum(
            "thd,hdr->thr",
            key_heads,
            self.loki_key_projector,
        )

    def forward(self, positions: Tensor, hidden_states: Tensor) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, value = qkv.split(
            [self.q_size, self.kv_size, self.cache_value_size],
            dim=-1,
        )
        q_heads = q.view(-1, NUM_QUERY_HEADS, HEAD_DIM)
        k_heads = k.view(-1, NUM_KV_HEADS, HEAD_DIM)
        q = self.q_norm(q_heads).view(q.shape)
        k = self.k_norm(k_heads).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        value_heads = value.view(-1, NUM_KV_HEADS, CACHE_VALUE_HEAD_SIZE)
        payload = value_heads[..., :VALUE_RANK]
        aux = (
            self._conditional_aux(positions, k, value)
            if self.router_mode == "conditional"
            else self._loki_aux(k)
        )
        cache_padding = value_heads[..., -CACHE_PADDING_DIM:]
        cache_value = torch.cat((payload, aux, cache_padding), dim=-1).reshape(
            -1,
            self.cache_value_size,
        )
        raw_output = self.attn(q, k, cache_value)
        coordinates = raw_output.view(-1, NUM_QUERY_HEADS, CACHE_VALUE_HEAD_SIZE)[
            ..., :VALUE_RANK
        ].reshape(-1, NUM_QUERY_HEADS * VALUE_RANK)
        return F.linear(coordinates, self.decoder_weight)

    def routing_statistics(self) -> dict[str, int]:
        return {
            "queries": int(self.attn._basisserve_routing_queries.item()),
            "physical_tokens": int(self.attn._basisserve_physical_tokens.item()),
            "logical_tokens": int(self.attn._basisserve_logical_tokens.item()),
        }


class BasisServeQwen3_8BSparseC1ForCausalLM(Qwen3ForCausalLM):
    """TP1 Qwen3-8B model with V80 storage and sparse exact-Key decode."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        observed = (
            int(config.hidden_size),
            int(config.num_attention_heads),
            int(config.num_key_value_heads),
            int(getattr(config, "head_dim", 0)),
            int(config.num_hidden_layers),
        )
        assert observed == (
            HIDDEN_SIZE,
            NUM_QUERY_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            NUM_LAYERS,
        )
        assert get_tensor_model_parallel_world_size() == 1
        assert get_pp_group().world_size == 1
        assert vllm_config.parallel_config.decode_context_parallel_size == 1
        assert vllm_config.quant_config is None and vllm_config.lora_config is None
        assert vllm_config.model_config.dtype == torch.bfloat16
        assert vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")
        assert int(vllm_config.cache_config.block_size) == 32

        value_root = Path(_required_string(config, _CONFIG_VALUE_DIR)).resolve()
        value_result_path = value_root / "results.json"
        assert _file_sha256(value_result_path) == _required_string(
            config,
            _CONFIG_VALUE_RESULT_SHA256,
        )
        value_manifest = json.loads(value_result_path.read_text(encoding="utf-8"))
        assert value_manifest.get("format") == VALUE_FORMAT
        assert value_manifest.get("status") == "complete"
        fit = value_manifest["fit_config"]
        assert int(fit["cache_rank_per_head"]) == VALUE_RANK
        assert int(fit["fit_windows"]) == 32
        assert int(fit["positions_per_window"]) == 32768

        router_mode = _required_string(config, _CONFIG_ROUTER_MODE)
        assert router_mode in {"conditional", "loki"}
        router_root = Path(_required_string(config, _CONFIG_ROUTER_DIR)).resolve()
        assert router_checkpoint_fingerprint(
            router_root, router_mode
        ) == _required_string(
            config,
            _CONFIG_ROUTER_FINGERPRINT,
        )
        token_budget = int(getattr(config, _CONFIG_TOKEN_BUDGET))
        pinned_prefix_pages = int(getattr(config, _CONFIG_PINNED_PREFIX_PAGES))
        assert token_budget == 2048
        assert pinned_prefix_pages == (1 if router_mode == "conditional" else 0)

        conditional_paths: tuple[Path, ...] = ()
        loki_projectors: Tensor | None = None
        if router_mode == "conditional":
            conditional_paths = _conditional_layer_paths(router_root)
            for result_path in router_root.rglob("result.json"):
                result = json.loads(result_path.read_text(encoding="utf-8"))
                assert result.get("format") == CONDITIONAL_FORMAT
                assert result.get("status") == "complete"
                assert int(result["protocol"]["page_size"]) == 32
                assert (
                    int(result["protocol"]["physical_token_budget_per_kv_group"])
                    == token_budget
                )
        else:
            result_path = router_root / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            assert result.get("format") == LOKI_FORMAT
            assert result.get("status") == "complete"
            factor_path = router_root / result["artifacts"]["factors"]["file"]
            assert _file_sha256(factor_path) == result["artifacts"]["factors"]["sha256"]
            loki_projectors = load_file(str(factor_path), device="cpu")["key_projector"]
            assert tuple(loki_projectors.shape) == (
                NUM_LAYERS,
                NUM_KV_HEADS,
                HEAD_DIM,
                ROUTING_RANK,
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer_index, layer in enumerate(self.model.layers):
            value_factors = _load_value_layer(
                value_root,
                value_manifest,
                layer_index,
            )
            router_factors = (
                load_file(str(conditional_paths[layer_index]), device="cpu")
                if router_mode == "conditional"
                else {"key_projector": loki_projectors[layer_index]}
            )
            layer.self_attn = BasisServeQwen3_8BSparseC1Attention(
                layer.self_attn,
                value_factors=value_factors,
                router_factors=router_factors,
                router_mode=router_mode,
                token_budget=token_budget,
                pinned_prefix_pages=pinned_prefix_pages,
                vllm_config=vllm_config,
            )

        self.basisserve_value_factor_dir = str(value_root)
        self.basisserve_router_factor_dir = str(router_root)
        self.basisserve_router_mode = router_mode

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        skip_prefixes = [
            f"model.layers.{layer}.self_attn.o_proj." for layer in range(NUM_LAYERS)
        ]
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")
        return AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        ).load_weights(weights)

    def routing_statistics(self) -> dict[str, Any]:
        per_layer = [
            layer.self_attn.routing_statistics() for layer in self.model.layers
        ]
        totals = {
            key: sum(record[key] for record in per_layer)
            for key in ("queries", "physical_tokens", "logical_tokens")
        }
        totals["physical_tokens_per_layer_query"] = (
            totals["physical_tokens"] / totals["queries"] if totals["queries"] else 0.0
        )
        totals["logical_tokens_per_layer_query"] = (
            totals["logical_tokens"] / totals["queries"] if totals["queries"] else 0.0
        )
        return {"totals": totals, "layers": per_layer}


__all__ = [
    "BasisServeQwen3_8BSparseC1ForCausalLM",
    "MODEL_ARCHITECTURE",
    "fold_value_projection",
    "router_checkpoint_fingerprint",
]
