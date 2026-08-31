"""Qwen3 checkpoint and reference runtime for the S80-R32 latent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F

from basisserve.core.c1_k_reverse_shadow import (
    ReverseShadowConfig,
    c1_k_reverse_shadow_block_attention,
)
from basisserve.core.gqa_joint_routing_payload_s80 import (
    FoldedS80Factors,
    S80Layout,
)


S80_BANK_FORMAT = "basisserve.qwen3.gqa_joint_routing_payload_s80_r32.v3"
S80_KEY_CONVENTION = "post_rope"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(tensors), str(temporary))
    os.replace(temporary, path)


def build_s80_joint_latent(
    v_derived_latent: torch.Tensor,
    post_rope_key_states: torch.Tensor,
    key_joint_encoder: torch.Tensor,
) -> torch.Tensor:
    """Form ``V A_V + K_post A_K`` in head-major layout."""

    return v_derived_latent + torch.einsum(
        "bgtd,gdr->bgtr",
        post_rope_key_states,
        key_joint_encoder.to(
            device=post_rope_key_states.device,
            dtype=post_rope_key_states.dtype,
        ),
    )


def compute_s80_routing_proxy_scores(
    query_states: torch.Tensor,
    cached_joint_latent: torch.Tensor,
    routing_query_factor: torch.Tensor,
    *,
    scaling: float,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute routing-only proxy logits without replacing exact attention."""

    query_heads = int(query_states.shape[1])
    kv_heads = int(cached_joint_latent.shape[1])
    routing_rank = int(routing_query_factor.shape[-1])
    projected_query = torch.einsum(
        "bhnd,hdr->bhnr",
        query_states,
        routing_query_factor.to(device=query_states.device, dtype=query_states.dtype),
    )
    selected_latent = cached_joint_latent[..., :routing_rank]
    expanded_latent = selected_latent.repeat_interleave(
        query_heads // kv_heads,
        dim=1,
    )
    scores = torch.matmul(projected_query, expanded_latent.transpose(-1, -2))
    scores = scores * float(scaling)
    if attention_mask is not None:
        scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)
    return scores


@dataclass(frozen=True)
class S80LayerExport:
    layer_index: int
    factors: FoldedS80Factors
    routing_weight: float
    payload_normalizer: float
    routing_normalizer: float
    fit_diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class S80ReplacementRecord:
    layer_index: int
    module_name: str
    original_module: nn.Module
    replacement_module: "S80Qwen3Attention"


def write_s80_factor_bank(
    output_dir: str | Path,
    *,
    layout: S80Layout,
    layers: Sequence[S80LayerExport],
    model_identifier: str,
    model_config_sha256: str,
    initial_factor_sources: Mapping[str, Any],
    solver_configuration: Mapping[str, Any],
    command: str,
    environment: Mapping[str, Any],
) -> Path:
    """Write a self-describing serving bank with one safetensor per layer."""

    root = Path(output_dir).expanduser().resolve()
    indices = tuple(int(layer.layer_index) for layer in layers)
    root.mkdir(parents=True)
    artifacts: list[dict[str, Any]] = []
    for layer in sorted(layers, key=lambda item: item.layer_index):
        tensors: dict[str, torch.Tensor] = {
            "v_joint_proj_weight": layer.factors.v_joint_proj_weight.contiguous(),
            "k_joint_encoder": layer.factors.k_joint_encoder.contiguous(),
            "routing_query_factor": layer.factors.routing_query_factor.contiguous(),
            "o_decoder_weight": layer.factors.o_decoder_weight.contiguous(),
            "head_to_kv_group": layer.factors.head_to_kv_group.long().contiguous(),
            "layer_index": torch.tensor(layer.layer_index, dtype=torch.int64),
            "joint_rank": torch.tensor(layout.joint_rank, dtype=torch.int64),
            "routing_rank": torch.tensor(layout.routing_rank, dtype=torch.int64),
            "routing_weight": torch.tensor(layer.routing_weight, dtype=torch.float64),
            "payload_normalizer": torch.tensor(
                layer.payload_normalizer, dtype=torch.float64
            ),
            "routing_normalizer": torch.tensor(
                layer.routing_normalizer, dtype=torch.float64
            ),
        }
        if layer.factors.v_joint_proj_bias is not None:
            tensors["v_joint_proj_bias"] = (
                layer.factors.v_joint_proj_bias.contiguous()
            )
        if layer.factors.o_decoder_bias is not None:
            tensors["o_decoder_bias"] = layer.factors.o_decoder_bias.contiguous()
        filename = f"layer_{layer.layer_index:03d}.safetensors"
        path = root / filename
        _atomic_safetensors(path, tensors)
        artifacts.append(
            {
                "layer_index": layer.layer_index,
                "file": filename,
                "sha256": _sha256(path),
                "routing_weight": float(layer.routing_weight),
                "payload_normalizer": float(layer.payload_normalizer),
                "routing_normalizer": float(layer.routing_normalizer),
                "factor_shapes": {
                    name: list(tensor.shape) for name, tensor in tensors.items()
                },
                "fit_diagnostics": dict(layer.fit_diagnostics),
            }
        )
    manifest = {
        "format": S80_BANK_FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "model": {
            "identifier": model_identifier,
            "config_sha256": model_config_sha256,
        },
        "geometry": asdict(layout),
        "routing_layout": {
            "stored_coordinates": layout.joint_rank,
            "routed_coordinates": layout.routing_rank,
            "routed_coordinate_start": 0,
            "selector_materialized": False,
        },
        "key_convention": S80_KEY_CONVENTION,
        "layer_coverage": sorted(indices),
        "initial_factor_sources": dict(initial_factor_sources),
        "solver_configuration": dict(solver_configuration),
        "environment": dict(environment),
        "artifacts": artifacts,
    }
    _atomic_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def load_s80_factor_bank(
    factor_dir: str | Path,
) -> tuple[dict[str, Any], dict[int, tuple[FoldedS80Factors, dict[str, float]]]]:
    root = Path(factor_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded: dict[int, tuple[FoldedS80Factors, dict[str, float]]] = {}
    for artifact in manifest["artifacts"]:
        index = int(artifact["layer_index"])
        path = root / str(artifact["file"])
        tensors = load_file(str(path), device="cpu")
        factors = FoldedS80Factors(
            v_joint_proj_weight=tensors["v_joint_proj_weight"],
            v_joint_proj_bias=tensors.get("v_joint_proj_bias"),
            k_joint_encoder=tensors["k_joint_encoder"],
            routing_query_factor=tensors["routing_query_factor"],
            o_decoder_weight=tensors["o_decoder_weight"],
            o_decoder_bias=tensors.get("o_decoder_bias"),
            head_to_kv_group=tensors["head_to_kv_group"],
        )
        scalars = {
            "routing_weight": float(tensors["routing_weight"]),
            "payload_normalizer": float(tensors["payload_normalizer"]),
            "routing_normalizer": float(tensors["routing_normalizer"]),
        }
        loaded[index] = (factors, scalars)
    return manifest, loaded


def merge_s80_factor_banks(
    output_dir: str | Path,
    *,
    input_dirs: Sequence[str | Path],
    command: str,
) -> Path:
    """Combine disjoint S80 layer banks."""

    output_root = Path(output_dir).expanduser().resolve()
    inputs = tuple(Path(item).expanduser().resolve() for item in input_dirs)
    manifests: list[dict[str, Any]] = []
    artifacts: list[tuple[Path, dict[str, Any]]] = []
    observed_layers: set[int] = set()
    for root in inputs:
        manifest, loaded = load_s80_factor_bank(root)
        coverage = set(loaded)
        observed_layers.update(coverage)
        by_layer = {
            int(artifact["layer_index"]): artifact for artifact in manifest["artifacts"]
        }
        artifacts.extend((root, by_layer[index]) for index in sorted(coverage))
        manifests.append(manifest)
    reference = manifests[0]
    reference_solver = dict(reference["solver_configuration"])
    reference_solver.pop("elapsed_seconds_before_write", None)
    output_root.mkdir(parents=True)
    merged_artifacts: list[dict[str, Any]] = []
    for source_root, artifact in sorted(
        artifacts,
        key=lambda item: int(item[1]["layer_index"]),
    ):
        source = source_root / str(artifact["file"])
        destination = output_root / str(artifact["file"])
        shutil.copy2(source, destination)
        merged_artifacts.append(dict(artifact))
    source_banks = []
    for root, manifest in zip(inputs, manifests):
        manifest_path = root / "manifest.json"
        source_banks.append(
            {
                "path": str(root),
                "manifest_sha256": _sha256(manifest_path),
                "layer_coverage": list(manifest["layer_coverage"]),
                "command": manifest["command"],
                "initial_factor_sources": manifest["initial_factor_sources"],
                "elapsed_seconds_before_write": manifest[
                    "solver_configuration"
                ].get("elapsed_seconds_before_write"),
            }
        )
    merged_manifest = {
        "format": S80_BANK_FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "model": reference["model"],
        "geometry": reference["geometry"],
        "routing_layout": reference["routing_layout"],
        "key_convention": reference["key_convention"],
        "layer_coverage": sorted(observed_layers),
        "initial_factor_sources": {
            "sharded_builds": [
                item["initial_factor_sources"] for item in source_banks
            ]
        },
        "solver_configuration": reference_solver,
        "environment": reference["environment"],
        "source_banks": source_banks,
        "artifacts": merged_artifacts,
    }
    _atomic_json(output_root / "manifest.json", merged_manifest)
    return output_root / "manifest.json"


class S80Qwen3Attention(nn.Module):
    """Reference Qwen3 exact-QK attention over the shared joint S80 cache."""

    def __init__(
        self,
        base_attention: nn.Module,
        *,
        factors: FoldedS80Factors,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        config = base_attention.config
        self.config = config
        self.layer_idx = int(base_attention.layer_idx)
        self.head_dim = int(base_attention.head_dim)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = int(base_attention.num_key_value_groups)
        self.scaling = float(base_attention.scaling)
        self.attention_dropout = float(base_attention.attention_dropout)
        self.is_causal = bool(base_attention.is_causal)
        self.sliding_window = base_attention.sliding_window
        self.attention_backend = str(attention_backend)
        hidden_size = int(config.hidden_size)
        compressed_width = int(factors.v_joint_proj_weight.shape[0])
        self.joint_rank = compressed_width // self.num_key_value_heads
        device = base_attention.q_proj.weight.device
        dtype = base_attention.q_proj.weight.dtype
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.v_joint_proj = nn.Linear(
            hidden_size,
            compressed_width,
            bias=factors.v_joint_proj_bias is not None,
            device=device,
            dtype=dtype,
        )
        self.o_decoder = nn.Linear(
            self.num_attention_heads * self.joint_rank,
            hidden_size,
            bias=factors.o_decoder_bias is not None,
            device=device,
            dtype=dtype,
        )
        self.v_joint_proj.weight.data.copy_(
            factors.v_joint_proj_weight.to(device=device, dtype=dtype)
        )
        self.o_decoder.weight.data.copy_(
            factors.o_decoder_weight.to(device=device, dtype=dtype)
        )
        if self.v_joint_proj.bias is not None:
            self.v_joint_proj.bias.data.copy_(
                factors.v_joint_proj_bias.to(device=device, dtype=dtype)
            )
        if self.o_decoder.bias is not None:
            self.o_decoder.bias.data.copy_(
                factors.o_decoder_bias.to(device=device, dtype=dtype)
            )
        self.register_buffer(
            "k_joint_encoder",
            factors.k_joint_encoder.to(device=device, dtype=dtype),
        )
        self.register_buffer(
            "routing_query_factor",
            factors.routing_query_factor.to(device=device, dtype=dtype),
        )
        self.register_buffer(
            "head_to_kv_group",
            factors.head_to_kv_group.to(device=device, dtype=torch.long),
        )
        self.sparse_routing_config: ReverseShadowConfig | None = None
        self.reset_sparse_routing_statistics()
        self.train(base_attention.training)

    def set_sparse_routing_config(
        self,
        config: ReverseShadowConfig | None,
    ) -> None:
        """Select dense exact-K attention or Route32-selected exact-K pages."""

        if config is not None:
            config.validate(self.head_dim)
        self.sparse_routing_config = config
        self.reset_sparse_routing_statistics()

    def reset_sparse_routing_statistics(self) -> None:
        self._sparse_routing_totals = {
            "queries": 0.0,
            "physical_valid_tokens": 0.0,
            "query_valid_tokens": 0.0,
            "selected_tokens": 0.0,
            "query_selected_tokens": 0.0,
            "selected_pages": 0.0,
            "logical_selected_pages": 0.0,
            "cpu_exact_key_bytes_fetched": 0.0,
            "selection_qk_flops": 0.0,
            "sparse_exact_qk_flops": 0.0,
            "sparse_c1_value_flops": 0.0,
            "adaptive_eligible_query_heads": 0.0,
            "adaptive_refined_query_heads": 0.0,
            "adaptive_tail_mass_ratio_sum": 0.0,
            "maximum_resident_selector_metadata_bytes": 0.0,
        }

    def sparse_routing_statistics(self) -> dict[str, float]:
        totals = dict(self._sparse_routing_totals)
        valid = totals["physical_valid_tokens"]
        query_valid = totals["query_valid_tokens"]
        totals["selected_token_fraction"] = (
            totals["selected_tokens"] / valid if valid else 0.0
        )
        totals["query_selected_token_fraction"] = (
            totals["query_selected_tokens"] / query_valid if query_valid else 0.0
        )
        return totals

    def _record_sparse_routing_results(self, results: Sequence[Any]) -> None:
        for result in results:
            statistics = result.statistics
            self._sparse_routing_totals["queries"] += float(result.output.shape[0])
            for name in (
                "physical_valid_tokens",
                "query_valid_tokens",
                "selected_tokens",
                "query_selected_tokens",
                "selected_pages",
                "logical_selected_pages",
                "selection_qk_flops",
                "sparse_exact_qk_flops",
                "sparse_c1_value_flops",
                "adaptive_eligible_query_heads",
                "adaptive_refined_query_heads",
                "adaptive_tail_mass_ratio_sum",
            ):
                self._sparse_routing_totals[name] += float(statistics[name])
            self._sparse_routing_totals["cpu_exact_key_bytes_fetched"] += float(
                statistics["oracle_page_store_key_bytes_read"]
            )
            name = "maximum_resident_selector_metadata_bytes"
            self._sparse_routing_totals[name] = max(
                self._sparse_routing_totals[name],
                float(statistics["resident_selector_metadata_bytes"]),
            )

    def compute_routing_proxy_scores(
        self,
        query_states: torch.Tensor,
        cached_joint_latent: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compute_s80_routing_proxy_scores(
            query_states,
            cached_joint_latent,
            self.routing_query_factor,
            scaling=self.scaling,
            attention_mask=attention_mask,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, self.num_attention_heads, self.head_dim)
        key_shape = (*input_shape, self.num_key_value_heads, self.head_dim)
        value_shape = (*input_shape, self.num_key_value_heads, self.joint_rank)
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(query_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(key_shape)).transpose(
            1, 2
        )
        v_derived = self.v_joint_proj(hidden_states).view(value_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )
        joint_states = build_s80_joint_latent(
            v_derived,
            key_states,
            self.k_joint_encoder,
        )
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, joint_states = past_key_values.update(
                key_states,
                joint_states,
                self.layer_idx,
                cache_kwargs,
            )
        attn_weights = None
        if self.sparse_routing_config is not None:
            head_major_output, routing_results = c1_k_reverse_shadow_block_attention(
                query_states,
                key_states,
                joint_states,
                self.sparse_routing_config,
                attention_mask,
                routing_query_projector=self.routing_query_factor,
                routing_sidecar=joint_states[..., : self.routing_query_factor.shape[-1]],
                layer_idx=self.layer_idx,
            )
            self._record_sparse_routing_results(routing_results)
            attn_output = head_major_output.transpose(1, 2).contiguous()
        elif self.attention_backend == "sdpa":
            attn_output = (
                F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    joint_states,
                    attn_mask=attention_mask,
                    dropout_p=0.0 if not self.training else self.attention_dropout,
                    is_causal=bool(
                        self.is_causal
                        and attention_mask is None
                        and query_states.shape[-2] > 1
                    ),
                    scale=self.scaling,
                    enable_gqa=self.num_key_value_groups > 1,
                )
                .transpose(1, 2)
                .contiguous()
            )
        else:
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                joint_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        output = self.o_decoder(attn_output.reshape(*input_shape, -1).contiguous())
        return output, attn_weights


def _replace_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent_name, child_name = module_name.rsplit(".", 1)
    setattr(root.get_submodule(parent_name), child_name, replacement)


def install_qwen3_s80_factor_bank(
    model: nn.Module,
    factor_dir: str | Path,
    *,
    attention_backend: str = "sdpa",
) -> list[S80ReplacementRecord]:
    _, bank = load_s80_factor_bank(factor_dir)
    records: list[S80ReplacementRecord] = []
    for layer_index in sorted(bank):
        module_name = f"model.layers.{layer_index}.self_attn"
        original = model.get_submodule(module_name)
        replacement = S80Qwen3Attention(
            original,
            factors=bank[layer_index][0],
            attention_backend=attention_backend,
        )
        _replace_module(model, module_name, replacement)
        records.append(
            S80ReplacementRecord(
                layer_index=layer_index,
                module_name=module_name,
                original_module=original,
                replacement_module=replacement,
            )
        )
    return records


def restore_qwen3_s80_layers(
    model: nn.Module,
    records: Sequence[S80ReplacementRecord],
) -> None:
    for record in records:
        _replace_module(model, record.module_name, record.original_module)


__all__ = [
    "S80_BANK_FORMAT",
    "S80_KEY_CONVENTION",
    "S80LayerExport",
    "S80Qwen3Attention",
    "S80ReplacementRecord",
    "build_s80_joint_latent",
    "compute_s80_routing_proxy_scores",
    "install_qwen3_s80_factor_bank",
    "load_s80_factor_bank",
    "merge_s80_factor_banks",
    "restore_qwen3_s80_layers",
    "write_s80_factor_bank",
]
