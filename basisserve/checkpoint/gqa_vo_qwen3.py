"""Install folded GQA-tied V/O factors into Hugging Face Qwen3 attention."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from basisserve.kernels.sageattention import sage_attention_with_compressed_value


@dataclass(frozen=True)
class GQAVOReplacementRecord:
    layer_index: int
    module_name: str
    head_dim: int
    value_head_dim: int
    dense_v_width: int
    compressed_v_width: int
    dense_o_width: int
    compressed_o_width: int


class GQATiedVOQwen3Attention(nn.Module):
    """Qwen3 eager attention with independent Q/K and compressed V dimensions."""

    def __init__(
        self,
        base_attention: nn.Module,
        *,
        v_proj_compressed_weight: torch.Tensor,
        o_decoder_weight: torch.Tensor,
        v_proj_compressed_bias: torch.Tensor | None = None,
        o_decoder_bias: torch.Tensor | None = None,
        attention_backend: str = "native",
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
        if self.attention_backend not in {"native", "sage"}:
            raise ValueError(
                "compressed attention backend must be 'native' or 'sage', got "
                f"{self.attention_backend!r}"
            )

        if v_proj_compressed_weight.ndim != 2:
            raise ValueError("compressed v_proj weight must be a matrix")
        compressed_v_width, hidden_size = v_proj_compressed_weight.shape
        if compressed_v_width % self.num_key_value_heads != 0:
            raise ValueError(
                "compressed V width must be divisible by num_key_value_heads: "
                f"{compressed_v_width} vs {self.num_key_value_heads}"
            )
        self.value_head_dim = compressed_v_width // self.num_key_value_heads
        expected_o_shape = (
            hidden_size,
            self.num_attention_heads * self.value_head_dim,
        )
        if tuple(o_decoder_weight.shape) != expected_o_shape:
            raise ValueError(
                f"o_decoder weight must have shape {expected_o_shape}, "
                f"got {tuple(o_decoder_weight.shape)}"
            )
        if int(config.hidden_size) != hidden_size:
            raise ValueError(
                f"compressed V input width must equal hidden_size={config.hidden_size}, got {hidden_size}"
            )

        device = base_attention.q_proj.weight.device
        dtype = base_attention.q_proj.weight.dtype
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.v_proj = nn.Linear(
            hidden_size,
            compressed_v_width,
            bias=v_proj_compressed_bias is not None,
            device=device,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            expected_o_shape[1],
            hidden_size,
            bias=o_decoder_bias is not None,
            device=device,
            dtype=dtype,
        )
        self.v_proj.weight.data.copy_(v_proj_compressed_weight.to(device=device, dtype=dtype))
        self.o_proj.weight.data.copy_(o_decoder_weight.to(device=device, dtype=dtype))
        if self.v_proj.bias is not None:
            self.v_proj.bias.data.copy_(v_proj_compressed_bias.to(device=device, dtype=dtype))
        if self.o_proj.bias is not None:
            self.o_proj.bias.data.copy_(o_decoder_bias.to(device=device, dtype=dtype))
        self.train(base_attention.training)

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
        value_shape = (*input_shape, self.num_key_value_heads, self.value_head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(key_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(value_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        if self.attention_backend == "sage":
            if self.training and self.attention_dropout:
                raise NotImplementedError(
                    "SageAttention compressed backend is inference-only"
                )
            if kwargs.get("output_attentions"):
                raise NotImplementedError(
                    "SageAttention compressed backend does not return attention weights"
                )
            if self.sliding_window is not None and attention_mask is None:
                raise NotImplementedError(
                    "SageAttention requires an explicit mask for sliding-window attention"
                )
            attn_output = sage_attention_with_compressed_value(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                is_causal=bool(
                    self.is_causal
                    and attention_mask is None
                    and query_states.shape[-2] > 1
                ),
                scaling=self.scaling,
            )
            attn_weights = None
        else:
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class TrainableGQATiedVOQwen3Attention(nn.Module):
    """Training-time Qwen3 attention with explicit tied bases and decoder."""

    def __init__(
        self,
        base_attention: nn.Module,
        *,
        basis_pt: torch.Tensor,
        decoder_weight: torch.Tensor,
        decoder_bias: torch.Tensor | None = None,
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

        expected_basis_prefix = (self.num_key_value_heads,)
        if basis_pt.ndim != 3 or tuple(basis_pt.shape[:1]) != expected_basis_prefix:
            raise ValueError(
                "basis_pt must have shape [num_key_value_heads, rank, head_dim], "
                f"got {tuple(basis_pt.shape)}"
            )
        if basis_pt.shape[2] != self.head_dim:
            raise ValueError(
                f"basis head dimension must be {self.head_dim}, got {basis_pt.shape[2]}"
            )
        self.value_head_dim = int(basis_pt.shape[1])
        hidden_size = int(config.hidden_size)
        expected_decoder_shape = (
            hidden_size,
            self.num_attention_heads * self.value_head_dim,
        )
        if tuple(decoder_weight.shape) != expected_decoder_shape:
            raise ValueError(
                f"decoder_weight must have shape {expected_decoder_shape}, "
                f"got {tuple(decoder_weight.shape)}"
            )

        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.register_buffer(
            "dense_v_weight",
            base_attention.v_proj.weight.detach(),
            persistent=False,
        )
        self.register_buffer(
            "dense_v_bias",
            None if base_attention.v_proj.bias is None else base_attention.v_proj.bias.detach(),
            persistent=False,
        )
        self.basis_pt = nn.Parameter(basis_pt.detach().to(torch.float32).clone())
        self.decoder_weight = nn.Parameter(decoder_weight.detach().to(torch.float32).clone())
        self.register_buffer(
            "decoder_bias",
            None if decoder_bias is None else decoder_bias.detach().clone(),
            persistent=False,
        )
        self.train(base_attention.training)

    def folded_v_weight(self) -> torch.Tensor:
        dense = self.dense_v_weight.reshape(
            self.num_key_value_heads,
            self.head_dim,
            -1,
        ).to(self.basis_pt.dtype)
        return torch.bmm(self.basis_pt, dense).reshape(
            self.num_key_value_heads * self.value_head_dim,
            -1,
        )

    def folded_v_bias(self) -> torch.Tensor | None:
        if self.dense_v_bias is None:
            return None
        dense = self.dense_v_bias.reshape(self.num_key_value_heads, self.head_dim, 1).to(
            self.basis_pt.dtype
        )
        return torch.bmm(self.basis_pt, dense).reshape(-1)

    @torch.no_grad()
    def retract_basis_(self) -> None:
        """QR-retract every basis and compensate decoder blocks exactly."""

        rank = self.value_head_dim
        heads_per_group = self.num_key_value_groups
        for group_index in range(self.num_key_value_heads):
            basis_row = self.basis_pt[group_index].transpose(0, 1)
            q_basis, transform = torch.linalg.qr(basis_row, mode="reduced")
            self.basis_pt[group_index].copy_(q_basis.transpose(0, 1))
            first_head = group_index * heads_per_group
            for head_offset in range(heads_per_group):
                head_index = first_head + head_offset
                start = head_index * rank
                block = self.decoder_weight[:, start : start + rank]
                block.copy_(block @ transform.transpose(0, 1))

    @torch.no_grad()
    def export_projection_payload(
        self,
        *,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        if dtype is None:
            dtype = self.dense_v_weight.dtype
        folded_bias = self.folded_v_bias()
        return {
            "format": "basisserve.gqa_vo_svdllm.layer.v1",
            "v_proj_compressed_weight": self.folded_v_weight().detach().to(dtype).cpu(),
            "v_proj_compressed_bias": (
                None if folded_bias is None else folded_bias.detach().to(dtype).cpu()
            ),
            "o_decoder_weight": self.decoder_weight.detach().to(dtype).cpu(),
            "o_decoder_bias": (
                None
                if self.decoder_bias is None
                else self.decoder_bias.detach().to(dtype).cpu()
            ),
        }

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
        value_shape = (*input_shape, self.num_key_value_heads, self.value_head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(key_shape)).transpose(1, 2)
        folded_weight = self.folded_v_weight().to(hidden_states.dtype)
        folded_bias = self.folded_v_bias()
        if folded_bias is not None:
            folded_bias = folded_bias.to(hidden_states.dtype)
        value_states = F.linear(hidden_states, folded_weight, folded_bias).view(value_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = F.linear(
            attn_output,
            self.decoder_weight.to(attn_output.dtype),
            None if self.decoder_bias is None else self.decoder_bias.to(attn_output.dtype),
        )
        return output, attn_weights


def _replace_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    setattr(parent, child_name, replacement)


def _load_projection_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "basisserve.gqa_vo_svdllm.layer.v1":
        raise ValueError(f"unsupported GQA V/O layer payload in {path}")
    return payload


@torch.no_grad()
def install_qwen3_gqa_vo_export(
    model: nn.Module,
    export_dir: str | Path,
) -> list[GQAVOReplacementRecord]:
    """Replace selected Qwen3 attention layers from an offline factor export."""

    export_dir = Path(export_dir).expanduser().resolve()
    config_path = export_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "basisserve.gqa_vo_svdllm.export.v1":
        raise ValueError(f"unsupported GQA V/O export format in {config_path}")
    if metadata.get("model_type") != "qwen3":
        raise ValueError(f"expected qwen3 export, got {metadata.get('model_type')!r}")
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError(f"expected Qwen3 model, got {getattr(model.config, 'model_type', None)!r}")

    layout = metadata["layout"]
    expected_config = {
        "hidden_size": int(model.config.hidden_size),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "head_dim": int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)),
    }
    for key, expected in expected_config.items():
        actual = int(layout[key])
        if actual != expected:
            raise ValueError(f"export/model {key} mismatch: {actual} vs {expected}")

    records = []
    for layer_spec in metadata["layers"]:
        layer_index = int(layer_spec["layer_index"])
        module_name = str(layer_spec["module_name"])
        base_attention = model.get_submodule(module_name)
        if isinstance(base_attention, GQATiedVOQwen3Attention):
            raise ValueError(f"attention module is already compressed: {module_name}")
        if bool(layer_spec.get("dense_passthrough", False)):
            rank = int(layer_spec.get("rank_per_kv_head", expected_config["head_dim"]))
            if rank != expected_config["head_dim"]:
                raise ValueError(
                    f"dense passthrough layer {layer_index} has non-dense rank {rank}"
                )
            continue
        payload = _load_projection_payload(
            export_dir / f"layer_{layer_index:03d}" / "compressed_projections.pt"
        )
        replacement = GQATiedVOQwen3Attention(
            base_attention,
            v_proj_compressed_weight=payload["v_proj_compressed_weight"],
            o_decoder_weight=payload["o_decoder_weight"],
            v_proj_compressed_bias=payload.get("v_proj_compressed_bias"),
            o_decoder_bias=payload.get("o_decoder_bias"),
        )
        _replace_module(model, module_name, replacement)
        records.append(
            GQAVOReplacementRecord(
                layer_index=layer_index,
                module_name=module_name,
                head_dim=replacement.head_dim,
                value_head_dim=replacement.value_head_dim,
                dense_v_width=replacement.num_key_value_heads * replacement.head_dim,
                compressed_v_width=replacement.v_proj.out_features,
                dense_o_width=replacement.num_attention_heads * replacement.head_dim,
                compressed_o_width=replacement.o_proj.in_features,
            )
        )
    return records
