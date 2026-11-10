"""TP4/TP8 V96/B16R16 Page32 decode with mapped exact K and persistent slots."""

from pathlib import Path

from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, rotate_half

from basisserve.core.llama_tp4_uniform_als import LlamaTP4UniformALSAttention
from basisserve.core.routing_basis import make_routing_basis
from basisserve.kernels.slot_indexed_attention import slot_indexed_attention
from benchmarks.system.fused_candidates import candidates
from benchmarks.system.two_stage_router import Metadata


PAGE_SIZE = 32
RECENT_TOKENS = 64
SUPPORT_TOKENS = 2048
SLOT_SPLITS = 16


class LlamaTP4JointSparseAttention(LlamaTP4UniformALSAttention):
    def __init__(
        self, original: nn.Module, factor_path: Path, router_path: Path, rank: int,
        *, batch: int, capacity: int, shared_rope: dict[str, torch.Tensor],
        full_router, fine_router, slot_extension, postprocess_extension,
    ) -> None:
        local_kv = 8 // dist.get_world_size()
        source_weight = original.v_proj.weight.detach().to_local().reshape(
            local_kv, 128, 4096
        ).float()
        super().__init__(original, factor_path, rank)
        device = torch.device("cuda", rank)
        factors = load_file(str(factor_path), device="cpu")
        router = load_file(str(router_path), device="cpu")
        routing_basis = make_routing_basis(router["base_left_b16"].to(device))
        encoder = factors["value_coordinate_encoders"].to(device=device, dtype=torch.float32)
        decoder = factors["head_output_decoders"].to(device=device, dtype=torch.float32)
        transformed_encoder = encoder @ routing_basis.transform.float()
        transformed_decoder = routing_basis.inverse.float().repeat_interleave(4, dim=0) @ decoder
        local_encoder = transformed_encoder[rank * local_kv:(rank + 1) * local_kv]
        folded = torch.einsum("hdr,hdm->hrm", local_encoder, source_weight).reshape(
            local_kv * 96, 4096
        )
        self.v_proj.weight = nn.Parameter(folded.bfloat16(), requires_grad=False)
        self.decoder = transformed_decoder.reshape(32 * 96, 4096).bfloat16().contiguous()

        kv_slice = slice(rank * self.local_kv, (rank + 1) * self.local_kv)
        q_slice = slice(rank * self.local_q, (rank + 1) * self.local_q)
        self.register_buffer("base_right", router["base_right_b16"][kv_slice].to(device, torch.bfloat16))
        self.register_buffer("base_bias", router["base_bias_b16"][kv_slice].to(device, torch.bfloat16))
        self.register_buffer(
            "residual_encoder", router["residual_encoder_b16_r16"][kv_slice].to(device, torch.bfloat16)
        )
        self.register_buffer(
            "residual_query", router["residual_query_b16_r16"][q_slice].to(device, torch.bfloat16)
        )
        self.batch = batch
        self.capacity = capacity
        self.length = 0
        self.metadata = None
        self.shared_rope = shared_rope
        self.full_router = full_router
        self.fine_router = fine_router
        self.slot_extension = slot_extension
        self.postprocess_extension = postprocess_extension
        self.host_key = full_router.mapped_host_bf16_empty(batch, self.local_kv, capacity, 128)
        self.host_key_pointer = int(full_router.device_pointer(self.host_key))
        self.register_buffer(
            "value_cache", torch.empty(batch, self.local_kv, capacity, 96, device=device, dtype=torch.bfloat16)
        )
        self.register_buffer(
            "base_cache", torch.empty(batch, self.local_kv, capacity, 16, device=device, dtype=torch.bfloat16)
        )
        self.register_buffer(
            "residual_cache", torch.empty_like(self.base_cache)
        )
        self.register_buffer(
            "slot_key_cache", torch.empty(batch, self.local_kv, SUPPORT_TOKENS, 128, device=device, dtype=torch.bfloat16)
        )
        self.register_buffer(
            "slot_resident", torch.full((batch, self.local_kv, SUPPORT_TOKENS), -1, device=device, dtype=torch.long)
        )
        self.register_buffer(
            "slot_lookup", torch.full((batch, self.local_kv, capacity), -1, device=device, dtype=torch.int32)
        )
        self.register_buffer(
            "support_ids", torch.empty(batch, self.local_kv, SUPPORT_TOKENS, device=device, dtype=torch.long)
        )
        self.register_buffer("selected_slots", torch.empty_like(self.support_ids))
        self.register_buffer("slot_missing", torch.empty_like(self.support_ids, dtype=torch.int32))
        self.register_buffer(
            "slot_counts", torch.empty(batch, self.local_kv, 2, device=device, dtype=torch.int32)
        )
        self.register_buffer(
            "selected_pages", torch.empty(batch, self.local_kv, 62, device=device, dtype=torch.long)
        )
        self.register_buffer(
            "candidate_ids", torch.empty(batch, self.local_kv, 512, device=device, dtype=torch.long)
        )
        self.register_buffer(
            "fine_scores", torch.empty(batch, self.local_kv, 4, 512, device=device, dtype=torch.float32)
        )
        self.register_buffer(
            "query_code", torch.empty(batch, self.local_kv, 4, 16, device=device, dtype=torch.bfloat16)
        )
        self.slot_workspace = (
            torch.empty(batch, self.local_q, SLOT_SPLITS, 96, device=device, dtype=torch.float32),
            torch.empty(batch * self.local_q, SLOT_SPLITS, device=device, dtype=torch.float32),
            torch.empty(batch, self.local_q, 1, 96, device=device, dtype=torch.bfloat16),
        )

    def _write_codes(self, key: torch.Tensor, value: torch.Tensor, cos: torch.Tensor,
                     sin: torch.Tensor, start: int) -> None:
        stop = start + key.shape[2]
        base = value[..., :16].contiguous()
        predicted = (base @ self.base_right + self.base_bias[None, :, None]).bfloat16()
        predicted_rotary = (
            predicted * cos[:, None] + rotate_half(predicted) * sin[:, None]
        ).bfloat16()
        residual = (key - predicted_rotary).bfloat16() @ self.residual_encoder
        self.value_cache[:, :, start:stop].copy_(value)
        self.base_cache[:, :, start:stop].copy_(base)
        self.residual_cache[:, :, start:stop].copy_(residual)

    def _sparse_decode(self, query: torch.Tensor, end: int) -> torch.Tensor:
        historical = end - RECENT_TOKENS
        coarse_scores = self.metadata.scores(query)
        candidate_ids = candidates(coarse_scores, historical, output=self.candidate_ids)
        candidate_count = candidate_ids.shape[-1]
        route_args = (
            query,
            self.base_cache[:, :, :historical],
            self.residual_cache[:, :, :historical],
            self.base_right, self.base_bias, self.residual_query,
            self.shared_rope["cos"][:historical],
            self.shared_rope["sin"][:historical],
            self.query_code,
        )
        self.fine_router.conditional_router_page_lse(
            *route_args,
            self.fine_scores[:, :, :, :candidate_count],
            self.scaling, False, candidate_ids,
        )
        self.postprocess_extension.select_and_pack(
            self.fine_scores[:, :, :, :candidate_count], candidate_ids,
            self.selected_pages, self.support_ids, historical, end,
        )
        self.slot_extension.refresh(
            self.host_key_pointer, self.slot_key_cache, self.support_ids,
            self.slot_resident, self.slot_lookup, self.selected_slots,
            self.slot_missing, self.slot_counts, True,
        )
        return slot_indexed_attention(
            query, self.slot_key_cache, self.value_cache, self.support_ids,
            self.selected_slots, self.slot_workspace, scale=self.scaling,
        )

    def forward(
        self, hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask=None, past_key_values=None, cache_position=None, **kwargs,
    ) -> tuple[torch.Tensor, None]:
        assert past_key_values is None
        batch, tokens, _ = hidden_states.shape
        assert batch == self.batch
        start = self.length
        end = start + tokens
        assert end <= self.capacity
        query = self.q_proj(hidden_states).view(batch, tokens, self.local_q, 128).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, tokens, self.local_kv, 128).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, tokens, self.local_kv, 96).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        cos_half = cos[..., :64].contiguous()
        sin_half = sin[..., :64].contiguous()
        if self.layer_idx == 0:
            self.shared_rope["cos"][start:end].copy_(cos_half[0])
            self.shared_rope["sin"][start:end].copy_(sin_half[0])

        if start == 0:
            assert tokens >= RECENT_TOKENS
            for offset in range(0, tokens, 2048):
                stop = min(tokens, offset + 2048)
                self._write_codes(
                    key[:, :, offset:stop], value[:, :, offset:stop],
                    cos[:, offset:stop], sin[:, offset:stop], offset,
                )
            self.full_router.append(self.host_key, key.contiguous(), 0)
            self.metadata = Metadata(key, self.capacity)
            coordinates = F.scaled_dot_product_attention(
                query, key, value, is_causal=True, scale=self.scaling, enable_gqa=True,
            )
        else:
            assert tokens == 1 and start >= SUPPORT_TOKENS
            self._write_codes(key, value, cos, sin, start)
            self.full_router.append(self.host_key, key.contiguous(), start)
            self.metadata.advance(key)
            coordinates = self._sparse_decode(query, end)
        self.length = end
        return self.project_output(coordinates), None


def install_tp4_joint_sparse(
    model: nn.Module, factor_dir: Path, router_dir: Path, rank: int,
    *, batch: int, capacity: int, full_router, fine_router,
    slot_extension, postprocess_extension,
) -> list[LlamaTP4JointSparseAttention]:
    assert model.config.model_type == "llama" and model.config.num_hidden_layers == 32
    device = torch.device("cuda", rank)
    shared_rope = {
        "cos": torch.empty(capacity, 64, device=device, dtype=torch.bfloat16),
        "sin": torch.empty(capacity, 64, device=device, dtype=torch.bfloat16),
    }
    installed = []
    for index, layer in enumerate(model.model.layers):
        replacement = LlamaTP4JointSparseAttention(
            layer.self_attn, factor_dir / f"layer_{index:03d}.safetensors",
            router_dir / f"layer_{index:03d}.safetensors", rank,
            batch=batch, capacity=capacity, shared_rope=shared_rope,
            full_router=full_router, fine_router=fine_router,
            slot_extension=slot_extension, postprocess_extension=postprocess_extension,
        )
        layer.self_attn = replacement
        installed.append(replacement)
    return installed
