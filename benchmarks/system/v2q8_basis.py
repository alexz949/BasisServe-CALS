"""BasisKV decode runtime for the V2Q8 rebuttal, TP1 or tensor-parallel: compressed or dense Values, any page
size dividing 32, optional sink, exact Keys GPU-resident or in mapped pinned host memory.

Trimmed from the archived Table 3 runtime (frozen TP1 harness bench_tp1_sparse_full_v6.py, optimized full-scan
path) with the value width, page geometry (routed tokens incl. sink, sink tokens) and Key placement explicit.

Tensor parallel (rank r of W): KV heads [r*8/W, (r+1)*8/W) and their query heads are local. With compressed
Values the output is source-local: each rank AllGathers its V-coordinate attention outputs and applies the
replicated GQA-tied decoder (the TP8 BasisKV design; no o_proj AllReduce). With dense Values the original
row-parallel o_proj (HF tensor parallel, AllReduce) is kept.

Batch rows are prefilled one request at a time into their cache row (set `prefill_row` before the forward);
decode advances every row together. With a compressed checkpoint v_proj/o_proj follow the quality evaluator
(evaluation/eval_k_routing_ruler.install): v_proj writes E^T W_v per KV head and the decoder is GQA-tied.
"""

import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

RECENT_TOKENS = 64
SLOT_SPLITS = 16
KV_HEADS, QUERY_HEADS, HEAD_DIM, GQA = 8, 32, 128, 4


def local(weight):
    """The local shard of an HF tensor-parallel parameter (a DTensor) or the tensor itself."""
    return weight.to_local() if hasattr(weight, "to_local") else weight


def compressed_value_weights(original, checkpoint, layer, kv_start, kv_count):
    """Local v_proj rows E^T W_v and the full GQA-tied decoder, as the quality evaluator builds them."""
    tensors = load_file(str(checkpoint / f"layer_{layer:03d}.safetensors"))
    encoder, decoder = tensors["value_coordinate_encoders"], tensors["head_output_decoders"]
    rank = encoder.shape[-1]
    value = local(original.v_proj.weight.detach())
    hidden = value.shape[1]
    assert encoder.shape == (KV_HEADS, HEAD_DIM, rank) and decoder.shape == (QUERY_HEADS, rank, hidden)
    assert value.shape == (kv_count * HEAD_DIM, hidden)
    folded = torch.bmm(encoder[kv_start:kv_start + kv_count].cuda().float().mT,
                       value.float().reshape(kv_count, HEAD_DIM, hidden))
    output = decoder.cuda().float().permute(2, 0, 1).reshape(hidden, QUERY_HEADS * rank)
    return folded.reshape(kv_count * rank, hidden).to(torch.bfloat16), output.to(torch.bfloat16), rank


def slot_extension(harness, build_root, support_tokens):
    """The archived persistent Key-slot kernel with its per-row slot arrays sized for `support_tokens`.

    The archived kernel holds at most 2048 slots per row (Table 3: 1984 routed + 64 recent) in shared memory;
    paper Table 2's Page4 support is 2048 + 64. Only those four 2048 bounds change, so at 2048 the source is the
    archived file byte for byte."""
    text = (harness.REPOSITORY_ROOT / "basisserve/kernels/csrc/persistent_key_slots.cu").read_text()
    assert text.count("2048") == 4
    edited = text.replace("2048", str(support_tokens))
    source = Path(build_root) / f"key_slots_s{support_tokens}" / "persistent_key_slots.cu"
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists() or source.read_text() != edited:
        # Atomic replace: tensor-parallel ranks build concurrently and must never read a partial file.
        staged = source.with_name(f"{source.name}.{os.getpid()}")
        staged.write_text(edited)
        os.replace(staged, source)
    return load(name=f"v2q8_persistent_key_slots_s{support_tokens}", sources=[str(source)],
                extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=["-O3", "-std=c++17"])


class BasisAttention(torch.nn.Module):
    """Exact prefill, then B16R16 full-scan page routing and exact attention over the selected support."""

    def __init__(self, original, factors, *, value_weights, batch, capacity, page_size, routed_tokens,
                 sink_tokens, offload, router, slot_extension, slot_attention, shared_rope, rank=0, world=1):
        super().__init__()
        self.kv_heads, self.kv_start = KV_HEADS // world, rank * (KV_HEADS // world)
        self.query_heads = self.kv_heads * GQA
        self.world, self.batch = world, batch
        # Local weight shards used directly (as the TP8 runtime does), bypassing HF tensor-parallel hooks.
        self.config = original.config
        self.q_weight = local(original.q_proj.weight.detach())
        self.k_weight = local(original.k_proj.weight.detach())
        if value_weights is None:
            # Dense Values: original v_proj; row-parallel o_proj with an AllReduce when TP>1.
            self.value_dim, self.gather_output = HEAD_DIM, False
            self.v_weight = local(original.v_proj.weight.detach())
            self.o_weight = local(original.o_proj.weight.detach())
        else:
            # Compressed Values: TP1 applies the GQA-tied decoder directly; TP>1 AllGathers the local
            # V coordinates and applies the replicated decoder (source-local encoding, no o_proj AllReduce).
            self.v_weight, self.o_weight, self.value_dim = value_weights
            self.gather_output = world > 1
        self.layer_idx = int(original.layer_idx)
        self.scaling = float(original.scaling)
        self.router, self.slot_extension, self.slot_attention = router, slot_extension, slot_attention
        # Optional faster router with bit-identical page scores (table3_router_fast.cu); None keeps the archived one.
        self.fast_router = None
        self.shared_rope, self.offload, self.capacity = shared_rope, bool(offload), int(capacity)
        self.page_size, self.routed_pages, self.sink_tokens = page_size, routed_tokens // page_size, sink_tokens
        self.support_tokens = routed_tokens + RECENT_TOKENS
        self.length, self.prefill_row = 0, 0
        self.profile_components, self.profile_events = False, {}
        self.timers = None  # breakdown mode: synchronized component timers shared with the other arms

        kv, q = slice(self.kv_start, self.kv_start + self.kv_heads), slice(self.kv_start * GQA,
                                                                           (self.kv_start + self.kv_heads) * GQA)
        # BF16-rounded, as the archived runtime rotates its BF16 buffer (bench_table3_k_offload.read_base_from_value).
        self.full_base_left = factors["base_left_b16"].to("cuda", torch.bfloat16).double()
        for name in ("base_left", "base_right", "base_bias"):
            self.register_buffer(name, factors[f"{name}_b16"][kv].to("cuda", torch.bfloat16).contiguous())
        self.register_buffer("residual_encoder",
                             factors["residual_encoder_b16_r16"][kv].to("cuda", torch.bfloat16).contiguous())
        self.register_buffer("residual_query",
                             factors["residual_query_b16_r16"][q].to("cuda", torch.bfloat16).contiguous())
        assert self.base_left.shape == (self.kv_heads, self.value_dim, 16)

        shape = (batch, self.kv_heads, capacity)
        self.value_cache = torch.empty(*shape, self.value_dim, device="cuda", dtype=torch.bfloat16)
        self.base_cache = torch.empty(*shape, 16, device="cuda", dtype=torch.bfloat16)
        self.residual_cache = torch.empty(*shape, 16, device="cuda", dtype=torch.bfloat16)
        if self.offload:
            self.key_cache = None
            self.host_key = router.mapped_host_bf16_empty(batch, self.kv_heads, capacity, HEAD_DIM)
            self.key_pointer = int(router.device_pointer(self.host_key))
        else:
            self.key_cache = torch.empty(*shape, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
            self.host_key = None
            self.key_pointer = int(self.key_cache.data_ptr())

        rows = (batch, self.kv_heads)
        self.query_code = torch.empty(*rows, GQA, 16, device="cuda", dtype=torch.bfloat16)
        self.router_output = torch.empty(*rows, GQA, math.ceil(capacity / page_size), device="cuda")
        self.selected_pages = torch.empty(*rows, self.routed_pages, device="cuda", dtype=torch.long)
        self.support_ids = torch.empty(*rows, self.support_tokens, device="cuda", dtype=torch.long)
        self.attention_output = torch.empty(batch, self.query_heads, 1, self.value_dim, device="cuda",
                                            dtype=torch.bfloat16)
        self.slot_workspace = (
            torch.empty(batch, self.query_heads, SLOT_SPLITS, self.value_dim, device="cuda", dtype=torch.float32),
            torch.empty(batch, self.query_heads, SLOT_SPLITS, device="cuda", dtype=torch.float32),
            self.attention_output)
        if self.offload:
            self.slot_key_cache = torch.empty(*rows, self.support_tokens, HEAD_DIM, device="cuda",
                                              dtype=torch.bfloat16)
            self.slot_resident = torch.full((*rows, self.support_tokens), -1, device="cuda", dtype=torch.long)
            self.slot_lookup = torch.full(shape, -1, device="cuda", dtype=torch.int32)
            self.selected_slots = torch.empty_like(self.support_ids)
            self.slot_missing = torch.empty_like(self.support_ids, dtype=torch.int32)
            self.slot_counts = torch.empty(*rows, 2, device="cuda", dtype=torch.int32)
        else:
            self.slot_key_cache = self.slot_resident = self.slot_lookup = None
            self.selected_slots = self.slot_missing = self.slot_counts = None
        self.single_rope = torch.empty(1, 64, device="cuda", dtype=torch.bfloat16)
        if self.gather_output:
            self.gathered = torch.empty(world, batch, self.query_heads * self.value_dim, device="cuda",
                                        dtype=torch.bfloat16)

    def read_base_from_value(self):
        """Rotate the Value basis so its first 16 coordinates are the Base code, then drop the Base copy.

        Algebraically exact; the rotated v_proj, decoder and base_right are re-rounded to BF16."""
        bases, upper = torch.linalg.qr(self.full_base_left, mode="complete")
        own = slice(self.kv_start, self.kv_start + self.kv_heads)
        # The output projection holds either every query head (the replicated V96 decoder) or only this rank's
        # (row-parallel o_proj, V128 under TP); it is rotated with the bases of the KV heads it reads.
        output_heads = self.o_weight.shape[1] // self.value_dim
        assert output_heads in (QUERY_HEADS, self.query_heads)
        output_bases = bases if output_heads == QUERY_HEADS else bases[own]
        hidden = self.v_weight.shape[1]
        value_weight = self.v_weight.view(self.kv_heads, self.value_dim, hidden)
        value_weight.copy_(bases[own].transpose(1, 2) @ value_weight.double())
        output_weight = self.o_weight.view(-1, output_heads, self.value_dim)
        output_weight.copy_(torch.einsum("ohd,hde->ohe", output_weight.double(),
                                         output_bases.repeat_interleave(GQA, dim=0)))
        self.base_right.copy_(upper[own, :16] @ self.base_right.double())
        self.base_left.copy_(torch.eye(self.value_dim, 16, device="cuda", dtype=torch.float64).expand(
            self.kv_heads, -1, -1))
        self.base_cache = self.value_cache[..., :16]

    def reset(self):
        self.length, self.prefill_row = 0, 0
        if self.offload:
            self.slot_resident.fill_(-1)
            self.slot_lookup.fill_(-1)

    def _start(self, name):
        if self.timers is not None:
            self.timers.start(name)
        elif self.profile_components:
            self.profile_events[name][0].record()

    def _end(self, name):
        if self.timers is not None:
            self.timers.stop(name)
        elif self.profile_components:
            self.profile_events[name][1].record()

    def component_times(self):
        return {name: begin.elapsed_time(end) for name, (begin, end) in self.profile_events.items()}

    def _write_prefill(self, key, value, cos_half, sin_half, row):
        if self.offload:
            # As the archived Table 3 runtime (bench_tp1_sparse_full_v6._store_exact_key): one strided
            # cudaMemcpy2DAsync per head on the current stream. A plain copy_ between these non-contiguous
            # tensors stages through a pageable host buffer (about 20x slower at 128K).
            self.router.append(self.host_key[row:row + 1], key, 0)
        else:
            self.key_cache[row:row + 1, :, :key.shape[2]].copy_(key)
        for start in range(0, key.shape[2], 2048):
            stop = min(key.shape[2], start + 2048)
            chunk_key, chunk_value = key[:, :, start:stop], value[:, :, start:stop]
            base = torch.einsum("bhtd,hdr->bhtr", chunk_value, self.base_left)
            predicted = (torch.einsum("bhtr,hrd->bhtd", base, self.base_right)
                         + self.base_bias[None, :, None, :]).to(torch.bfloat16)
            cos, sin = cos_half[start:stop][None, None], sin_half[start:stop][None, None]
            first = ((predicted[..., :64] * cos).to(torch.bfloat16)
                     - (predicted[..., 64:] * sin).to(torch.bfloat16)).to(torch.bfloat16)
            second = ((predicted[..., 64:] * cos).to(torch.bfloat16)
                      + (predicted[..., :64] * sin).to(torch.bfloat16)).to(torch.bfloat16)
            residual = torch.einsum("bhtd,hdr->bhtr",
                                    (chunk_key - torch.cat((first, second), dim=-1)).to(torch.bfloat16),
                                    self.residual_encoder)
            self.value_cache[row:row + 1, :, start:stop].copy_(chunk_value)
            self.base_cache[row:row + 1, :, start:stop].copy_(base)
            self.residual_cache[row:row + 1, :, start:stop].copy_(residual)

    def _append_decode(self, key, value, cos_half, sin_half, start):
        self.router.conditional_router_append_decode(
            key, value, self.base_left, self.base_right, self.base_bias, self.residual_encoder,
            cos_half, sin_half, self.value_cache, self.base_cache, self.residual_cache,
            self.single_rope, self.single_rope, int(start), False,
            self.key_pointer if self.offload else 0, self.capacity if self.offload else 0,
            self.single_rope, self.single_rope, self.single_rope, 0, 0, False)
        if not self.offload:
            self.key_cache[:, :, start:start + 1].copy_(key)

    def _decode_attention(self, query, end):
        historical = end - RECENT_TOKENS
        pages = math.ceil(historical / self.page_size)
        assert pages >= self.routed_pages
        self._start("router_scan")
        if self.fast_router is None:
            self.router.conditional_router_page_lse(
                query, self.base_cache[:, :, :historical], self.residual_cache[:, :, :historical],
                self.base_right, self.base_bias, self.residual_query,
                self.shared_rope["cos"][:historical], self.shared_rope["sin"][:historical],
                self.query_code, self.router_output[:, :, :, :pages], self.scaling, False)
        else:
            self.router.conditional_router_query_code(query, self.residual_query, self.query_code)
            self.fast_router.router_page_lse_fast(
                query, self.base_cache[:, :, :historical], self.residual_cache[:, :, :historical],
                self.base_right, self.base_bias,
                self.shared_rope["cos"][:historical], self.shared_rope["sin"][:historical],
                self.query_code, self.router_output[:, :, :, :pages], self.scaling)
        self._end("router_scan")
        self._start("page_selection")
        self.router.select_full_group_max_pages_and_pack(
            self.router_output[:, :, :, :pages], self.selected_pages, self.support_ids,
            int(historical), int(end))
        self._end("page_selection")
        if self.offload:
            self._start("key_refresh")
            self.slot_extension.refresh(
                self.key_pointer, self.slot_key_cache, self.support_ids, self.slot_resident,
                self.slot_lookup, self.selected_slots, self.slot_missing, self.slot_counts, True)
            self._end("key_refresh")
        self._start("sparse_attention")
        self.slot_attention(
            query, self.slot_key_cache if self.offload else self.key_cache, self.value_cache,
            self.support_ids, self.selected_slots if self.offload else self.support_ids,
            self.slot_workspace, scale=self.scaling)
        self._end("sparse_attention")
        return self.attention_output

    def _project(self, attention):
        batch, tokens = attention.shape[0], attention.shape[2]
        output = attention.transpose(1, 2).reshape(batch * tokens, -1).contiguous()
        if not self.gather_output:
            projected = F.linear(output, self.o_weight)
            if self.world > 1:
                dist.all_reduce(projected)
            return projected.view(batch, tokens, -1)
        self._start("output_allgather")
        gathered = (self.gathered if tokens == 1 and batch == self.batch else
                    torch.empty(self.world, *output.shape, device="cuda", dtype=output.dtype))
        dist.all_gather_into_tensor(gathered, output)
        self._end("output_allgather")
        full = gathered.permute(1, 0, 2).reshape(batch * tokens, -1)
        return F.linear(full, self.o_weight).view(batch, tokens, -1)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        batch, tokens, _ = hidden_states.shape
        assert position_embeddings is not None
        query = F.linear(hidden_states, self.q_weight).view(batch, tokens, self.query_heads, HEAD_DIM).transpose(1, 2)
        key = F.linear(hidden_states, self.k_weight).view(batch, tokens, self.kv_heads, HEAD_DIM).transpose(1, 2)
        value = F.linear(hidden_states, self.v_weight).view(batch, tokens, self.kv_heads, self.value_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        cos_half = (cos[0] if cos.ndim == 3 else cos)[:, :64].contiguous()
        sin_half = (sin[0] if sin.ndim == 3 else sin)[:, :64].contiguous()
        if tokens > 1:
            # Prefill of one request into its batch row, from position 0.
            assert batch == 1 and tokens <= self.capacity
            if self.layer_idx == 0:
                self.shared_rope["cos"][:tokens].copy_(cos_half)
                self.shared_rope["sin"][:tokens].copy_(sin_half)
            self._start("routing_codes_and_key_offload")
            self._write_prefill(key, value, cos_half, sin_half, self.prefill_row)
            self._end("routing_codes_and_key_offload")
            self._start("prefill_attention")
            padded = F.pad(value, (0, HEAD_DIM - self.value_dim))
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attention = F.scaled_dot_product_attention(
                    query, key, padded, is_causal=True, scale=self.scaling, enable_gqa=True)[..., :self.value_dim]
            self._end("prefill_attention")
            self.length = tokens
        else:
            start, end = self.length, self.length + 1
            assert batch == self.batch and start >= RECENT_TOKENS and end <= self.capacity
            if self.layer_idx == 0:
                self.shared_rope["cos"][start:end].copy_(cos_half)
                self.shared_rope["sin"][start:end].copy_(sin_half)
            self._start("attention_block")
            self._start("cache_append")
            self._append_decode(key, value, cos_half, sin_half, start)
            self._end("cache_append")
            attention = self._decode_attention(query, end)
            self._end("attention_block")
            self.length = end
        return self._project(attention), None


def install(model, *, harness, router, bank, value_checkpoint, batch, capacity, page_size, routed_tokens,
            sink_tokens, offload, value_reuse, build_root, rank=0, world=1):
    """Replace every attention layer; harness is the frozen TP1 module (slot kernels, chunked MLP)."""
    shared_rope = {name: torch.empty(capacity, 64, device="cuda", dtype=torch.bfloat16)
                   for name in ("cos", "sin")}
    slots = slot_extension(harness, build_root, routed_tokens + RECENT_TOKENS) if offload else None
    kv_count = KV_HEADS // world
    layers = []
    for index, layer in enumerate(model.model.layers):
        factors = load_file(str(Path(bank) / f"layer_{index:03d}.safetensors"))
        weights = (compressed_value_weights(layer.self_attn, Path(value_checkpoint), index, rank * kv_count, kv_count)
                   if value_checkpoint else None)
        replacement = BasisAttention(
            layer.self_attn, factors, value_weights=weights, batch=batch, capacity=capacity, page_size=page_size,
            routed_tokens=routed_tokens, sink_tokens=sink_tokens, offload=offload, router=router,
            slot_extension=slots, slot_attention=harness.slot_indexed_attention,
            shared_rope=shared_rope, rank=rank, world=world)
        if value_reuse:
            replacement.read_base_from_value()
        layer.self_attn = replacement
        layer.mlp = harness.ChunkedTokenwise(layer.mlp, chunk_size=1024)
        layer.input_layernorm = harness.ChunkedTokenwise(layer.input_layernorm, chunk_size=2048)
        layer.post_attention_layernorm = harness.ChunkedTokenwise(layer.post_attention_layernorm,
                                                                  chunk_size=2048)
        layers.append(replacement)
    model.model.norm = harness.ChunkedTokenwise(model.model.norm, chunk_size=2048)
    return layers
