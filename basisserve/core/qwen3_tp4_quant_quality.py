"""TP4 quality reference: KVQuant NUQ4 numerics plus real INT4/FP8 wire.

KV is quantize/dequantize simulated and stored in BF16. This is NOT a packed
KV-cache deployment kernel or a performance benchmark. All arms use the same
SDPA, padded V, global decoder and TP4 output path.
"""
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from basisserve.core.qwen3_8b_tp4_decode import (
    fold_local_c1_value_projection, load_qwen3_tp4_c1_factor_layer,
)
from basisserve.kernels.int4_wire import all_gather_int4, pack_int4, unpack_int4
from basisserve.kernels.fp8_wire import FP8_E4M3_MAX, quantize_e4m3_static


def gather_rows(local):
    """Equal-width TP shards, source-major along features, token-major rows."""
    world = dist.get_world_size()
    received = torch.empty((world * local.shape[0], local.shape[1]),
                           device=local.device, dtype=local.dtype)
    dist.all_gather_into_tensor(received, local.contiguous())
    return received.reshape(world, *local.shape).permute(1, 0, 2).flatten(1)


def gather_e4m3(local, source_scales):
    """One byte per coordinate; calibrated scales are replicated offline."""
    codes = quantize_e4m3_static(local, source_scales[dist.get_rank()])
    received = gather_rows(codes.view(torch.uint8))
    row_scales = source_scales.repeat_interleave(local.shape[-1])
    return (received.view(torch.float8_e4m3fn).float()*row_scales).to(local.dtype)


def nuq_reconstruct(upstream, data, params, *, dynamic):
    """Exactly the existing KVQuant recipe, including global-token V groups."""
    hi, lo, lut = params
    hi = hi.flatten().to(data.device)
    lo = lo.flatten().to(data.device)
    mask = (upstream.get_outliers_dynamic(data, channel=-1, thresh=0.99)
            if dynamic else upstream.get_outliers(
                data, channel=0, outlier_threshold_upper=hi,
                outlier_threshold_lower=lo))
    result = upstream.quant_fn_nuq_recon(
        data.float(), bits=4, qchannel=-1 if dynamic else 0,
        dynamicquantization=dynamic, include_sparse=True, outlier_mask=mask,
        maxval=hi, minval=lo, lut=lut, first_few_fp16=-1)
    assert torch.isfinite(result).all()
    return result.to(data.dtype)


def simulate_padded_wire(value, rank):
    """Single-GPU oracle: same four INT4 groups in HF's padded head slots."""
    shape = value.shape
    heads = value.reshape(-1, 32, 128)
    result = heads.clone()
    for source in range(4):
        local = heads[:, source*8:(source+1)*8, :rank].reshape(-1, 8*rank)
        decoded = unpack_int4(pack_int4(local), 8*rank, value.dtype)
        result[:, source*8:(source+1)*8, :rank] = decoded.reshape(-1, 8, rank)
    return result.reshape(shape)


class Qwen3TP4QuantQualityAttention(nn.Module):
    def __init__(self, base, factors, params, upstream):
        super().__init__()
        assert dist.get_world_size() == 4
        self.layer_idx = factors.layer_index
        self.rank = factors.source_rank
        self.process_rank = dist.get_rank()
        self.scaling = base.scaling
        self.q_proj, self.k_proj = base.q_proj, base.k_proj
        self.q_norm, self.k_norm = base.q_norm, base.k_norm
        encoders = factors.encoders[2*self.process_rank:2*self.process_rank+2]
        weight, bias = fold_local_c1_value_projection(
            base.v_proj.weight, encoders.to(base.v_proj.weight), base.v_proj.bias)
        self.register_buffer('v_weight', weight)
        self.register_buffer('v_bias', bias)
        self.register_buffer('decoder', factors.decoders.reshape(32*self.rank, 4096).to(weight))
        hi, lo, lut = params[f'{self.layer_idx}.k']
        start = self.process_rank * 256
        self.key_params = (hi.flatten()[start:start+256].to(weight.device),
                           lo.flatten()[start:start+256].to(weight.device), lut)
        self.value_params = params[f'{self.layer_idx}.v']
        self.upstream = upstream
        self.quant_k = False
        self.quant_v = False
        self.wire = 'bf16'
        self.observe_wire = False
        self.register_buffer('wire_amax', torch.zeros((), device=weight.device))
        self.register_buffer('wire_scales', torch.ones(4, device=weight.device))
        self.register_buffer('wire_clipped', torch.zeros((), device=weight.device, dtype=torch.int64))
        self.wire_elements = 0
        self.register_buffer('key_cache', None, persistent=False)
        self.register_buffer('value_cache', None, persistent=False)
        self.factor_sha256 = factors.sha256

    def reset_cache(self):
        self.key_cache = self.value_cache = None

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        assert past_key_values is None
        batch, tokens, _ = hidden_states.shape
        query = self.q_norm(self.q_proj(hidden_states).reshape(batch, tokens, 8, 128)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).reshape(batch, tokens, 2, 128))
        value = F.linear(hidden_states, self.v_weight, self.v_bias)
        if self.quant_k:
            key = nuq_reconstruct(self.upstream, key.reshape(-1, 256),
                                  self.key_params, dynamic=False).reshape_as(key)
        if self.quant_v:
            # Preserve the original across-ALL-KV-heads grouping exactly.
            # This auxiliary BF16 gather is reference-only, NOT wire savings.
            global_value = gather_rows(value.reshape(-1, 2*self.rank))
            global_value = nuq_reconstruct(self.upstream, global_value,
                                           self.value_params, dynamic=True)
            value = global_value[:, self.process_rank*2*self.rank:(self.process_rank+1)*2*self.rank]
        value = value.reshape(batch, tokens, 2, self.rank).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key.transpose(1, 2), *position_embeddings)
        past = 0 if self.key_cache is None else self.key_cache.shape[2]
        self.key_cache = key if past == 0 else torch.cat((self.key_cache, key), dim=2)
        self.value_cache = value if past == 0 else torch.cat((self.value_cache, value), dim=2)
        mask = attention_mask
        if past and tokens > 1:
            mask = torch.arange(past+tokens, device=key.device)[None, :] <= (
                past + torch.arange(tokens, device=key.device)[:, None])
        # Padding only for SDPA geometry; never included in quantizer statistics.
        padded_v = F.pad(self.value_cache, (0, 128-self.rank))
        local = F.scaled_dot_product_attention(
            query, self.key_cache, padded_v, attn_mask=mask,
            dropout_p=0, is_causal=(past == 0 and tokens > 1 and mask is None),
            scale=self.scaling, enable_gqa=True)[..., :self.rank]
        local = local.transpose(1, 2).reshape(batch*tokens, 8*self.rank).contiguous()
        if self.observe_wire:
            torch.maximum(self.wire_amax, local.float().abs().amax(), out=self.wire_amax)
        if self.wire == 'int4':
            gathered = all_gather_int4(local)
        elif self.wire == 'e4m3':
            self.wire_clipped.add_((local.float().abs() > self.wire_scales[self.process_rank]*FP8_E4M3_MAX).sum())
            self.wire_elements += local.numel()
            gathered = gather_e4m3(local, self.wire_scales)
        else:
            assert self.wire == 'bf16'
            gathered = gather_rows(local)
        return (gathered @ self.decoder).reshape(batch, tokens, 4096), None


def install_quant_quality(model, checkpoint, params, upstream):
    modules = []
    for i, layer in enumerate(model.model.layers):
        factors = load_qwen3_tp4_c1_factor_layer(checkpoint, i)
        layer.self_attn = Qwen3TP4QuantQualityAttention(layer.self_attn, factors, params, upstream)
        modules.append(layer.self_attn)
    return modules


def vocab_parallel_nll(local_logits, labels):
    """FP32 summed CE over disjoint equal vocabulary shards, no logit gather."""
    logits = local_logits.float()
    maximum = logits.amax(-1)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    denominator = (logits-maximum[:, None]).exp().sum(-1)
    dist.all_reduce(denominator)
    local_labels = labels - dist.get_rank()*logits.shape[-1]
    valid = (local_labels >= 0) & (local_labels < logits.shape[-1])
    selected = logits.gather(1, local_labels.clamp(0, logits.shape[-1]-1)[:, None]).squeeze(1)
    selected = torch.where(valid, selected, 0)
    dist.all_reduce(selected)
    return (maximum + denominator.log() - selected).sum()
