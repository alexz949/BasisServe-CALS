"""A lean Llama forward around per-rank attention modules (BasisKV), with the non-attention computation of upstream
ShadowKV's models.llama.Llama and of its TP port (basisserve/core/shadowkv_upstream_tp.py):
  * embedding and lm_head replicated as plain tensors; flashinfer RMSNorm (models/tensor_op.layer_norm);
  * MLP with a fused gate_up projection, vLLM silu_and_mul, down_proj, and an AllReduce of down_proj's
    row-parallel output under TP (upstream post_attention_compute);
  * attention modules project and reduce their own output (o_proj + AllReduce inside the module);
  * long prefills run the MLP in 8192-token chunks (upstream chunks post-attention work at 8192 tokens).
The HF modules it replaces are released, so weights are not held twice. Numerically this differs from the HF path
only by kernel rounding (flashinfer RMSNorm, fused silu_and_mul).
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

CHUNK = 8192


def _full(weight):
    weight = weight.detach()
    return (weight.full_tensor() if hasattr(weight, "full_tensor") else weight).contiguous()


def _local(weight):
    weight = weight.detach()
    return (weight.to_local() if hasattr(weight, "to_local") else weight).contiguous()


def _inner(module):
    return getattr(module, "inner", module)  # unwrap evaluation.chunked_prefill_mlp.ChunkedTokenwise


class LeanLlama:
    def __init__(self, model, attention, world):
        from flashinfer.norm import rmsnorm
        import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

        self.rmsnorm, self.silu_and_mul = rmsnorm, torch.ops._C.silu_and_mul
        self.world, self.attention = world, attention
        inner = model.model
        self.embed = _full(inner.embed_tokens.weight)
        self.lm_head = _full(model.lm_head.weight)
        norm = _inner(inner.norm)
        self.norm_weight, self.norm_eps = _full(norm.weight), norm.variance_epsilon
        self.rotary = inner.rotary_emb
        self.layers = []
        for layer in inner.layers:
            mlp = _inner(layer.mlp)
            first, second = _inner(layer.input_layernorm), _inner(layer.post_attention_layernorm)
            self.layers.append(dict(
                input_weight=_full(first.weight), input_eps=first.variance_epsilon,
                post_weight=_full(second.weight), post_eps=second.variance_epsilon,
                gate_up=torch.cat((_local(mlp.gate_proj.weight), _local(mlp.up_proj.weight)), dim=0).contiguous(),
                down=_local(mlp.down_proj.weight)))
            layer.mlp = layer.input_layernorm = layer.post_attention_layernorm = torch.nn.Identity()
        inner.embed_tokens = inner.norm = model.lm_head = torch.nn.Identity()
        torch.cuda.empty_cache()

    def norm(self, hidden, weight, eps):
        return self.rmsnorm(hidden.reshape(-1, hidden.shape[-1]), weight, eps).view_as(hidden)

    def mlp(self, hidden, layer):
        gate_up = F.linear(hidden, layer["gate_up"])
        out = torch.empty(gate_up.shape[:-1] + (gate_up.shape[-1] // 2,), dtype=gate_up.dtype, device=gate_up.device)
        self.silu_and_mul(out, gate_up)
        del gate_up
        down = F.linear(out, layer["down"])
        if self.world > 1:
            dist.all_reduce(down)
        return down

    @torch.inference_mode()
    def forward(self, tokens, positions):
        """Logits [B, vocab] (float32) of the last position."""
        hidden = F.embedding(tokens, self.embed)
        position_embeddings = self.rotary(hidden, positions)
        length = hidden.shape[1]
        for attention, layer in zip(self.attention, self.layers):
            hidden.add_(attention(self.norm(hidden, layer["input_weight"], layer["input_eps"]),
                                  position_embeddings=position_embeddings)[0])
            for start in range(0, length, CHUNK):
                part = hidden[:, start:start + CHUNK]
                part.add_(self.mlp(self.norm(part, layer["post_weight"], layer["post_eps"]), layer))
        last = self.norm(hidden[:, -1:], self.norm_weight, self.norm_eps)
        return F.linear(last, self.lm_head)[:, -1].float()
