"""Bound prefill RoPE temporaries on inference-owned Q/K tensors."""
import torch
from transformers.models.llama import modeling_llama


def install_chunked_llama_rope(chunk_tokens=1024):
    original=modeling_llama.apply_rotary_pos_emb
    @torch.inference_mode()
    def rotary(q,k,cos,sin,unsqueeze_dim=1):
        assert unsqueeze_dim==1
        if q.shape[2]<=chunk_tokens:
            return original(q,k,cos,sin,unsqueeze_dim=unsqueeze_dim)
        # Runtime Q/K are fresh projection outputs; consume their pre-RoPE values.
        for start in range(0,q.shape[2],chunk_tokens):
            stop=start+chunk_tokens
            qr,kr=original(q[:,:,start:stop],k[:,:,start:stop],cos[:,start:stop],sin[:,start:stop],unsqueeze_dim=1)
            q[:,:,start:stop].copy_(qr);k[:,:,start:stop].copy_(kr)
        return q,k
    modeling_llama.apply_rotary_pos_emb=rotary
