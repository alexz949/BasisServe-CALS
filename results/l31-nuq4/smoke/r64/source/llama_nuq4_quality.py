"""Structure-only Llama C1 factors and the frozen NUQ4 quality protocol."""

from pathlib import Path
import torch
from safetensors.torch import load_file
from basisserve.core.qwen3_kv4_fp8_quality import QualityLinear


def load_llama_factors(root, layer, rank):
    data = load_file(str(Path(root) / f"layer_{layer:03d}.safetensors"))
    assert set(data) == {"value_coordinate_encoders", "head_output_decoders"}
    encoder, decoder = data["value_coordinate_encoders"], data["head_output_decoders"]
    assert encoder.shape == (8, 128, rank) and decoder.shape == (32, rank, 4096)
    assert torch.isfinite(encoder).all() and torch.isfinite(decoder).all()
    return encoder, decoder


@torch.no_grad()
def install_llama_factors(model, root, rank):
    c = model.config
    assert (c.model_type, c.hidden_size, c.num_hidden_layers, c.num_attention_heads,
            c.num_key_value_heads) == ("llama", 4096, 32, 32, 8)
    assert rank in (64, 96)
    projections, modules, indices = {}, {}, {}
    for i, layer in enumerate(model.model.layers):
        a = layer.self_attn
        enc, dec = load_llama_factors(root, i, rank)
        device = a.v_proj.weight.device
        enc, dec = enc.to(device).float(), dec.to(device).float()
        v = a.v_proj.weight.float()
        ev, od = torch.zeros_like(v), torch.zeros_like(a.o_proj.weight).float()
        for h in range(8):
            ev[h*128:h*128+rank] = enc[h].T @ v[h*128:(h+1)*128]
            for q in range(h*4,(h+1)*4):
                od[:,q*128:q*128+rank] = dec[q].T
        a.v_proj = QualityLinear(ev.to(torch.bfloat16))
        a.o_proj = QualityLinear(od.to(torch.bfloat16))
        projections[f"{i}.encoder"], projections[f"{i}.decoder"] = a.v_proj, a.o_proj
        modules[f"{i}.k"], modules[f"{i}.v"] = a.k_proj, a.v_proj
        indices[f"{i}.k"] = torch.arange(1024, device=device)
        indices[f"{i}.v"] = torch.tensor([h*128+j for h in range(8) for j in range(rank)], device=device)
    return projections, modules, indices
