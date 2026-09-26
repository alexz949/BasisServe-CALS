"""Structure-validated Llama adaptive C1 factors for NUQ4 quality evaluation."""

from pathlib import Path
import torch
from safetensors.torch import load_file


@torch.no_grad()
def fold_layer(attention, factors, ranks):
    assert set(factors) == {"value_coordinate_encoders", "head_output_decoders", "source_ranks"}
    assert factors["source_ranks"].tolist() == ranks
    assert len(ranks) == 8 and all(0 < rank <= 128 and rank % 16 == 0 for rank in ranks)
    encoder, decoder = factors["value_coordinate_encoders"], factors["head_output_decoders"]
    assert encoder.shape == (8, 128, max(ranks)) and decoder.shape == (32, max(ranks), 4096)
    assert torch.isfinite(encoder).all() and torch.isfinite(decoder).all()
    assert attention.v_proj.weight.shape == (1024, 4096)
    assert attention.o_proj.weight.shape == (4096, 4096)
    assert attention.v_proj.bias is None and attention.o_proj.bias is None
    device = attention.v_proj.weight.device
    encoder, decoder = encoder.to(device).float(), decoder.to(device).float()
    dense_v = attention.v_proj.weight.float()
    padded_v = torch.zeros_like(dense_v)
    padded_o = torch.zeros_like(attention.o_proj.weight).float()
    for h, rank in enumerate(ranks):
        padded_v[h * 128:h * 128 + rank] = encoder[h, :, :rank].T @ dense_v[h * 128:(h + 1) * 128]
        for q in range(h * 4, (h + 1) * 4):
            padded_o[:, q * 128:q * 128 + rank] = decoder[q, :rank].T
    attention.v_proj.weight.copy_(padded_v)
    attention.o_proj.weight.copy_(padded_o)
    return torch.tensor([h * 128 + j for h, rank in enumerate(ranks) for j in range(rank)], device=device)


@torch.no_grad()
def install_adaptive_factors(model, root, schedule, target_rank):
    c = model.config
    assert (c.model_type, c.hidden_size, c.num_hidden_layers, c.num_attention_heads,
            c.num_key_value_heads) == ("llama", 4096, 32, 32, 8)
    assert target_rank in (64, 96)
    assert len(schedule) == 32 and sum(map(sum, schedule)) == 32 * 8 * target_rank
    modules, indices = {}, {}
    for i, layer in enumerate(model.model.layers):
        factors = load_file(str(Path(root) / "selected_factors" / f"layer_{i:03d}.safetensors"))
        a = layer.self_attn
        indices[f"{i}.v"] = fold_layer(a, factors, schedule[i])
        indices[f"{i}.k"] = torch.arange(1024, device=a.k_proj.weight.device)
        modules[f"{i}.k"], modules[f"{i}.v"] = a.k_proj, a.v_proj
    return modules, indices
