"""Matched full-router K reconstruction and causal selected attention mass."""

import torch

from basisserve.core.qwen35_k_routing_runtime import build_qwen35_routing_sidecar, page_support


@torch.inference_mode()
def window_metrics(value, key, queries, positions, cos, sin, *, rank, factors, budget=2048):
    batch, kv_heads, length, dim = key.shape
    query_heads = queries.shape[2]
    assert batch == 1 and query_heads % kv_heads == 0
    assert queries.shape == (batch, len(positions), query_heads, dim)
    assert positions and min(positions) >= 32 and max(positions) < length
    groups = query_heads//kv_heads
    sidecar = build_qwen35_routing_sidecar(value, key, cos, sin, rank=rank, factors=factors)
    query_factor = factors[f'residual_query_b{rank}_r{rank}']
    errors = energy = 0.0
    for start in range(0, length, 2048):
        part = sidecar[:, :, start:start+2048].repeat_interleave(groups, 1)
        predicted = part[..., :dim]+torch.einsum('bhtr,hdr->bhtd', part[..., dim:], query_factor)
        exact = key[:, :, start:start+2048].float().repeat_interleave(groups, 1)
        errors += (predicted-exact).double().square().sum().item()
        energy += exact.double().square().sum().item()
    masses = {name: [] for name in ('attention_mass', 'non_sink_attention_mass',
        'exact_attention_mass', 'exact_non_sink_attention_mass')}
    for offset, position in enumerate(positions):
        q = queries[:, offset].float()
        code = torch.cat((q, torch.einsum('bhd,hdr->bhr', q, query_factor)), -1)
        scores = (code.reshape(batch, kv_heads, groups, -1)@sidecar[:, :, :position+1].transpose(-1, -2))*(dim**-0.5)
        logits = (q.reshape(batch, kv_heads, groups, dim)@key[:, :, :position+1].float().transpose(-1, -2))*(dim**-0.5)
        supports = {}
        for label, proxy in (('', scores), ('exact_', logits)):
            ids, valid = page_support(proxy, budget=budget)
            ids = ids.clamp_max(position)[:, :, None].expand(batch, kv_heads, groups, -1)
            supports[label] = (ids, valid[:, :, None].expand_as(ids))
        for sink_label in ('', 'non_sink_'):
            if sink_label:
                logits[..., :32] = -torch.inf
            probabilities = logits.softmax(-1)
            for label, (ids, valid) in supports.items():
                mass = probabilities.gather(-1, ids).masked_fill(~valid, 0).sum(-1)
                assert torch.isfinite(mass).all() and mass.min() >= 0 and mass.max() <= 1.00001
                masses[label+sink_label+'attention_mass'].append(mass.cpu())
    assert energy > 0
    return dict(squared_error=errors, key_energy=energy, relative_mse=errors/energy,
        **{name: torch.stack(values).mean().item() for name, values in masses.items()})
