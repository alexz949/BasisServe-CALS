"""Deterministic stratified pivots of uncentered, head-whitened Query Grams."""

import hashlib
import json

import torch


def candidate_positions(context_length, candidate_stride=64, excluded_query_prefix=32):
    assert context_length > 0 and candidate_stride > 0 and excluded_query_prefix >= 0
    return [p for p in range(candidate_stride-1, context_length, candidate_stride)
            if p >= excluded_query_prefix]


def canonical_hash(manifest):
    content = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def select_positions_pivoted_gram(gram, positions, count, tolerance=1e-10):
    assert gram.shape == (len(positions), len(positions)) and 0 < count <= len(positions)
    assert positions == sorted(set(positions)) and torch.isfinite(gram).all()
    gram = (gram.double()+gram.double().T)*.5
    scale = max(float(gram.diag().abs().max()), 1.)
    assert float(torch.linalg.eigvalsh(gram).min()) >= -tolerance*scale
    diagonal = gram.diag().clone()
    assert float(diagonal.min()) >= -tolerance*scale
    diagonal.clamp_min_(0)
    factor = torch.zeros(len(positions), count, dtype=torch.float64, device=gram.device)
    selected = torch.zeros(len(positions), dtype=torch.bool, device=gram.device)
    diagnostics = []
    for step in range(count):
        candidates = diagonal.masked_fill(selected, -torch.inf)
        pivot = int(candidates.argmax())  # positions are sorted: first exact maximum wins.
        energy = float(diagonal[pivot])
        original = float(gram[pivot, pivot])
        diagnostics.append({'absolute_position': positions[pivot], 'pivot_order':step,
                            'residual_energy':energy, 'original_energy':original,
                            'relative_residual_energy':energy/original if original > 0 else 0.,
                            'numerically_zero_pivot':energy <= tolerance*scale})
        selected[pivot] = True
        if energy > tolerance*scale:
            column = gram[:, pivot] - factor[:, :step] @ factor[pivot, :step]
            factor[:, step] = column / energy**.5
            diagonal -= factor[:, step].square()
            assert float(diagonal.min()) >= -tolerance*scale*10
            diagonal.clamp_min_(0)
        diagonal[selected] = 0.
    trace = float(gram.diag().sum())
    return [d['absolute_position'] for d in diagnostics], diagnostics, {
        'residual_trace_fraction': float(diagonal.sum())/trace if trace > 0 else 0.,
        'minimum_residual_diagonal':float(diagonal.min()),
    }


@torch.inference_mode()
def select_stratified_query_positions(queries, positions, *, context_length,
                                      num_bins=4, queries_per_bin=8, whitening_eps=1e-6):
    assert queries.ndim == 4 and torch.isfinite(queries).all()
    windows, candidates, heads, dim = queries.shape
    assert windows > 0 and candidates == len(positions) and positions == sorted(set(positions))
    assert num_bins > 0 and queries_per_bin > 0 and 0 < whitening_eps < 1
    assert all(0 <= p < context_length for p in positions)
    moment = torch.zeros(heads, dim, dim, dtype=torch.float64, device=queries.device)
    for window in queries:
        q = window.double().permute(1, 0, 2)
        moment += q.transpose(-1, -2) @ q
    moment /= windows*candidates
    values, vectors = torch.linalg.eigh((moment+moment.transpose(-1, -2))*.5)
    floor = whitening_eps * values[:, -1:]
    keep = values > floor
    inv = values.clamp_min(torch.finfo(torch.float64).tiny).rsqrt().masked_fill(~keep, 0.)
    whitening = (vectors * inv[:, None]) @ vectors.transpose(-1, -2)
    assert torch.isfinite(whitening).all()
    bins = [[i for i,p in enumerate(positions) if p*num_bins//context_length == b] for b in range(num_bins)]
    assert all(len(indices) >= queries_per_bin for indices in bins)
    grams = [torch.zeros(len(i),len(i),dtype=torch.float64,device=queries.device) for i in bins]
    for window in queries:
        q = window.double().permute(1, 0, 2) @ whitening
        for gram, indices in zip(grams,bins):
            x = q[:,indices]
            gram += (x @ x.transpose(-1,-2)).sum(0)/(windows*heads)
    records, selected = [], []
    for b,(gram,indices) in enumerate(zip(grams,bins)):
        gram.copy_((gram+gram.T)*.5)
        local_positions = [positions[i] for i in indices]
        pivots, diagnostics, residual = select_positions_pivoted_gram(gram,local_positions,queries_per_bin)
        norm = gram.diag().clamp_min(0).sqrt()
        correlation = gram/(norm[:,None]*norm[None]).clamp_min(torch.finfo(torch.float64).tiny)
        off = ~torch.eye(len(indices),device=gram.device,dtype=torch.bool)
        records.append({'bin':b, 'candidate_count':len(indices), 'pivot_order':pivots,
                        'selected_positions':sorted(pivots), 'pivots':diagnostics, **residual,
                        'mean_absolute_offdiagonal_correlation':float(correlation[off].abs().mean()) if off.any() else 0.,
                        'minimum_gram_eigenvalue':float(torch.linalg.eigvalsh(gram).min())})
        selected.extend(pivots)
    return {'selected_positions':sorted(selected), 'bins':records,
            'whitening_effective_ranks':keep.sum(-1).cpu().tolist(),
            'whitening_eps':whitening_eps}, whitening.float(), grams


def uniform_query_positions(context_length, count=32, terminal_fraction=1.):
    assert 0 < terminal_fraction <= 1 and context_length >= count > 0
    start = int(context_length*(1-terminal_fraction))
    result = [start+(i+1)*(context_length-start)//count-1 for i in range(count)]
    assert len(set(result)) == count
    return result
