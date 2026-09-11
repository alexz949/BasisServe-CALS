"""Compare feature-space CPQR with the existing position-Gram selector."""

import numpy as np
import torch
from scipy.linalg import qr

from basisserve.core.query_position_sampling import select_stratified_query_positions


def compare_bin(features, positions, count):
    """Rows are candidate positions; columns concatenate window/head features."""
    from basisserve.core.query_position_sampling import select_positions_pivoted_gram

    x = np.asarray(features, dtype=np.float64)
    assert x.ndim == 2 and x.shape[0] == len(positions)
    assert np.isfinite(x).all() and 0 < count <= min(x.shape)
    gram = torch.from_numpy(x @ x.T)
    old, pivots, _ = select_positions_pivoted_gram(gram, positions, count)
    _, _, permutation = qr(x.T, pivoting=True, mode='economic')
    new = [positions[int(i)] for i in permutation[:count]]

    def geometry(selected):
        rows = x[[positions.index(p) for p in selected]]
        _, singular, vh = np.linalg.svd(rows, full_matrices=False)
        cutoff = np.finfo(np.float64).eps * max(rows.shape) * singular[0]
        rank = int(np.count_nonzero(singular > cutoff))
        basis = vh[:rank]
        residual = x - (x @ basis.T) @ basis
        total = float(np.square(x).sum())
        return dict(numerical_rank=rank,
                    condition_number=float(singular[0] / singular[-1]) if rank == count else None,
                    residual_energy_fraction=float(np.square(residual).sum()) / total if total else 0.,
                    maximum_residual_energy=float(np.square(residual).sum(axis=1).max()),
                    selected_singular_values=singular.tolist())

    # Gaps along the Cholesky sequence distinguish robust pivots from near ties.
    gaps = []
    for step, position in enumerate(old):
        if step:
            rows = x[[positions.index(p) for p in old[:step]]]
            _, s, vh = np.linalg.svd(rows, full_matrices=False)
            rank = int(np.count_nonzero(s > np.finfo(float).eps * max(rows.shape) * s[0]))
            residual = x - (x @ vh[:rank].T) @ vh[:rank]
        else:
            residual = x
        energy = np.square(residual).sum(axis=1)
        energy[[positions.index(p) for p in old[:step]]] = -np.inf
        ordered = np.sort(energy)[::-1]
        gaps.append(float((ordered[0] - ordered[1]) / ordered[0])
                    if len(ordered) > 1 and np.isfinite(ordered[1]) and ordered[0] > 0 else None)
    return dict(cholesky_positions=old, cpqr_positions=new,
                identical_sequence=old == new, identical_set=set(old) == set(new),
                shared_positions=len(set(old) & set(new)),
                cholesky_pivot_diagnostics=pivots, relative_pivot_gaps=gaps,
                cholesky_geometry=geometry(old), cpqr_geometry=geometry(new))


@torch.inference_mode()
def audit_queries(queries, positions, *, context_length=32768, num_bins=4,
                  queries_per_bin=8, whitening_eps=1e-6):
    """Recompute FP64 whitening; the production returned whitening is FP32."""
    queries = queries.cpu()
    reference, _, grams = select_stratified_query_positions(
        queries, positions, context_length=context_length, num_bins=num_bins,
        queries_per_bin=queries_per_bin, whitening_eps=whitening_eps)
    windows, candidates, heads, dim = queries.shape
    moment = torch.zeros(heads, dim, dim, dtype=torch.float64)
    for window in queries:
        q = window.double().permute(1, 0, 2)
        moment += q.transpose(-1, -2) @ q
    moment /= windows * candidates
    values, vectors = torch.linalg.eigh((moment + moment.transpose(-1, -2)) * .5)
    keep = values > whitening_eps * values[:, -1:]
    inv = values.clamp_min(torch.finfo(torch.float64).tiny).rsqrt().masked_fill(~keep, 0.)
    whitening = (vectors * inv[:, None]) @ vectors.transpose(-1, -2)
    records = []
    for b, gram in enumerate(grams):
        indices = [i for i, p in enumerate(positions) if p * num_bins // context_length == b]
        # Materialize one bin at a time, retaining every window and head.
        features = torch.empty(len(indices), windows * heads * dim, dtype=torch.float64)
        for w, window in enumerate(queries):
            q = window[indices].double().permute(1, 0, 2) @ whitening
            features[:, w * heads * dim:(w + 1) * heads * dim] = q.permute(1, 0, 2).reshape(len(indices), -1)
        features /= (windows * heads) ** .5
        observed = features @ features.T
        difference = float((observed - gram).abs().max())
        torch.testing.assert_close(observed, gram, rtol=1e-10, atol=1e-10)
        record = compare_bin(features.numpy(), [positions[i] for i in indices], queries_per_bin)
        original = reference['bins'][b]['pivot_order']
        record.update(bin=b, gram_max_absolute_difference=difference,
                      production_positions=original,
                      production_matches_feature_cholesky=original == record['cholesky_positions'],
                      production_matches_cpqr=original == record['cpqr_positions'])
        records.append(record)
    return dict(query_shape=list(queries.shape), context_length=context_length,
                num_bins=num_bins, queries_per_bin=queries_per_bin,
                whitening_eps=whitening_eps, reference_selection=reference, bins=records,
                scope='Selector geometry only; no residual fitting or routing quality measurement')
