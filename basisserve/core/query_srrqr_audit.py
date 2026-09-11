"""Full-Gram isometry for query-position strong-RRQR diagnostics."""

import numpy as np
import torch
from scipy.linalg import qr

from basisserve.core.strong_rrqr import strong_rrqr_basis


def full_gram_feature(gram):
    """Return A with A.T @ A = G, retaining the complete PSD spectrum."""
    work = np.asarray(gram, dtype=np.float64)
    assert work.ndim == 2 and work.shape[0] == work.shape[1]
    assert np.isfinite(work).all()
    work = (work + work.T) * .5
    values, vectors = np.linalg.eigh(work)
    scale = float(np.max(np.abs(values)))
    tolerance = 64 * np.finfo(float).eps * len(values) * scale
    assert scale > 0 and values[0] >= -tolerance
    feature = np.sqrt(np.maximum(values, 0.))[:, None] * vectors.T
    difference = float(np.max(np.abs(feature.T @ feature - work)))
    assert difference <= max(tolerance, np.finfo(float).tiny)
    return feature, dict(minimum_eigenvalue=float(values[0]),
                         clipped_roundoff_eigenvalues=int(np.count_nonzero(values < 0)),
                         gram_max_absolute_difference=difference,
                         feature_shape=list(feature.shape), spectral_truncation=False)


def selection_geometry(matrix, selected):
    rows = matrix[:, selected]
    q, r = qr(rows, mode='economic')
    singular = np.linalg.svd(rows, compute_uv=False)
    assert singular[-1] > np.finfo(float).eps * max(rows.shape) * singular[0]
    residual = matrix - q @ (q.T @ matrix)
    total = float(np.square(matrix).sum())
    return dict(condition_number=float(singular[0] / singular[-1]),
                log_volume=float(np.log(np.abs(np.diag(r))).sum()),
                residual_energy_fraction=float(np.square(residual).sum()) / total,
                maximum_residual_energy=float(np.square(residual).sum(axis=0).max()))


def audit_srrqr_bin(gram, positions, *, count=8, bounds=(2., 1.01), max_swaps=512):
    assert len(positions) == len(gram) and len(set(positions)) == len(positions)
    assert all(np.isfinite(f) and f > 1 for f in bounds)
    feature, isometry = full_gram_feature(gram)
    _, _, pivots = qr(feature, pivoting=True, mode='economic')
    initial = list(map(int, pivots[:count]))
    initial_geometry = selection_geometry(feature, initial)
    results = []
    for bound in bounds:
        # Small full-spectrum feature avoids the existing solver's large row Gram.
        fitted = strong_rrqr_basis(torch.from_numpy(feature), count,
                                   bound=bound, max_swaps=max_swaps)
        diagnostic = fitted.diagnostics.to_dict()
        selected = diagnostic['selected_columns']
        geometry = selection_geometry(feature, selected)
        assert diagnostic['converged']
        assert diagnostic['final_max_rho'] <= bound * (1 + 32 * np.finfo(float).eps)
        assert geometry['log_volume'] >= initial_geometry['log_volume'] - 1e-10
        # Evaluate geometry in FP64, not using the solver's returned FP32 basis.
        assert np.isclose(geometry['residual_energy_fraction'],
                          diagnostic['relative_weighted_error'], rtol=1e-10, atol=1e-12)
        results.append(dict(bound=bound, selected_positions=[positions[i] for i in selected],
                            identical_sequence=selected == initial,
                            identical_set=set(selected) == set(initial),
                            shared_positions=len(set(selected) & set(initial)),
                            volume_ratio=float(np.exp(geometry['log_volume'] - initial_geometry['log_volume'])),
                            geometry=geometry, diagnostics=diagnostic))
    return dict(isometry=isometry, cpqr_positions=[positions[i] for i in initial],
                cpqr_geometry=initial_geometry, results=results)
