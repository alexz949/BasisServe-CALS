import numpy as np
from scipy.linalg import qr

from basisserve.core.query_srrqr_audit import audit_srrqr_bin, full_gram_feature
from basisserve.core.strong_rrqr import _rrqr_state


def test_full_gram_root_preserves_cpqr_and_singular_spectrum():
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(40, 12))
    matrix[:, -1] = matrix[:, -2]
    feature, record = full_gram_feature(matrix.T @ matrix)
    np.testing.assert_allclose(feature.T @ feature, matrix.T @ matrix, atol=1e-12)
    _, _, old = qr(matrix, mode='economic', pivoting=True)
    _, _, new = qr(feature, mode='economic', pivoting=True)
    np.testing.assert_array_equal(old[:4], new[:4])
    assert not record['spectral_truncation']


def test_rho_predicts_actual_single_swap_volume_gain():
    rng = np.random.default_rng(32)
    matrix = rng.normal(size=(11, 7))
    selected, remaining = np.arange(3), np.arange(3, 7)
    _, r, rho, _, (i, j) = _rrqr_state(matrix, selected, remaining)
    new = selected.copy()
    new[i] = remaining[j]
    _, new_r = qr(matrix[:, new], mode='economic')
    ratio = np.prod(np.abs(np.diag(new_r))) / np.prod(np.abs(np.diag(r)))
    np.testing.assert_allclose(ratio, rho, rtol=1e-12)


def test_strict_bound_triggers_swap_and_improves_volume():
    matrix = np.array([[1., 0., .9], [0., 1., .9], [0., 0., .05]])
    result = audit_srrqr_bin(matrix.T @ matrix, [63, 127, 191], count=2,
                             bounds=(2., 1.01))
    loose, strict = result['results']
    assert loose['diagnostics']['swaps'] == 0
    assert strict['diagnostics']['swaps'] >= 1
    assert strict['diagnostics']['final_max_rho'] <= 1.01
    assert strict['volume_ratio'] > 1.01
    assert set(strict['selected_positions']) == {63, 127}


def test_diagonal_geometry_already_satisfies_strong_bound():
    result = audit_srrqr_bin(np.diag([9., 4., 1.]), [0, 1, 2], count=2)
    for r in result['results']:
        assert r['diagnostics']['swaps'] == 0 and r['identical_sequence']
        assert r['diagnostics']['converged']
