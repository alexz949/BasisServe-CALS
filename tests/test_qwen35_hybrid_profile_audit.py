import json
import math

import pytest

from evaluation.summarize_qwen35_hybrid import read_kl_profile, read_kl_confirmation


@pytest.mark.parametrize('corruption', [None, 'short', 'nonfinite', 'mean', 'factor', 'schedule', 'model', 'windows'])
def test_profile_audit_rejects_incomplete_or_mismatched_evidence(tmp_path, corruption):
    schedule = [64] * 8
    identity = {'model_revision': 'frozen'}
    payload = {'schedule': schedule.copy(), 'factor_sha256': 'factors',
               'verified_model_identity': identity.copy(), 'windows_sha256': 'windows',
               'window_kl': [0.125] * 128, 'mean_kl': 0.125}
    if corruption == 'short':
        payload['window_kl'].pop()
    elif corruption == 'nonfinite':
        payload['window_kl'][0] = float('nan')
    elif corruption == 'mean':
        payload['mean_kl'] = 0.25
    elif corruption == 'factor':
        payload['factor_sha256'] = 'different'
    elif corruption == 'schedule':
        payload['schedule'][0] = 32
    elif corruption == 'model':
        payload['verified_model_identity']['model_revision'] = 'different'
    elif corruption == 'windows':
        payload['windows_sha256'] = 'different'
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(payload))
    if corruption is None:
        assert read_kl_profile(path, schedule, 'factors', identity, 'windows') == payload
    else:
        with pytest.raises(AssertionError):
            read_kl_profile(path, schedule, 'factors', identity, 'windows')


@pytest.mark.parametrize('corruption', [None, 'missing_method', 'short', 'paired', 'mean', 'stderr'])
def test_confirmation_audit_recomputes_independent_paired_statistics(tmp_path, corruption):
    uniform = [0.125] * 16
    twosided = [0.125 + i / 1024 for i in range(16)]
    differences = [a - b for a, b in zip(twosided, uniform)]
    mean = sum(differences) / 16
    identity = {'model_revision': 'frozen'}
    payload = {'windows_sha256': 'windows', 'window_count': 16,
               'verified_model_identity': identity,
               'results': {name: {'window_kl': values, 'mean_kl': sum(values) / 16}
                           for name, values in [('uniform', uniform), ('twosided', twosided)]},
               'paired_kl_difference_twosided_minus_uniform': differences.copy(),
               'mean_difference': mean,
               'standard_error_across_windows': math.sqrt(sum((x - mean) ** 2 for x in differences) / 15 / 16)}
    if corruption == 'missing_method':
        del payload['results']['twosided']
    elif corruption == 'short':
        payload['results']['uniform']['window_kl'].pop()
    elif corruption == 'paired':
        payload['paired_kl_difference_twosided_minus_uniform'][0] = 0.5
    elif corruption == 'mean':
        payload['mean_difference'] = 0.5
    elif corruption == 'stderr':
        payload['standard_error_across_windows'] = 0.5
    path = tmp_path / 'confirmation.json'
    path.write_text(json.dumps(payload))
    if corruption is None:
        assert read_kl_confirmation(path, identity, 'windows') == payload
    else:
        with pytest.raises(AssertionError):
            read_kl_confirmation(path, identity, 'windows')
