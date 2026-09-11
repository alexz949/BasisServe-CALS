import pytest

from evaluation.eval_qwen35_hybrid_gsm8k import validate_gsm8k_samples


def test_each_question_has_both_filter_records():
    rows = [{'doc_id': i, 'filter': f} for i in range(3)
            for f in ('strict-match', 'flexible-extract')]
    validate_gsm8k_samples(rows, 3)
    with pytest.raises(AssertionError):
        validate_gsm8k_samples(rows[:-1], 3)
    with pytest.raises(AssertionError):
        validate_gsm8k_samples(rows[:-1] + [rows[0]], 3)
