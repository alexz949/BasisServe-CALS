import torch

from evaluation.qwen35_hybrid_local_errors import error_terms, normalized_errors


def test_cross_term_can_cancel_two_nonzero_component_errors():
    native = torch.tensor([[1., 2.]], dtype=torch.float64)
    values = normalized_errors(error_terms(native, 2 * native, native))
    assert values['err_v'] == 1.
    assert values['err_ag_given_v'] == 0.25
    assert values['err_composed'] == 0.
    assert values['normalized_cross_term'] == -2.


def test_error_terms_are_additive_over_row_chunks():
    torch.manual_seed(15)
    native, v_only, composed = [torch.randn(17, 11, dtype=torch.float64) for _ in range(3)]
    full = normalized_errors(error_terms(native, v_only, composed))
    totals = {}
    for start in range(0, 17, 4):
        partial = error_terms(native[start:start + 4], v_only[start:start + 4], composed[start:start + 4])
        totals = {key: totals.get(key, 0.) + value for key, value in partial.items()}
    chunked = normalized_errors(totals)
    for key in full:
        torch.testing.assert_close(torch.tensor(chunked[key]), torch.tensor(full[key]))
