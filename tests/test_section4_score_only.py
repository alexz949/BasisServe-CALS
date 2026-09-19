import torch

from evaluation.capture_llama_section4_score_only_statistics import (
    length_stratified_positions,
    prefix_grams,
)


def test_length_stratified_positions_are_fixed_midpoints():
    assert length_stratified_positions(65536, 64) == list(range(512, 65536, 1024))
    assert length_stratified_positions(65536, 32) == list(range(1024, 65536, 2048))


def test_prefix_grams_are_causal_and_preserve_requested_order():
    residual = torch.arange(40 * 2, dtype=torch.float32).reshape(1, 40, 2)
    positions = [39, 35]
    actual = prefix_grams(residual, positions)
    expected = []
    for position in positions:
        rows = residual[:, 32:position + 1]
        expected.append(torch.einsum("gtd,gte->gde", rows, rows))
    torch.testing.assert_close(actual, torch.stack(expected, dim=1))
