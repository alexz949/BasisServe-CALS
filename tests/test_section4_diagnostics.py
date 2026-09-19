import torch

from evaluation.eval_llama_section4_diagnostics import window_metrics


def test_window_metrics_skips_undefined_historical_metrics_at_token_63():
    generator = torch.Generator().manual_seed(17)
    value = torch.randn(1, 1, 128, 4, generator=generator)
    key = torch.randn(1, 1, 128, 4, generator=generator)
    queries = torch.randn(1, 2, 2, 4, generator=generator)
    output_weight = torch.randn(8, 8, generator=generator)

    metrics = window_metrics(
        value,
        key,
        queries,
        [63, 127],
        torch.empty(0),
        torch.empty(0),
        {},
        output_weight,
    )

    for value in metrics["exact_k"].values():
        assert torch.isfinite(torch.tensor(value))
    assert metrics["exact_k"]["page_kl_exact_to_proxy"] == 0.0
    assert metrics["exact_k"]["routed_page_recall"] == 1.0
