import torch

from basisserve.core.c1_lrqk import LRQKConfig, LRQKState, prefill_factors
from evaluation.lrqk_chunked_prefill import head_blocked_factors, HeadBlockedLRQKState


@torch.inference_mode()
def test_head_blocks_match_full_factors_and_stopping():
    torch.manual_seed(4)
    q = torch.randn(1, 12, 43, 16, dtype=torch.bfloat16)
    k = torch.randn(1, 3, 43, 16, dtype=torch.bfloat16)
    for tolerance in (0., 1.e8):
        cfg = LRQKConfig(rank=4, topk=12, recent=4, tolerance=tolerance)
        gen = torch.Generator().manual_seed(cfg.seed + 3)
        aq = torch.randn(1, 12, 43, 4, generator=gen)
        ak = torch.randn(1, 12, 43, 4, generator=gen)
        expected = prefill_factors(q.float(), k.repeat_interleave(4, dim=1).float(),
            aq, ak, cfg.prefill_iterations, tolerance)
        actual = head_blocked_factors(q, k, cfg, 3)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=2.e-5, atol=2.e-5)
        baseline = LRQKState(q, k, cfg, 3)
        blocked = HeadBlockedLRQKState(q, k, cfg, 3)
        assert blocked.ak.dtype == torch.bfloat16
        torch.testing.assert_close(blocked.selected, baseline.selected, rtol=0, atol=0)
