"""Head-blocked LRQK prefill with shared iteration and stopping decisions."""
import torch
from torch.nn import functional as F

from basisserve.core.c1_lrqk import LRQKState, _solve, select_tokens


@torch.inference_mode()
def head_blocked_factors(q, k, config, layer):
    heads = q.shape[1]
    groups = heads // k.shape[1]
    assert heads % k.shape[1] == 0
    generator = torch.Generator(device=q.device).manual_seed(config.seed + layer)
    shape = (*q.shape[:-1], config.rank)
    aq = torch.randn(shape, device=q.device, dtype=torch.float32, generator=generator)
    ak = torch.randn(shape, device=q.device, dtype=torch.float32, generator=generator)
    for _ in range(config.prefill_iterations):
        qerrors, kerrors = [], []
        for start in range(0, heads, 8):
            stop = min(start + 8, heads)
            indices = torch.arange(start, stop, device=k.device) // groups
            qs = q[:, start:stop].float()
            ks = k.index_select(1, indices).float()
            a, b = aq[:, start:stop], ak[:, start:stop]
            aa, bb = a.transpose(-1, -2) @ a, b.transpose(-1, -2) @ b
            bq = _solve(aa, a.transpose(-1, -2) @ qs)
            bk = _solve(bb, b.transpose(-1, -2) @ ks)
            new_a = qs @ _solve(bb + bq @ bq.transpose(-1, -2),
                ks.transpose(-1, -2) @ b + bq.transpose(-1, -2), left=False)
            aa = new_a.transpose(-1, -2) @ new_a
            new_b = ks @ _solve(aa + bk @ bk.transpose(-1, -2),
                qs.transpose(-1, -2) @ new_a + bk.transpose(-1, -2), left=False)
            qerrors.append(F.mse_loss(a, new_a) * (stop - start) / heads)
            kerrors.append(F.mse_loss(b, new_b) * (stop - start) / heads)
            a.copy_(new_a)
            b.copy_(new_b)
            del qs, ks, new_a, new_b
        if max(torch.stack(qerrors).sum(), torch.stack(kerrors).sum()) < config.tolerance:
            break
    bqs, bks = [], []
    for start in range(0, heads, 8):
        stop = min(start + 8, heads)
        indices = torch.arange(start, stop, device=k.device) // groups
        a, b = aq[:, start:stop], ak[:, start:stop]
        bqs.append(_solve(a.transpose(-1, -2) @ a,
            a.transpose(-1, -2) @ q[:, start:stop].float()))
        bks.append(_solve(b.transpose(-1, -2) @ b,
            b.transpose(-1, -2) @ k.index_select(1, indices).float()))
    return aq, torch.cat(bqs, dim=1), ak, torch.cat(bks, dim=1)


class HeadBlockedLRQKState(LRQKState):
    @torch.inference_mode()
    def __init__(self, q, k, config, layer=0):
        self.config = config
        self.length = k.shape[2]
        self.steps = 0
        factors = head_blocked_factors(q, k, config, layer)
        aq, self.bq, self.ak, self.bk = [value.to(q.dtype) for value in factors]
        del factors
        assert all(torch.isfinite(value).all() for value in (aq, self.bq, self.ak, self.bk))
        self.selected = select_tokens(aq[:, :, -1:], self.ak, config)
