import pytest
import torch

from basisserve.core.gqa_routed_ov_joint import _cholesky_with_jitter


@pytest.mark.parametrize('invalid_result',['factor','solution'])
def test_nonfinite_result_with_success_status_continues_existing_jitter_search(monkeypatch,invalid_result):
    matrix = torch.tensor([[2.,.5],[.5,1.]],dtype=torch.float64)
    rhs = torch.tensor([[1.],[3.]],dtype=torch.float64)
    original_factor = torch.linalg.cholesky_ex
    original_solve = torch.cholesky_solve
    calls = 0

    def factor(*args,**kwargs):
        nonlocal calls
        calls += 1
        chol,info = original_factor(*args,**kwargs)
        if invalid_result == 'factor' and calls == 1:
            chol.fill_(float('nan'))
        return chol,info

    def solve(*args,**kwargs):
        result = original_solve(*args,**kwargs)
        if invalid_result == 'solution' and calls == 1:
            result.fill_(float('nan'))
        return result

    monkeypatch.setattr(torch.linalg,'cholesky_ex',factor)
    monkeypatch.setattr(torch,'cholesky_solve',solve)
    solution,jitter,condition = _cholesky_with_jitter(matrix,rhs,relative_jitter=0.)
    assert calls == 2
    assert jitter == pytest.approx(1.5e-12,rel=1e-12,abs=0.)
    assert condition > 0 and torch.isfinite(solution).all()
    torch.testing.assert_close((matrix+jitter*torch.eye(2))@solution,rhs)
