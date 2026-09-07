import torch

from basisserve.core.residual_page_ranking import (
    PageRoutingExample, proxy_scores, page_state, boundary_pairs,
    page_gradient, repair_page_boundary, mass_coverage,
)


def fixture(dtype=torch.float64):
    torch.manual_seed(173)
    query = torch.randn(2, 4, dtype=dtype)
    base = torch.randn(12, 4, dtype=dtype)
    residual = torch.randn(12, 4, dtype=dtype)
    mass = torch.randn(2, 4, dtype=dtype).softmax(-1)
    example = PageRoutingExample(query, base, residual, mass, page_size=3, page_budget=2)
    return example, torch.randn(4, 2, dtype=dtype), torch.randn(2, 4, 2, dtype=dtype)


def test_page_derivative_includes_head_normalization():
    example, e, u = fixture()
    scores = proxy_scores(example, e, u, native=False)
    log_mass, owner, _ = page_state(example, scores)
    for block in ("encoder", "query"):
        original = e if block == "encoder" else u
        direction = torch.randn_like(original)
        analytic = page_gradient(example, scores, owner, 2, e, u, block)
        eps = 1e-6
        values = []
        for sign in (-1, 1):
            changed = original + sign * eps * direction
            left, right = (changed, u) if block == "encoder" else (e, changed)
            mass, _, _ = page_state(example, proxy_scores(example, left, right, native=False))
            values.append(mass[2])
        torch.testing.assert_close((values[1] - values[0]) / (2 * eps),
                                   (analytic * direction).sum(), rtol=1e-5, atol=1e-7)


def test_pairs_exclude_pinned_and_require_positive_mass_gain():
    example, _, _ = fixture()
    example.teacher_mass[:] = torch.tensor([.1, .2, .6, .1])
    selected = torch.tensor([True, True, False, False])
    pos, neg, weight = boundary_pairs(example, selected, maximum_pairs=8)
    assert pos.tolist() == [2] and neg.tolist() == [1]
    torch.testing.assert_close(weight, torch.tensor([.4], dtype=weight.dtype))


def test_repair_keeps_inputs_and_actual_fit_mass_non_decreasing():
    example, e, u = fixture(torch.float32)
    old_e, old_u, old_base = e.clone(), u.clone(), example.base.clone()
    fitted_e, fitted_u, history = repair_page_boundary([example], e, u, sweeps=2)
    assert torch.equal(e, old_e) and torch.equal(u, old_u)
    assert torch.equal(example.base, old_base)
    assert fitted_e.shape == e.shape and fitted_u.shape == u.shape
    assert all(row["after"]["mass"] >= row["before"]["mass"] for row in history)
    assert mass_coverage([example], fitted_e, fitted_u)["mass"] >= mass_coverage([example], e, u)["mass"]


def test_full_budget_has_no_wrongly_omitted_pages():
    example, e, u = fixture(torch.float32)
    full = PageRoutingExample(example.query, example.base, example.residual,
                              example.teacher_mass, page_size=3, page_budget=4)
    new_e, new_u, history = repair_page_boundary([full], e, u, sweeps=1)
    assert torch.equal(new_e, e) and torch.equal(new_u, u)
    assert all(row["constraints"] == 0 for row in history)


def test_native_proxy_is_one_bf16_concatenated_dot_product():
    example, e, u = fixture(torch.float32)
    q = example.query.bfloat16()
    code = example.residual.bfloat16() @ e.bfloat16()
    projected = torch.cat((q, torch.einsum("hd,hdr->hr", q, u.bfloat16())), -1)
    expected = ((projected @ torch.cat((example.base.bfloat16(), code), -1).T) * .5).float()
    torch.testing.assert_close(proxy_scores(example, e, u, native=True), expected, rtol=0, atol=0)
