"""Local alternating least-squares repair of physical GQA page misrankings.

The differentiable local model uses FP32. Acceptance measures the actual
BF16 concatenated-sidecar selector. No autograd or optimizer is used.
"""

from dataclasses import dataclass
import torch

from basisserve.core.c1_conditional_page_attention import _selected_pages


@dataclass(frozen=True)
class PageRoutingExample:
    query: torch.Tensor  # [GQA query heads, key dimension], post-RoPE
    base: torch.Tensor  # [causal tokens, key dimension], frozen post-RoPE
    residual: torch.Tensor  # same geometry; BF16 exact K minus BF16 Base
    teacher_mass: torch.Tensor  # [heads, pages], full-attention probabilities
    page_size: int = 32
    page_budget: int = 64
    pinned: int = 1


def proxy_scores(example, encoder, query_factor, *, native):
    dtype = torch.bfloat16 if native else encoder.dtype
    q = example.query.to(dtype)
    residual = example.residual.to(dtype)
    code = residual @ encoder.to(dtype)
    identity = torch.eye(q.shape[-1], device=q.device, dtype=dtype)
    projector = torch.cat((identity.expand(q.shape[0], -1, -1), query_factor.to(dtype)), dim=-1)
    projected = torch.einsum("hd,hdr->hr", q, projector)
    sidecar = torch.cat((example.base.to(dtype), code), dim=-1)
    # Native attention scales the BF16 matmul result before converting to FP32.
    return ((projected @ sidecar.T) * (q.shape[-1] ** -.5)).float() if native else (
        projected @ sidecar.T) * (q.shape[-1] ** -.5)


def page_state(example, scores):
    heads, tokens = scores.shape
    assert tokens % example.page_size == 0
    pages = tokens // example.page_size
    assert 0 <= example.pinned < example.page_budget <= pages
    logits = scores.reshape(heads, pages, example.page_size).logsumexp(-1)
    logits = logits.clone()
    logits[:, :example.pinned] = -torch.inf
    head_log_mass = logits - logits.logsumexp(-1, keepdim=True)
    group_log_mass, owner = head_log_mass.max(0)
    ids, valid = _selected_pages(
        scores[None, :, None, :], torch.ones_like(scores[None, :, None, :], dtype=torch.bool),
        kv_heads=1, page_size=example.page_size, page_budget=example.page_budget,
        pinned_prefix_pages=example.pinned)
    assert valid.all()
    selected = torch.zeros(pages, device=scores.device, dtype=torch.bool)
    selected[ids.flatten()] = True
    return group_log_mass, owner, selected


def boundary_pairs(example, selected, *, maximum_pairs):
    """Best omitted versus worst selected, excluding pinned pages.

    Teacher weights are head-mean full-attention page masses. Thus a physical
    swap is scored by its true gain in mean attention-mass coverage.
    """
    mass = example.teacher_mass.mean(0)
    candidates = torch.arange(mass.numel(), device=mass.device) >= example.pinned
    omitted = torch.where(candidates & ~selected)[0]
    included = torch.where(candidates & selected)[0]
    n = min(maximum_pairs, omitted.numel(), included.numel())
    positive = omitted[mass[omitted].argsort(descending=True)[:n]]
    negative = included[mass[included].argsort()[:n]]
    weight = mass[positive] - mass[negative]
    useful = weight > 0
    return positive[useful], negative[useful], weight[useful]


def page_gradient(example, scores, owner, page, encoder, query_factor, block):
    """Derivative of log(max-head normalized non-sink page mass).

    Away from owner ties, differentiate both page LSE and its head-specific
    non-sink normalization. Dropping the latter would optimize a different
    selector when the two pages have different active heads.
    """
    h = int(owner[page])
    first = example.pinned * example.page_size
    start = int(page) * example.page_size
    rows = example.residual.to(scores.dtype)
    mean = scores[h, first:].softmax(0) @ rows[first:]
    local = scores[h, start:start + example.page_size].softmax(0) @ rows[start:start + example.page_size]
    difference = local - mean
    q = example.query[h].to(scores.dtype)
    scale = q.numel() ** -.5
    if block == "encoder":
        return scale * torch.outer(difference, q @ query_factor[h])
    assert block == "query"
    result = torch.zeros_like(query_factor)
    result[h] = scale * torch.outer(q, difference @ encoder)
    return result


def mass_coverage(examples, encoder, query_factor):
    values, non_sink = [], []
    for example in examples:
        _, _, selected = page_state(example, proxy_scores(example, encoder, query_factor, native=True))
        mass = example.teacher_mass
        values.append(mass[:, selected].sum(-1).mean())
        routed = selected.clone()
        routed[:example.pinned] = False
        denom = mass[:, example.pinned:].sum(-1)
        assert (denom > 0).all()
        non_sink.append((mass[:, routed].sum(-1) / denom).mean())
    return {"mass": float(torch.stack(values).mean()),
            "non_sink_mass": float(torch.stack(non_sink).mean())}


def fixed_pair_loss(examples, pairs, encoder, query_factor, margin):
    numerator = denominator = 0.0
    for example, (positive, negative, weight) in zip(examples, pairs):
        score, _, _ = page_state(example, proxy_scores(example, encoder, query_factor, native=True))
        loss = (margin - score[positive] + score[negative]).clamp_min(0).square()
        numerator += float((weight * loss).sum())
        denominator += float(weight.sum())
    return numerator / max(denominator, 1e-30)


@torch.inference_mode()
def repair_page_boundary(examples, encoder, query_factor, *, sweeps=2,
                         maximum_pairs=8, margin=.05, relative_damping=.01):
    """Alternating local hinge least squares, with fit-only acceptance gates.

    Solve in observation space, so the small smoke needs no dense parameter
    Hessian. This is not an exact optimizer of the discontinuous Top-B objective.
    Diagnostic examples are deliberately absent from the fitting interface.
    """
    assert examples and sweeps > 0 and maximum_pairs > 0
    assert margin > 0 and relative_damping > 0
    encoder, query_factor = encoder.float().clone(), query_factor.float().clone()
    history = []
    for sweep in range(sweeps):
        for block in ("encoder", "query"):
            before = mass_coverage(examples, encoder, query_factor)
            pairs, jacobians, targets, weights = [], [], [], []
            for example in examples:
                _, _, selected = page_state(example, proxy_scores(example, encoder, query_factor, native=True))
                pair = boundary_pairs(example, selected, maximum_pairs=maximum_pairs)
                pairs.append(pair)
                scores = proxy_scores(example, encoder, query_factor, native=False)
                log_mass, owner, _ = page_state(example, scores)
                for pos, neg, weight in zip(*pair):
                    target = margin - log_mass[pos] + log_mass[neg]
                    if float(target) <= 0:
                        continue
                    derivative = page_gradient(example, scores, owner, pos, encoder, query_factor, block)
                    derivative -= page_gradient(example, scores, owner, neg, encoder, query_factor, block)
                    jacobians.append(derivative.flatten())
                    targets.append(target)
                    weights.append(weight)
            record = {"sweep": sweep, "block": block, "constraints": len(targets),
                      "before": before, "accepted_step": 0.0}
            if targets:
                weight = torch.stack(weights)
                root = (weight / weight.sum()).sqrt()
                matrix = torch.stack(jacobians) * root[:, None]
                target = torch.stack(targets) * root
                gram = matrix @ matrix.T
                damping = relative_damping * gram.diag().mean().clamp_min(1e-12)
                dual = torch.linalg.solve(gram + damping * torch.eye(len(targets), device=gram.device), target)
                original = encoder if block == "encoder" else query_factor
                delta = (matrix.T @ dual).reshape_as(original)
                assert torch.isfinite(delta).all()
                loss_before = fixed_pair_loss(examples, pairs, encoder, query_factor, margin)
                for step in (1.0, .5, .25, .125):
                    candidate = original + step * delta
                    e, u = (candidate, query_factor) if block == "encoder" else (encoder, candidate)
                    after = mass_coverage(examples, e, u)
                    loss_after = fixed_pair_loss(examples, pairs, e, u, margin)
                    if after["mass"] >= before["mass"] and loss_after < loss_before:
                        encoder, query_factor = e, u
                        record.update(accepted_step=step, pair_loss_before=loss_before, pair_loss_after=loss_after)
                        break
            record["after"] = mass_coverage(examples, encoder, query_factor)
            history.append(record)
    return encoder, query_factor, history
