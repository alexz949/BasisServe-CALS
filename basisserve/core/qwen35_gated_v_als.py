"""Group-shared V reconstruction with token-dependent output gates.

Captures are pre-gate aggregated values, never historical attention matrices.
Both ALS blocks retain the full output residual and cross-group coupling.
"""

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from basisserve.core.gqa_routed_ov_joint import conjugate_gradient_matrix


@dataclass(frozen=True)
class GatedVCapture:
    z: Tensor  # [rows, query_heads, value_dim]
    gate: Tensor  # native sigmoid, same shape as z
    weight: Tensor  # [query_heads, value_dim, output_dim]
    target: Tensor  # bias-free [rows, output_dim]
    head_to_group: Tensor

    def validate(self, groups: int) -> None:
        assert self.z.ndim == 3 and self.gate.shape == self.z.shape
        assert self.weight.shape[:2] == self.z.shape[1:]
        assert self.target.shape == (self.z.shape[0], self.weight.shape[-1])
        assert self.z.shape[0] > 0
        assert self.head_to_group.shape == (self.z.shape[1],)
        assert self.head_to_group.dtype == torch.long
        assert set(self.head_to_group.tolist()) == set(range(groups))
        assert all(torch.isfinite(t).all() for t in (self.z, self.gate, self.weight, self.target))
        assert bool(((self.gate >= 0) & (self.gate <= 1)).all())


class GatedVBlock:
    """Row-streamed apply/adjoint for one joint encoder or decoder block."""

    def __init__(self, capture: GatedVCapture, fixed: Tensor, *, block: str, chunk_rows: int = 256):
        assert block in {"encoder", "decoder"} and chunk_rows > 0
        capture.validate(fixed.shape[0])
        assert fixed.ndim == 3
        self.capture, self.fixed, self.block = capture, fixed, block
        self.chunk_rows = chunk_rows
        self.groups = fixed.shape[0]
        self.shape = (self.groups, fixed.shape[2], fixed.shape[1])
        self.weight = capture.weight.to(fixed)
        flat_weight = self.weight.flatten(0, 1)
        self.output_gram = flat_weight @ flat_weight.T

    def chunks(self):
        c = self.capture
        for start in range(0, c.z.shape[0], self.chunk_rows):
            sl = slice(start, start + self.chunk_rows)
            yield sl, c.z[sl].to(self.fixed), c.gate[sl].to(self.fixed)

    def _preoutput(self, x: Tensor, z: Tensor, gate: Tensor) -> Tensor:
        mapping = self.capture.head_to_group.to(x.device)
        encoder, decoder = (x, self.fixed) if self.block == "encoder" else (self.fixed, x)
        latent = torch.einsum("nhd,hdr->nhr", z, encoder[mapping])
        restored = torch.einsum("nhr,hrd->nhd", latent, decoder[mapping])
        return restored * gate

    def _apply(self, x: Tensor, z: Tensor, gate: Tensor) -> Tensor:
        return torch.einsum("nhd,hdo->no", self._preoutput(x, z, gate), self.weight)

    def _adjoint(self, h: Tensor, z: Tensor, gate: Tensor) -> Tensor:
        back = torch.einsum("no,hdo->nhd", h, self.weight) * gate
        return self._factor_adjoint(back, z)

    def _factor_adjoint(self, back: Tensor, z: Tensor) -> Tensor:
        mapping = self.capture.head_to_group.to(self.fixed.device)
        if self.block == "decoder":
            latent = torch.einsum("nhd,hdr->nhr", z, self.fixed[mapping])
            per_head = torch.einsum("nhr,nhd->hrd", latent, back)
        else:
            back = torch.einsum("nhd,hrd->nhr", back, self.fixed[mapping])
            per_head = torch.einsum("nhd,nhr->hdr", z, back)
        result = self.fixed.new_zeros(self.shape)
        return result.index_add_(0, mapping, per_head)

    def apply(self, x: Tensor) -> Tensor:
        return torch.cat([self._apply(x, z, g) for _, z, g in self.chunks()])

    def adjoint(self, h: Tensor) -> Tensor:
        result = self.fixed.new_zeros(self.shape)
        for sl, z, g in self.chunks():
            result += self._adjoint(h[sl].to(self.fixed), z, g)
        return result

    def normal(self, x: Tensor) -> Tensor:
        result = torch.zeros_like(x)
        for _, z, g in self.chunks():
            # W W^T is fixed and retains all cross-head blocks. This replaces
            # two output-width GEMMs by one without averaging token gates.
            post = self._preoutput(x, z, g)
            back = (post.flatten(1) @ self.output_gram).reshape_as(post) * g
            result += self._factor_adjoint(back, z)
        return result / self.capture.z.shape[0]

    def loss(self, x: Tensor) -> float:
        total = 0.0
        for sl, z, g in self.chunks():
            residual = self._apply(x, z, g) - self.capture.target[sl].to(x)
            total += float(residual.double().square().sum())
        return total / (2 * self.capture.z.shape[0])

    def preconditioner_diagonal(self) -> Tensor:
        """Positive Jacobi approximation, omitting cross-head/channel terms.

        Only the preconditioner uses this approximation; normal() remains the
        exact coupled operator, including every cross-group contribution.
        """
        mapping = self.capture.head_to_group.to(self.fixed.device)
        row_energy = self.capture.weight.to(self.fixed).square().sum(-1)
        result = self.fixed.new_zeros(self.shape)
        for _, z, gate in self.chunks():
            weighted = gate.square() * row_energy[None]
            if self.block == 'decoder':
                latent = torch.einsum('nhd,hdr->nhr', z, self.fixed[mapping])
                diagonal = torch.einsum('nhr,nhd->hrd', latent.square(), weighted)
            else:
                latent_energy = torch.einsum('nhd,hrd->nhr', weighted, self.fixed[mapping].square())
                diagonal = torch.einsum('nhd,nhr->hdr', z.square(), latent_energy)
            result.index_add_(0, mapping, diagonal)
        return result / self.capture.z.shape[0]

    def encoder_separable_preconditioner(self, damping: float):
        """Pooled input covariance and gate-aware decoder Gram per KV group.

        This approximates ONLY the preconditioner as a Kronecker product.
        Its damped inverse is evaluated in the two small eigenbases.
        """
        assert self.block == 'encoder'
        mapping = self.capture.head_to_group.to(self.fixed.device)
        heads, width = self.capture.z.shape[1:]
        z_gram = self.fixed.new_zeros(heads, width, width)
        gate_gram = torch.zeros_like(z_gram)
        for _, z, gate in self.chunks():
            z_gram += torch.einsum('nhd,nhe->hde', z, z)
            gate_gram += torch.einsum('nhd,nhe->hde', gate, gate)
        output_gram = self.weight @ self.weight.transpose(1, 2)
        r = self.fixed[mapping]
        right_heads = r @ (gate_gram * output_gram) @ r.transpose(1, 2)
        left = self.fixed.new_zeros(self.groups, width, width).index_add_(0, mapping, z_gram)
        right = self.fixed.new_zeros(self.groups, self.fixed.shape[1], self.fixed.shape[1]).index_add_(0, mapping, right_heads)
        counts = torch.bincount(mapping, minlength=self.groups).to(self.fixed)
        divisor = self.capture.z.shape[0] * counts[:, None, None]
        left, right = left / divisor, right / divisor
        lv, lq = torch.linalg.eigh((left + left.mT) * 0.5)
        rv, rq = torch.linalg.eigh((right + right.mT) * 0.5)
        denominator = counts[:, None, None] * lv.clamp_min(0)[:, :, None] * rv.clamp_min(0)[:, None, :] + damping
        denominator = denominator.clamp_min(torch.finfo(self.fixed.dtype).eps * denominator.amax(dim=(1, 2), keepdim=True))
        denominator = denominator.clamp_min(torch.finfo(self.fixed.dtype).tiny)
        def inverse(value):
            transformed = lq.mT @ value @ rq
            return lq @ (transformed / denominator) @ rq.mT
        return inverse


@torch.no_grad()
def initialize_gated_v(capture: GatedVCapture, rank: int, *, device="cpu", dtype=torch.float32, chunk_rows=256):
    groups = int(capture.head_to_group.max()) + 1
    capture.validate(groups)
    width = capture.z.shape[-1]
    assert 0 < rank <= width
    gram = torch.zeros(groups, width, width, device=device, dtype=torch.float64)
    mapping = capture.head_to_group.to(device)
    for start in range(0, capture.z.shape[0], chunk_rows):
        z = capture.z[start:start + chunk_rows].to(device=device, dtype=torch.float64)
        gram.index_add_(0, mapping, torch.einsum("nhd,nhe->hde", z, z))
    _, vectors = torch.linalg.eigh(gram)
    encoder = vectors[:, :, -rank:].flip(-1)
    pivot = encoder.abs().argmax(dim=1, keepdim=True)
    signs = encoder.gather(1, pivot).sign()
    encoder = (encoder * signs).to(dtype=dtype)
    return encoder, encoder.transpose(1, 2).contiguous()


@torch.no_grad()
def gated_v_update(operator: GatedVBlock, old: Tensor, *, relative_damping=1e-5, linear_tol=1e-5, linear_max_iter=200, progress=None, encoder_preconditioner='jacobi'):
    """Proximal update; deterministic Rayleigh scale, explicitly recorded.

    Damping is relative to N evaluated on the current factor, not a ridge
    penalty that shrinks the predictor toward zero.
    """
    normal_old = operator.normal(old)
    scale = float((old * normal_old).sum() / old.square().sum().clamp_min(torch.finfo(old.dtype).tiny))
    damping = relative_damping * max(scale, torch.finfo(old.dtype).eps)
    rhs = operator.adjoint(operator.capture.target) / operator.capture.z.shape[0] - normal_old
    # Normalize so the generic CG curvature guard does not interpret small
    # physical units as breakdown. This does not alter the linear system.
    normalizer = max(scale, torch.finfo(old.dtype).eps)
    assert encoder_preconditioner in {'jacobi', 'separable'}
    preconditioner_name = 'jacobi_omitting_cross_head_channel_terms'
    if operator.block == 'encoder' and encoder_preconditioner == 'separable':
        inverse = operator.encoder_separable_preconditioner(damping)
        precondition = lambda x: inverse(x) * normalizer
        preconditioner_name = 'pooled_input_covariance_gate_aware_decoder_kronecker'
    else:
        diagonal = (operator.preconditioner_diagonal() + damping).clamp_min(torch.finfo(old.dtype).tiny)
        precondition = lambda x: x * normalizer / diagonal
    calls = 0
    def normalized(x):
        nonlocal calls
        calls += 1
        if progress is not None and calls % 10 == 0:
            progress({'block': operator.block, 'normal_applications': calls})
        return operator.normal(x) / normalizer
    delta, diagnostics = conjugate_gradient_matrix(normalized,
        rhs / normalizer, relative_tolerance=linear_tol, max_iterations=linear_max_iter,
        absolute_damping=damping / normalizer, preconditioner=precondition)
    rhs_norm = rhs.norm().clamp_min(torch.finfo(old.dtype).tiny)
    true_residual = operator.normal(delta) + damping * delta - rhs
    used = diagnostics.iterations
    replacements = []
    # Recursive CG residuals can drift in FP32. Replace them with the measured
    # residual and spend only the remaining iterations of the original cap.
    while float(true_residual.norm() / rhs_norm) > linear_tol and used < linear_max_iter:
        if diagnostics.negative_curvature or diagnostics.iterations == 0:
            break
        before_residual = float(true_residual.norm() / rhs_norm)
        correction_tolerance = min(0.5, linear_tol / before_residual)
        correction, corrected = conjugate_gradient_matrix(normalized, -true_residual / normalizer,
            relative_tolerance=correction_tolerance, max_iterations=linear_max_iter - used,
            absolute_damping=damping / normalizer, preconditioner=precondition)
        delta = delta + correction
        used += corrected.iterations
        true_residual = operator.normal(delta) + damping * delta - rhs
        measured = float(true_residual.norm() / rhs_norm)
        replacements.append({'iterations': corrected.iterations, 'before': before_residual, 'after': measured})
        if corrected.iterations == 0 or measured >= before_residual:
            break
    candidate = old + delta
    before, after = operator.loss(old), operator.loss(candidate)
    accepted = bool(torch.isfinite(candidate).all()) and after <= before
    record = {**asdict(diagnostics), "absolute_damping": damping,
        "iterations": used, "converged": float(true_residual.norm() / rhs_norm) <= linear_tol,
        "relative_residual": float(true_residual.norm() / rhs_norm), "residual_replacements": replacements,
        "preconditioner": preconditioner_name, "operator_normalizer": normalizer,
        "block": operator.block, "loss_before": before,
        "candidate_loss": after, "loss_after": after if accepted else before, "accepted": accepted,
        "damping_scale_rule": "current_factor_rayleigh_quotient_floor_dtype_epsilon",
        "damping_scale": scale, "recomputed_relative_residual": float(true_residual.norm() / rhs.norm().clamp_min(torch.finfo(old.dtype).tiny)),
        "hit_iteration_cap": used == linear_max_iter}
    return (candidate if accepted else old), record


@torch.no_grad()
def fit_gated_v(capture: GatedVCapture, rank: int, *, heldout: GatedVCapture | None = None,
                encoder_sweeps=6, device="cpu", work_dtype=torch.float32, chunk_rows=256,
                relative_damping=1e-5, linear_tol=1e-5, linear_max_iter=200, progress=None, encoder_preconditioner='jacobi'):
    encoder, decoder = initialize_gated_v(capture, rank, device=device, dtype=work_dtype, chunk_rows=chunk_rows)
    history = []
    endpoints = []
    best = None
    for sweep in range(encoder_sweeps + 1):
        op = GatedVBlock(capture, encoder, block="decoder", chunk_rows=chunk_rows)
        decoder, record = gated_v_update(op, decoder, relative_damping=relative_damping,
            linear_tol=linear_tol, linear_max_iter=linear_max_iter, progress=progress, encoder_preconditioner=encoder_preconditioner)
        history.append({"sweep": sweep, **record})
        if progress is not None:
            progress(history[-1])
        validation = GatedVBlock(heldout or capture, encoder, block="decoder", chunk_rows=chunk_rows).loss(decoder)
        endpoints.append({"sweep": sweep, "heldout_loss" if heldout else "fit_loss": validation})
        if best is None or validation < best[0]:
            best = (validation, encoder.clone(), decoder.clone(), sweep)
        if sweep == encoder_sweeps:
            break
        op = GatedVBlock(capture, decoder, block="encoder", chunk_rows=chunk_rows)
        encoder, record = gated_v_update(op, encoder, relative_damping=relative_damping,
            linear_tol=linear_tol, linear_max_iter=linear_max_iter, progress=progress, encoder_preconditioner=encoder_preconditioner)
        history.append({"sweep": sweep, **record})
        if progress is not None:
            progress(history[-1])
        q, t = torch.linalg.qr(encoder, mode="reduced")
        history.append({"sweep": sweep, "block": "gauge", "numerical_ranks": torch.linalg.matrix_rank(t).tolist()})
        encoder, decoder = q, t @ decoder
    return {"E_V": best[1], "R_V": best[2], "selected_sweep": best[3],
            "selection": "heldout_decoder_closed" if heldout else "fit_decoder_closed",
            "history": history, "endpoints": endpoints}
