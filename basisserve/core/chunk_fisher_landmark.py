"""Direct Chunk8 Fisher landmarks for GQA Key routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from basisserve.core.gqa_routed_ov_joint import (
    CGDiagnostics,
    conjugate_gradient_matrix,
)


@dataclass(frozen=True)
class ChunkFisherExample:
    """One causal query position and its routable historical chunks."""

    position: int
    queries: torch.Tensor
    features_by_group: torch.Tensor
    base_logits: torch.Tensor
    teacher_logits: torch.Tensor

    def validate(self, head_to_group: torch.Tensor) -> None:
        heads, head_dim = map(int, self.queries.shape)
        groups, chunks, _ = map(int, self.features_by_group.shape)
        assert self.base_logits.shape == self.teacher_logits.shape == (heads, chunks)
        assert head_to_group.shape == (heads,)
        assert int(head_to_group.min()) == 0
        assert int(head_to_group.max()) + 1 == groups
        assert head_dim > 0 and chunks > 0
        assert all(
            torch.isfinite(value).all()
            for value in (
                self.queries,
                self.features_by_group,
                self.base_logits,
                self.teacher_logits,
            )
        )


@dataclass(frozen=True)
class ChunkFisherDataset:
    examples: tuple[ChunkFisherExample, ...]
    head_to_group: torch.Tensor
    scaling: float

    def validate(self) -> None:
        assert self.examples and self.scaling > 0
        for example in self.examples:
            example.validate(self.head_to_group)
        heads, head_dim = map(int, self.examples[0].queries.shape)
        groups = int(self.examples[0].features_by_group.shape[0])
        feature_dim = int(self.examples[0].features_by_group.shape[-1])
        assert all(
            example.queries.shape == (heads, head_dim)
            and int(example.features_by_group.shape[0]) == groups
            and int(example.features_by_group.shape[-1]) == feature_dim
            for example in self.examples
        )

    @property
    def heads(self) -> int:
        return int(self.examples[0].queries.shape[0])

    @property
    def groups(self) -> int:
        return int(self.examples[0].features_by_group.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self.examples[0].queries.shape[-1])

    @property
    def feature_dim(self) -> int:
        return int(self.examples[0].features_by_group.shape[-1])


@dataclass(frozen=True)
class ChunkFisherHalfStep:
    sweep: int
    boundary: str
    train: dict[str, float]
    heldout: dict[str, float]
    solver_count: int
    converged_count: int
    hit_iteration_limit_count: int
    total_iterations: int
    mean_iterations: float
    maximum_iterations: int
    maximum_relative_residual: float
    solve_wall_seconds: float
    diagnostic_wall_seconds: float
    total_wall_seconds: float


@dataclass(frozen=True)
class ChunkFisherFit:
    encoders: torch.Tensor
    query_factors: torch.Tensor
    half_steps: tuple[ChunkFisherHalfStep, ...]
    final_query_diagnostics: tuple[CGDiagnostics, ...]
    encoder_diagnostics: tuple[tuple[CGDiagnostics, ...], ...]
    preconditioner_wall_seconds: float = 0.0


@dataclass(frozen=True)
class PackedChunkFisherWindow:
    """Many query positions sharing one window's chunk features."""

    positions: torch.Tensor
    queries: torch.Tensor
    features_by_group: torch.Tensor
    base_logits: torch.Tensor
    teacher_logits: torch.Tensor
    candidate_counts: torch.Tensor

    def validate(self, head_to_group: torch.Tensor) -> None:
        query_count, heads, head_dim = map(int, self.queries.shape)
        groups, chunks, _ = map(int, self.features_by_group.shape)
        assert self.positions.shape == self.candidate_counts.shape == (query_count,)
        assert self.base_logits.shape == self.teacher_logits.shape == (
            query_count,
            heads,
            chunks,
        )
        assert head_to_group.shape == (heads,)
        assert int(head_to_group.min()) == 0
        assert int(head_to_group.max()) + 1 == groups
        assert head_dim > 0 and chunks > 0
        assert int(self.candidate_counts.min()) > 0
        assert int(self.candidate_counts.max()) <= chunks
        assert all(
            torch.isfinite(value).all()
            for value in (
                self.queries,
                self.features_by_group,
                self.base_logits,
                self.teacher_logits,
            )
        )

    def valid_mask(self) -> torch.Tensor:
        chunks = int(self.features_by_group.shape[1])
        return torch.arange(
            chunks,
            device=self.candidate_counts.device,
        ).unsqueeze(0) < self.candidate_counts.unsqueeze(1)


@dataclass(frozen=True)
class PackedChunkFisherDataset:
    windows: tuple[PackedChunkFisherWindow, ...]
    head_to_group: torch.Tensor
    scaling: float

    def validate(self) -> None:
        assert self.windows and self.scaling > 0
        for window in self.windows:
            window.validate(self.head_to_group)
        heads = int(self.windows[0].queries.shape[1])
        head_dim = int(self.windows[0].queries.shape[2])
        groups = int(self.windows[0].features_by_group.shape[0])
        feature_dim = int(self.windows[0].features_by_group.shape[-1])
        assert all(
            int(window.queries.shape[1]) == heads
            and int(window.queries.shape[2]) == head_dim
            and int(window.features_by_group.shape[0]) == groups
            and int(window.features_by_group.shape[-1]) == feature_dim
            for window in self.windows
        )

    @property
    def heads(self) -> int:
        return int(self.windows[0].queries.shape[1])

    @property
    def groups(self) -> int:
        return int(self.windows[0].features_by_group.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self.windows[0].queries.shape[-1])

    @property
    def feature_dim(self) -> int:
        return int(self.windows[0].features_by_group.shape[-1])


@dataclass(frozen=True)
class CompactChunkFisherDataset:
    """Exact fixed-teacher Fisher sufficient statistics for Chunk landmarks."""

    queries_by_head: torch.Tensor
    fisher_grams_by_head: torch.Tensor
    target_cross_by_head: torch.Tensor
    head_to_group: torch.Tensor
    scaling: float
    target_fisher_energy: float

    def validate(self) -> None:
        heads, observations, head_dim = map(int, self.queries_by_head.shape)
        feature_dim = int(self.fisher_grams_by_head.shape[-1])
        assert self.fisher_grams_by_head.shape == (
            heads,
            observations,
            feature_dim,
            feature_dim,
        )
        assert self.target_cross_by_head.shape == (
            heads,
            observations,
            feature_dim,
        )
        assert self.head_to_group.shape == (heads,)
        assert int(self.head_to_group.min()) == 0
        assert int(self.head_to_group.max()) + 1 > 0
        assert self.scaling > 0 and self.target_fisher_energy > 0
        assert head_dim > 0 and observations > 0 and feature_dim > 0
        assert all(
            torch.isfinite(value).all()
            for value in (
                self.queries_by_head,
                self.fisher_grams_by_head,
                self.target_cross_by_head,
            )
        )

    @property
    def heads(self) -> int:
        return int(self.queries_by_head.shape[0])

    @property
    def groups(self) -> int:
        return int(self.head_to_group.max()) + 1

    @property
    def head_dim(self) -> int:
        return int(self.queries_by_head.shape[-1])

    @property
    def feature_dim(self) -> int:
        return int(self.fisher_grams_by_head.shape[-1])


def fisher_multiply(probabilities: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Apply ``(Diag(p) - p p.T)`` without materializing the matrix."""

    assert probabilities.shape == values.shape
    return probabilities * (
        values - torch.sum(probabilities * values, dim=-1, keepdim=True)
    )


def chunk_teacher_logits(
    grouped_queries: torch.Tensor,
    exact_keys: torch.Tensor,
    *,
    chunk_size: int,
    scaling: float,
) -> torch.Tensor:
    """Direct token-QK log-sum-exp teacher for complete contiguous chunks."""

    groups, query_heads_per_group, head_dim = map(int, grouped_queries.shape)
    assert exact_keys.ndim == 3 and exact_keys.shape[0] == groups
    assert int(exact_keys.shape[-1]) == head_dim
    assert int(exact_keys.shape[1]) % int(chunk_size) == 0
    token_logits = float(scaling) * torch.einsum(
        "ghd,gtd->ght",
        grouped_queries,
        exact_keys,
    )
    return torch.logsumexp(
        token_logits.reshape(
            groups,
            query_heads_per_group,
            int(exact_keys.shape[1]) // int(chunk_size),
            int(chunk_size),
        ),
        dim=-1,
    )


def chunk_mean_logits(
    grouped_queries: torch.Tensor,
    post_rope_keys: torch.Tensor,
    *,
    chunk_size: int,
    scaling: float,
) -> torch.Tensor:
    """Score post-RoPE Key means with the exact equal-chunk count correction."""

    groups, tokens, head_dim = map(int, post_rope_keys.shape)
    assert grouped_queries.shape[0] == groups
    assert int(grouped_queries.shape[-1]) == head_dim
    assert tokens % int(chunk_size) == 0
    means = post_rope_keys.reshape(
        groups,
        tokens // int(chunk_size),
        int(chunk_size),
        head_dim,
    ).mean(dim=-2)
    return (
        float(scaling)
        * torch.einsum("ghd,gcd->ghc", grouped_queries, means)
        + math.log(int(chunk_size))
    )


def residual_chunk_features(
    residual: torch.Tensor,
    *,
    chunk_size: int,
    feature: str,
) -> torch.Tensor:
    """Build mean or flat features from complete post-RoPE residual chunks."""

    groups, tokens, head_dim = map(int, residual.shape)
    assert tokens % int(chunk_size) == 0
    chunks = residual.reshape(
        groups,
        tokens // int(chunk_size),
        int(chunk_size),
        head_dim,
    )
    assert feature in ("mean", "flat")
    if feature == "mean":
        return chunks.mean(dim=-2)
    return chunks.flatten(-2)


def adapt_token_residual_initialization(
    token_encoder: torch.Tensor,
    token_query: torch.Tensor,
    *,
    chunk_size: int,
    feature: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt token-R factors only as a deterministic Chunk8 initialization."""

    assert token_encoder.ndim == token_query.ndim == 3
    assert int(token_encoder.shape[-1]) == int(token_query.shape[-1])
    assert feature in ("mean", "flat")
    if feature == "mean":
        encoder = token_encoder.clone()
    else:
        encoder = (
            token_encoder.unsqueeze(1)
            .expand(-1, int(chunk_size), -1, -1)
            .reshape(
                int(token_encoder.shape[0]),
                int(chunk_size) * int(token_encoder.shape[1]),
                int(token_encoder.shape[2]),
            )
            / float(chunk_size)
        )
    return encoder.contiguous(), token_query.clone().contiguous()


def factor_logits(
    dataset: ChunkFisherDataset,
    example: ChunkFisherExample,
    encoders: torch.Tensor,
    query_factors: torch.Tensor,
) -> torch.Tensor:
    mapping = dataset.head_to_group.to(device=encoders.device, dtype=torch.long)
    features = example.features_by_group.to(encoders)
    queries = example.queries.to(query_factors)
    chunk_codes = torch.einsum("gci,gir->gcr", features, encoders)
    head_codes = torch.einsum("hd,hdr->hr", queries, query_factors)
    return float(dataset.scaling) * torch.einsum(
        "hr,hcr->hc",
        head_codes,
        chunk_codes.index_select(0, mapping),
    )


def chunk_fisher_metrics(
    dataset: ChunkFisherDataset,
    encoders: torch.Tensor,
    query_factors: torch.Tensor,
    *,
    routed_chunks: int = 244,
) -> dict[str, float]:
    dataset.validate()
    fisher_loss = 0.0
    base_fisher_energy = 0.0
    score_error = 0.0
    score_energy = 0.0
    recalls = []
    masses = []
    for example in dataset.examples:
        teacher = example.teacher_logits.to(encoders)
        base = example.base_logits.to(encoders)
        residual_target = teacher - base
        correction = factor_logits(dataset, example, encoders, query_factors)
        error = correction - residual_target
        probabilities = torch.softmax(teacher, dim=-1)
        fisher_loss += float(
            0.5 * torch.sum(error * fisher_multiply(probabilities, error))
        )
        base_fisher_energy += float(
            0.5
            * torch.sum(
                residual_target
                * fisher_multiply(probabilities, residual_target)
            )
        )
        student = base + correction
        score_error += float((student - teacher).double().square().sum())
        score_energy += float(teacher.double().square().sum())
        count = min(int(routed_chunks), int(teacher.shape[-1]))
        selected = student.topk(count, dim=-1).indices
        exact = teacher.topk(count, dim=-1).indices
        selected_mask = torch.zeros_like(teacher, dtype=torch.bool).scatter_(
            -1, selected, True
        )
        exact_mask = torch.zeros_like(teacher, dtype=torch.bool).scatter_(
            -1, exact, True
        )
        recalls.extend(
            ((selected_mask & exact_mask).sum(-1).float() / count).tolist()
        )
        masses.extend(
            probabilities.gather(-1, selected).sum(-1).tolist()
        )
    return {
        "fisher_loss": fisher_loss,
        "fisher_nmse": fisher_loss / max(
            base_fisher_energy,
            torch.finfo(torch.float64).tiny,
        ),
        "base_fisher_energy": base_fisher_energy,
        "chunk_logit_rel_mse": score_error / score_energy,
        "exact_chunk_support_recall": sum(recalls) / len(recalls),
        "routed_candidate_attention_mass": sum(masses) / len(masses),
    }


def _relative_damping(diagonal: torch.Tensor, relative: float) -> float:
    return float(relative) * max(
        float(diagonal.abs().mean()),
        torch.finfo(diagonal.dtype).tiny,
    )


def _preconditioner(diagonal: torch.Tensor, damping: float):
    shifted = diagonal.clamp_min(0).add(float(damping))
    shifted.clamp_min_(torch.finfo(shifted.dtype).tiny)

    def apply(value: torch.Tensor) -> torch.Tensor:
        return value / shifted

    return apply


def _cg_statistics(
    diagnostics: tuple[CGDiagnostics, ...],
    iteration_limit: int,
) -> dict[str, int | float]:
    iterations = [item.iterations for item in diagnostics]
    return {
        "solver_count": len(diagnostics),
        "converged_count": sum(item.converged for item in diagnostics),
        "hit_iteration_limit_count": sum(
            item.iterations == int(iteration_limit) for item in diagnostics
        ),
        "total_iterations": sum(iterations),
        "mean_iterations": sum(iterations) / len(iterations),
        "maximum_iterations": max(iterations),
        "maximum_relative_residual": max(
            item.relative_residual for item in diagnostics
        ),
    }


def refit_chunk_query_factors(
    dataset: ChunkFisherDataset,
    *,
    encoders: torch.Tensor,
    initial_query_factors: torch.Tensor | None = None,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, tuple[CGDiagnostics, ...]]:
    """Solve each query-head factor with fixed group-shared chunk encoders."""

    dataset.validate()
    mapping = dataset.head_to_group.to(device=encoders.device, dtype=torch.long)
    rank = int(encoders.shape[-1])
    fitted = encoders.new_empty(dataset.heads, dataset.head_dim, rank)
    diagnostics = []
    scale = float(dataset.scaling)
    for head, group in enumerate(mapping.tolist()):
        terms = []
        rhs = encoders.new_zeros(dataset.head_dim, rank)
        diagonal = encoders.new_zeros(dataset.head_dim, rank)
        for example in dataset.examples:
            query = example.queries[head].to(encoders)
            features = example.features_by_group[group].to(encoders)
            latent = features @ encoders[group]
            teacher = example.teacher_logits[head].to(encoders)
            target = teacher - example.base_logits[head].to(encoders)
            probability = torch.softmax(teacher, dim=-1)
            mean = probability @ latent
            covariance = latent.mT @ (probability[:, None] * latent)
            covariance -= torch.outer(mean, mean)
            covariance = 0.5 * (covariance + covariance.mT)
            cross = latent.mT @ fisher_multiply(probability, target)
            rhs.add_(scale * torch.outer(query, cross))
            diagonal.add_(
                scale**2
                * torch.outer(query.square(), covariance.diagonal().clamp_min(0))
            )
            terms.append((query, covariance))

        def operator(value: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(value)
            for query, covariance in terms:
                result.add_(
                    scale**2
                    * torch.outer(query, covariance @ (query @ value))
                )
            return result

        damping = _relative_damping(diagonal, relative_damping)
        initial = (
            torch.zeros_like(rhs)
            if initial_query_factors is None
            else initial_query_factors[head].to(rhs)
        )
        delta_rhs = rhs - operator(initial) - damping * initial
        delta, record = conjugate_gradient_matrix(
            operator,
            delta_rhs,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            absolute_damping=damping,
            preconditioner=_preconditioner(diagonal, damping),
        )
        fitted[head] = initial + delta
        diagnostics.append(record)
    return fitted, tuple(diagnostics)


def refit_chunk_encoders(
    dataset: ChunkFisherDataset,
    *,
    query_factors: torch.Tensor,
    initial_encoders: torch.Tensor | None = None,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, tuple[CGDiagnostics, ...]]:
    """Solve one shared feature encoder for each physical GQA KV group."""

    dataset.validate()
    mapping = dataset.head_to_group.to(
        device=query_factors.device,
        dtype=torch.long,
    )
    rank = int(query_factors.shape[-1])
    encoders = query_factors.new_empty(dataset.groups, dataset.feature_dim, rank)
    diagnostics = []
    scale = float(dataset.scaling)
    for group in range(dataset.groups):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        terms = []
        rhs = query_factors.new_zeros(dataset.feature_dim, rank)
        diagonal = query_factors.new_zeros(dataset.feature_dim, rank)
        for example in dataset.examples:
            features = example.features_by_group[group].to(query_factors)
            queries = example.queries.index_select(0, heads).to(query_factors)
            head_factors = query_factors.index_select(0, heads)
            query_codes = torch.einsum("hd,hdr->hr", queries, head_factors)
            teacher = example.teacher_logits.index_select(0, heads).to(
                query_factors
            )
            target = teacher - example.base_logits.index_select(0, heads).to(
                query_factors
            )
            probabilities = torch.softmax(teacher, dim=-1)
            feature_mean = torch.einsum("hc,ci->hi", probabilities, features)
            feature_second = torch.einsum(
                "hc,ci->hi",
                probabilities,
                features.square(),
            )
            feature_variance = (feature_second - feature_mean.square()).clamp_min(0)
            weighted_target = fisher_multiply(probabilities, target)
            cross = torch.einsum("ci,hc->hi", features, weighted_target)
            rhs.add_(scale * torch.einsum("hi,hr->ir", cross, query_codes))
            diagonal.add_(
                scale**2
                * torch.einsum(
                    "hi,hr->ir",
                    feature_variance,
                    query_codes.square(),
                )
            )
            terms.append((features, probabilities, query_codes))

        def operator(value: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(value)
            for features, probabilities, query_codes in terms:
                latent = features @ value
                predicted = torch.einsum("hr,cr->hc", query_codes, latent)
                transformed = fisher_multiply(probabilities, predicted)
                feature_gradient = torch.einsum("ci,hc->hi", features, transformed)
                result.add_(
                    scale**2
                    * torch.einsum("hi,hr->ir", feature_gradient, query_codes)
                )
            return result

        damping = _relative_damping(diagonal, relative_damping)
        initial = (
            torch.zeros_like(rhs)
            if initial_encoders is None
            else initial_encoders[group].to(rhs)
        )
        delta_rhs = rhs - operator(initial) - damping * initial
        delta, record = conjugate_gradient_matrix(
            operator,
            delta_rhs,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            absolute_damping=damping,
            preconditioner=_preconditioner(diagonal, damping),
        )
        encoders[group] = initial + delta
        diagnostics.append(record)
    return encoders, tuple(diagnostics)


def balance_chunk_factors(
    encoders: torch.Tensor,
    query_factors: torch.Tensor,
    head_to_group: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """QR-balance each group without changing any bilinear routing score."""

    balanced_encoders = torch.empty_like(encoders)
    balanced_queries = query_factors.clone()
    mapping = head_to_group.to(device=query_factors.device, dtype=torch.long)
    for group in range(int(encoders.shape[0])):
        orthogonal, transform = torch.linalg.qr(encoders[group], mode="reduced")
        balanced_encoders[group] = orthogonal
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        balanced_queries.index_copy_(
            0,
            heads,
            query_factors.index_select(0, heads) @ transform.mT,
        )
    return balanced_encoders, balanced_queries


def fit_chunk_fisher_landmarks(
    train: ChunkFisherDataset,
    heldout: ChunkFisherDataset,
    *,
    initial_encoders: torch.Tensor,
    initial_query_factors: torch.Tensor,
    sweeps: int,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
) -> ChunkFisherFit:
    """Run direct Chunk-Fisher ALS and finish with a query closure solve."""

    train.validate()
    heldout.validate()
    assert train.head_to_group.tolist() == heldout.head_to_group.tolist()
    assert train.feature_dim == heldout.feature_dim
    assert sweeps > 0
    encoders = initial_encoders.to(
        device=train.examples[0].queries.device,
        dtype=torch.float32,
    ).clone()
    queries = initial_query_factors.to(encoders).clone()
    half_steps = []
    encoder_diagnostics = []
    final_query_diagnostics = ()

    def synchronize() -> None:
        if encoders.is_cuda:
            torch.cuda.synchronize(encoders.device)

    def record(
        sweep: int,
        boundary: str,
        diagnostics,
        solve_wall_seconds: float,
    ) -> None:
        synchronize()
        diagnostic_start = time.perf_counter()
        train_metrics = chunk_fisher_metrics(train, encoders, queries)
        heldout_metrics = chunk_fisher_metrics(heldout, encoders, queries)
        synchronize()
        diagnostic_wall_seconds = time.perf_counter() - diagnostic_start
        statistics = _cg_statistics(diagnostics, max_iterations)
        half_steps.append(
            ChunkFisherHalfStep(
                sweep=sweep,
                boundary=boundary,
                train=train_metrics,
                heldout=heldout_metrics,
                solve_wall_seconds=solve_wall_seconds,
                diagnostic_wall_seconds=diagnostic_wall_seconds,
                total_wall_seconds=solve_wall_seconds + diagnostic_wall_seconds,
                **statistics,
            )
        )

    for sweep in range(1, int(sweeps) + 1):
        synchronize()
        solve_start = time.perf_counter()
        queries, final_query_diagnostics = refit_chunk_query_factors(
            train,
            encoders=encoders,
            initial_query_factors=queries,
            relative_damping=relative_damping,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
        )
        synchronize()
        record(
            sweep,
            "query",
            final_query_diagnostics,
            time.perf_counter() - solve_start,
        )
        synchronize()
        solve_start = time.perf_counter()
        encoders, encoder_step = refit_chunk_encoders(
            train,
            query_factors=queries,
            initial_encoders=encoders,
            relative_damping=relative_damping,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
        )
        encoders, queries = balance_chunk_factors(
            encoders,
            queries,
            train.head_to_group,
        )
        encoder_diagnostics.append(encoder_step)
        synchronize()
        record(
            sweep,
            "encoder",
            encoder_step,
            time.perf_counter() - solve_start,
        )
        query_record, encoder_record = half_steps[-2:]
        print(
            f"Chunk-Fisher sweep {sweep}/{sweeps}: "
            f"train={encoder_record.train['fisher_nmse']:.6g} "
            f"heldout={encoder_record.heldout['fisher_nmse']:.6g} "
            f"query_cg={query_record.mean_iterations:.1f}/{query_record.maximum_iterations} "
            f"encoder_cg={encoder_record.mean_iterations:.1f}/{encoder_record.maximum_iterations} "
            f"wall={query_record.total_wall_seconds + encoder_record.total_wall_seconds:.3f}s",
            flush=True,
        )
    synchronize()
    solve_start = time.perf_counter()
    queries, final_query_diagnostics = refit_chunk_query_factors(
        train,
        encoders=encoders,
        initial_query_factors=queries,
        relative_damping=relative_damping,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    synchronize()
    record(
        int(sweeps),
        "final_query_closure",
        final_query_diagnostics,
        time.perf_counter() - solve_start,
    )
    return ChunkFisherFit(
        encoders=encoders,
        query_factors=queries,
        half_steps=tuple(half_steps),
        final_query_diagnostics=final_query_diagnostics,
        encoder_diagnostics=tuple(encoder_diagnostics),
    )


def _packed_targets(
    window: PackedChunkFisherWindow,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = window.valid_mask().to(device=like.device)
    teacher = window.teacher_logits.to(like)
    base = window.base_logits.to(like)
    probability = torch.softmax(
        teacher.masked_fill(~mask[:, None], -torch.inf),
        dim=-1,
    )
    target = (teacher - base).masked_fill(~mask[:, None], 0.0)
    return teacher, probability, target


def packed_factor_logits(
    dataset: PackedChunkFisherDataset,
    window: PackedChunkFisherWindow,
    encoders: torch.Tensor,
    query_factors: torch.Tensor,
) -> torch.Tensor:
    mapping = dataset.head_to_group.to(device=encoders.device, dtype=torch.long)
    features = window.features_by_group.to(encoders)
    queries = window.queries.to(query_factors)
    chunk_codes = torch.einsum("gci,gir->gcr", features, encoders)
    head_codes = torch.einsum("qhd,hdr->qhr", queries, query_factors)
    return float(dataset.scaling) * torch.einsum(
        "qhr,hcr->qhc",
        head_codes,
        chunk_codes.index_select(0, mapping),
    )


def packed_chunk_fisher_metrics(
    dataset: PackedChunkFisherDataset,
    encoders: torch.Tensor,
    query_factors: torch.Tensor,
    *,
    routed_chunks: int = 244,
) -> dict[str, float]:
    dataset.validate()
    fisher_loss = 0.0
    base_fisher_energy = 0.0
    score_error = 0.0
    score_energy = 0.0
    recalls = []
    masses = []
    for window in dataset.windows:
        teacher, probabilities, target = _packed_targets(window, encoders)
        correction = packed_factor_logits(dataset, window, encoders, query_factors)
        mask = window.valid_mask().to(device=encoders.device)
        error = (correction - target).masked_fill(~mask[:, None], 0.0)
        fisher_loss += float(
            0.5 * torch.sum(error * fisher_multiply(probabilities, error))
        )
        base_fisher_energy += float(
            0.5 * torch.sum(target * fisher_multiply(probabilities, target))
        )
        student = window.base_logits.to(encoders) + correction
        difference = (student - teacher).masked_fill(~mask[:, None], 0.0)
        score_error += float(difference.double().square().sum())
        score_energy += float(
            teacher.masked_fill(~mask[:, None], 0.0).double().square().sum()
        )
        for query_index, count_value in enumerate(window.candidate_counts.tolist()):
            count = int(count_value)
            selected_count = min(int(routed_chunks), count)
            current_student = student[query_index, :, :count]
            current_teacher = teacher[query_index, :, :count]
            current_probability = probabilities[query_index, :, :count]
            selected = current_student.topk(selected_count, dim=-1).indices
            exact = current_teacher.topk(selected_count, dim=-1).indices
            selected_mask = torch.zeros_like(
                current_teacher,
                dtype=torch.bool,
            ).scatter_(-1, selected, True)
            exact_mask = torch.zeros_like(
                current_teacher,
                dtype=torch.bool,
            ).scatter_(-1, exact, True)
            recalls.extend(
                (
                    (selected_mask & exact_mask).sum(-1).float()
                    / selected_count
                ).tolist()
            )
            masses.extend(
                current_probability.gather(-1, selected).sum(-1).tolist()
            )
    return {
        "fisher_loss": fisher_loss,
        "fisher_nmse": fisher_loss
        / max(base_fisher_energy, torch.finfo(torch.float64).tiny),
        "base_fisher_energy": base_fisher_energy,
        "chunk_logit_rel_mse": score_error / score_energy,
        "exact_chunk_support_recall": sum(recalls) / len(recalls),
        "routed_candidate_attention_mass": sum(masses) / len(masses),
    }


def compact_packed_chunk_fisher_dataset(
    dataset: PackedChunkFisherDataset,
    *,
    statistic_batch_size: int = 8,
) -> CompactChunkFisherDataset:
    """Eliminate Chunk candidates into exact per-query Gram/cross statistics."""

    dataset.validate()
    assert statistic_batch_size > 0
    device = dataset.windows[0].queries.device
    dtype = dataset.windows[0].queries.dtype
    observations = sum(int(window.queries.shape[0]) for window in dataset.windows)
    queries = torch.empty(
        dataset.heads,
        observations,
        dataset.head_dim,
        device=device,
        dtype=dtype,
    )
    grams = torch.empty(
        dataset.heads,
        observations,
        dataset.feature_dim,
        dataset.feature_dim,
        device=device,
        dtype=dtype,
    )
    crosses = torch.empty(
        dataset.heads,
        observations,
        dataset.feature_dim,
        device=device,
        dtype=dtype,
    )
    mapping = dataset.head_to_group.to(device=device, dtype=torch.long)
    energy = 0.0
    offset = 0
    for window_index, window in enumerate(dataset.windows):
        query_count = int(window.queries.shape[0])
        queries[:, offset : offset + query_count].copy_(
            window.queries.transpose(0, 1)
        )
        _, probabilities, target = _packed_targets(window, queries)
        features_by_group = window.features_by_group.to(queries)
        for group in range(dataset.groups):
            heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
            head_count = int(heads.numel())
            features = features_by_group[group]
            probability = probabilities.index_select(1, heads).reshape(
                query_count * head_count,
                -1,
            )
            current_target = target.index_select(1, heads).reshape(
                query_count * head_count,
                -1,
            )
            group_grams = torch.empty(
                query_count * head_count,
                dataset.feature_dim,
                dataset.feature_dim,
                device=device,
                dtype=dtype,
            )
            group_crosses = torch.empty(
                query_count * head_count,
                dataset.feature_dim,
                device=device,
                dtype=dtype,
            )
            for start in range(0, query_count * head_count, statistic_batch_size):
                stop = min(start + statistic_batch_size, query_count * head_count)
                current_probability = probability[start:stop]
                current_residual = current_target[start:stop]
                feature_mean = current_probability @ features
                weighted_features = (
                    torch.sqrt(current_probability).unsqueeze(-1)
                    * features.unsqueeze(0)
                )
                second = torch.bmm(
                    weighted_features.transpose(1, 2),
                    weighted_features,
                )
                gram = second - torch.einsum(
                    "bi,bj->bij",
                    feature_mean,
                    feature_mean,
                )
                group_grams[start:stop].copy_(
                    0.5 * (gram + gram.transpose(-1, -2))
                )
                residual_mean = torch.sum(
                    current_probability * current_residual,
                    dim=-1,
                    keepdim=True,
                )
                centered_residual = current_residual - residual_mean
                group_crosses[start:stop].copy_(
                    (current_probability * centered_residual) @ features
                )
                energy += float(
                    0.5
                    * torch.sum(
                        current_probability * centered_residual.square()
                    )
                )
            group_grams = group_grams.reshape(
                query_count,
                head_count,
                dataset.feature_dim,
                dataset.feature_dim,
            ).transpose(0, 1)
            group_crosses = group_crosses.reshape(
                query_count,
                head_count,
                dataset.feature_dim,
            ).transpose(0, 1)
            for local, head in enumerate(heads.tolist()):
                grams[head, offset : offset + query_count].copy_(
                    group_grams[local]
                )
                crosses[head, offset : offset + query_count].copy_(
                    group_crosses[local]
                )
        offset += query_count
        print(
            f"Chunk-Fisher sufficient statistics "
            f"{window_index + 1}/{len(dataset.windows)}",
            flush=True,
        )
    assert offset == observations
    result = CompactChunkFisherDataset(
        queries_by_head=queries,
        fisher_grams_by_head=grams,
        target_cross_by_head=crosses,
        head_to_group=mapping,
        scaling=dataset.scaling,
        target_fisher_energy=energy,
    )
    result.validate()
    return result


def compact_chunk_fisher_metrics(
    dataset: CompactChunkFisherDataset,
    encoders: torch.Tensor,
    query_factors: torch.Tensor,
) -> dict[str, float]:
    """Evaluate the exact quadratic using only sufficient statistics."""

    dataset.validate()
    mapping = dataset.head_to_group.to(device=encoders.device, dtype=torch.long)
    loss = float(dataset.target_fisher_energy)
    for head, group in enumerate(mapping.tolist()):
        query = dataset.queries_by_head[head].to(encoders)
        gram = dataset.fisher_grams_by_head[head].to(encoders)
        cross = dataset.target_cross_by_head[head].to(encoders)
        codes = float(dataset.scaling) * query @ query_factors[head]
        coefficients = codes @ encoders[group].mT
        loss += float(
            0.5
            * torch.einsum(
                "ni,nij,nj->",
                coefficients,
                gram,
                coefficients,
            )
            - torch.sum(coefficients * cross)
        )
    return {
        "fisher_loss": max(loss, 0.0),
        "fisher_nmse": max(loss, 0.0) / dataset.target_fisher_energy,
        "base_fisher_energy": dataset.target_fisher_energy,
    }


def refit_compact_chunk_query_factors(
    dataset: CompactChunkFisherDataset,
    *,
    encoders: torch.Tensor,
    initial_query_factors: torch.Tensor,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, tuple[CGDiagnostics, ...]]:
    """Solve query factors after Chunk candidates have been eliminated."""

    dataset.validate()
    mapping = dataset.head_to_group.to(device=encoders.device, dtype=torch.long)
    fitted = encoders.new_empty(dataset.heads, dataset.head_dim, encoders.shape[-1])
    diagnostics = []
    scale = float(dataset.scaling)
    for head, group in enumerate(mapping.tolist()):
        query = dataset.queries_by_head[head].to(encoders)
        gram = dataset.fisher_grams_by_head[head].to(encoders)
        cross = dataset.target_cross_by_head[head].to(encoders)
        encoder = encoders[group]
        right = torch.einsum("ir,nij,js->nrs", encoder, gram, encoder)
        weighted_target = cross @ encoder
        rhs = scale * query.mT @ weighted_target

        def operator(value: torch.Tensor) -> torch.Tensor:
            codes = query @ value
            transformed = torch.einsum("nr,nrs->ns", codes, right)
            return scale**2 * query.mT @ transformed

        diagonal = scale**2 * torch.einsum(
            "ni,nr->ir",
            query.square(),
            right.diagonal(dim1=-2, dim2=-1).clamp_min(0),
        )
        damping = _relative_damping(diagonal, relative_damping)
        initial = initial_query_factors[head].to(rhs)
        delta_rhs = rhs - operator(initial) - damping * initial
        delta, record = conjugate_gradient_matrix(
            operator,
            delta_rhs,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            absolute_damping=damping,
            preconditioner=_preconditioner(diagonal, damping),
        )
        fitted[head] = initial + delta
        diagnostics.append(record)
    return fitted, tuple(diagnostics)


def refit_compact_chunk_encoders(
    dataset: CompactChunkFisherDataset,
    *,
    query_factors: torch.Tensor,
    initial_encoders: torch.Tensor,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, tuple[CGDiagnostics, ...]]:
    """Solve encoders from fixed-size per-query Fisher statistics."""

    dataset.validate()
    mapping = dataset.head_to_group.to(device=query_factors.device, dtype=torch.long)
    encoders = query_factors.new_empty(
        dataset.groups,
        dataset.feature_dim,
        query_factors.shape[-1],
    )
    diagnostics = []
    scale = float(dataset.scaling)
    for group in range(dataset.groups):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        query = dataset.queries_by_head.index_select(0, heads).to(query_factors)
        gram = dataset.fisher_grams_by_head.index_select(0, heads).to(query_factors)
        cross = dataset.target_cross_by_head.index_select(0, heads).to(query_factors)
        factors = query_factors.index_select(0, heads)
        codes = scale * torch.einsum("hnd,hdr->hnr", query, factors)
        rhs = torch.einsum("hni,hnr->ir", cross, codes)

        def operator(value: torch.Tensor) -> torch.Tensor:
            projected = torch.einsum("ir,hnr->hni", value, codes)
            transformed = torch.einsum("hnij,hnj->hni", gram, projected)
            return torch.einsum("hni,hnr->ir", transformed, codes)

        diagonal = torch.einsum(
            "hni,hnr->ir",
            gram.diagonal(dim1=-2, dim2=-1).clamp_min(0),
            codes.square(),
        )
        damping = _relative_damping(diagonal, relative_damping)
        initial = initial_encoders[group].to(rhs)
        delta_rhs = rhs - operator(initial) - damping * initial
        delta, record = conjugate_gradient_matrix(
            operator,
            delta_rhs,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            absolute_damping=damping,
            preconditioner=_preconditioner(diagonal, damping),
        )
        encoders[group] = initial + delta
        diagnostics.append(record)
    return encoders, tuple(diagnostics)


def fit_compact_chunk_fisher_landmarks(
    train: CompactChunkFisherDataset,
    heldout: CompactChunkFisherDataset,
    *,
    initial_encoders: torch.Tensor,
    initial_query_factors: torch.Tensor,
    sweeps: int,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
) -> ChunkFisherFit:
    """Run exact Chunk-Fisher ALS without revisiting Chunk candidates."""

    train.validate()
    heldout.validate()
    assert train.head_to_group.tolist() == heldout.head_to_group.tolist()
    assert train.feature_dim == heldout.feature_dim
    assert sweeps > 0
    encoders = initial_encoders.to(
        device=train.queries_by_head.device,
        dtype=torch.float32,
    ).clone()
    queries = initial_query_factors.to(encoders).clone()
    half_steps = []
    encoder_diagnostics = []
    final_query_diagnostics = ()

    def synchronize() -> None:
        if encoders.is_cuda:
            torch.cuda.synchronize(encoders.device)

    def record(sweep: int, boundary: str, diagnostics, solve_seconds: float) -> None:
        synchronize()
        diagnostic_start = time.perf_counter()
        train_metrics = compact_chunk_fisher_metrics(train, encoders, queries)
        heldout_metrics = compact_chunk_fisher_metrics(heldout, encoders, queries)
        synchronize()
        diagnostic_seconds = time.perf_counter() - diagnostic_start
        statistics = _cg_statistics(diagnostics, max_iterations)
        half_steps.append(
            ChunkFisherHalfStep(
                sweep=sweep,
                boundary=boundary,
                train=train_metrics,
                heldout=heldout_metrics,
                solve_wall_seconds=solve_seconds,
                diagnostic_wall_seconds=diagnostic_seconds,
                total_wall_seconds=solve_seconds + diagnostic_seconds,
                **statistics,
            )
        )

    for sweep in range(1, int(sweeps) + 1):
        synchronize()
        start = time.perf_counter()
        queries, final_query_diagnostics = refit_compact_chunk_query_factors(
            train,
            encoders=encoders,
            initial_query_factors=queries,
            relative_damping=relative_damping,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
        )
        synchronize()
        record(sweep, "query", final_query_diagnostics, time.perf_counter() - start)
        synchronize()
        start = time.perf_counter()
        encoders, encoder_step = refit_compact_chunk_encoders(
            train,
            query_factors=queries,
            initial_encoders=encoders,
            relative_damping=relative_damping,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
        )
        encoders, queries = balance_chunk_factors(encoders, queries, train.head_to_group)
        encoder_diagnostics.append(encoder_step)
        synchronize()
        record(sweep, "encoder", encoder_step, time.perf_counter() - start)
        query_record, encoder_record = half_steps[-2:]
        print(
            f"Compact Chunk-Fisher sweep {sweep}/{sweeps}: "
            f"train={encoder_record.train['fisher_nmse']:.6g} "
            f"heldout={encoder_record.heldout['fisher_nmse']:.6g} "
            f"query_cg={query_record.mean_iterations:.1f}/{query_record.maximum_iterations} "
            f"encoder_cg={encoder_record.mean_iterations:.1f}/{encoder_record.maximum_iterations} "
            f"wall={query_record.total_wall_seconds + encoder_record.total_wall_seconds:.3f}s",
            flush=True,
        )
    synchronize()
    start = time.perf_counter()
    queries, final_query_diagnostics = refit_compact_chunk_query_factors(
        train,
        encoders=encoders,
        initial_query_factors=queries,
        relative_damping=relative_damping,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    synchronize()
    record(
        int(sweeps),
        "final_query_closure",
        final_query_diagnostics,
        time.perf_counter() - start,
    )
    return ChunkFisherFit(
        encoders=encoders,
        query_factors=queries,
        half_steps=tuple(half_steps),
        final_query_diagnostics=final_query_diagnostics,
        encoder_diagnostics=tuple(encoder_diagnostics),
    )


def refit_packed_chunk_query_factors(
    dataset: PackedChunkFisherDataset,
    *,
    encoders: torch.Tensor,
    initial_query_factors: torch.Tensor,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, tuple[CGDiagnostics, ...]]:
    """Compact every window to rank-by-rank Fisher covariances, then solve U."""

    dataset.validate()
    mapping = dataset.head_to_group.to(device=encoders.device, dtype=torch.long)
    rank = int(encoders.shape[-1])
    query_rows = [[] for _ in range(dataset.heads)]
    covariance_rows = [[] for _ in range(dataset.heads)]
    cross_rows = [[] for _ in range(dataset.heads)]
    for window in dataset.windows:
        _, probabilities, target = _packed_targets(window, encoders)
        features = window.features_by_group.to(encoders)
        latent = torch.einsum("gci,gir->gcr", features, encoders)
        for group in range(dataset.groups):
            heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
            group_probability = probabilities.index_select(1, heads)
            group_target = target.index_select(1, heads)
            group_latent = latent[group]
            means = torch.einsum("qhc,cr->qhr", group_probability, group_latent)
            second = torch.einsum(
                "qhc,cr,cs->qhrs",
                group_probability,
                group_latent,
                group_latent,
            )
            covariance = second - torch.einsum("qhr,qhs->qhrs", means, means)
            covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
            cross = torch.einsum(
                "cr,qhc->qhr",
                group_latent,
                fisher_multiply(group_probability, group_target),
            )
            for local, head in enumerate(heads.tolist()):
                query_rows[head].append(window.queries[:, head].to(encoders))
                covariance_rows[head].append(covariance[:, local])
                cross_rows[head].append(cross[:, local])
    fitted = encoders.new_empty(
        dataset.heads,
        dataset.head_dim,
        rank,
    )
    diagnostics = []
    scale = float(dataset.scaling)
    for head in range(dataset.heads):
        query = torch.cat(query_rows[head], dim=0)
        covariance = torch.cat(covariance_rows[head], dim=0)
        cross = torch.cat(cross_rows[head], dim=0)
        rhs = scale * query.mT @ cross
        diagonal = scale**2 * torch.einsum(
            "nd,nr->dr",
            query.square(),
            covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0),
        )

        def operator(value: torch.Tensor) -> torch.Tensor:
            codes = query @ value
            transformed = torch.einsum("nrs,ns->nr", covariance, codes)
            return scale**2 * query.mT @ transformed

        damping = _relative_damping(diagonal, relative_damping)
        initial = initial_query_factors[head].to(rhs)
        delta_rhs = rhs - operator(initial) - damping * initial
        delta, record = conjugate_gradient_matrix(
            operator,
            delta_rhs,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            absolute_damping=damping,
            preconditioner=_preconditioner(diagonal, damping),
        )
        fitted[head] = initial + delta
        diagnostics.append(record)
    return fitted, tuple(diagnostics)


def refit_packed_chunk_encoders(
    dataset: PackedChunkFisherDataset,
    *,
    query_factors: torch.Tensor,
    initial_encoders: torch.Tensor,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
    feature_eigensystems: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
) -> tuple[torch.Tensor, tuple[CGDiagnostics, ...]]:
    """Matrix-free group solves using window-shared flat or mean features."""

    dataset.validate()
    mapping = dataset.head_to_group.to(
        device=query_factors.device,
        dtype=torch.long,
    )
    rank = int(query_factors.shape[-1])
    encoders = query_factors.new_empty(dataset.groups, dataset.feature_dim, rank)
    diagnostics = []
    scale = float(dataset.scaling)
    for group in range(dataset.groups):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        terms = []
        rhs = query_factors.new_zeros(dataset.feature_dim, rank)
        diagonal = query_factors.new_zeros(dataset.feature_dim, rank)
        for window in dataset.windows:
            features = window.features_by_group[group].to(query_factors)
            _, probabilities, target = _packed_targets(window, query_factors)
            probabilities = probabilities.index_select(1, heads)
            target = target.index_select(1, heads)
            queries = window.queries.index_select(1, heads).to(query_factors)
            head_factors = query_factors.index_select(0, heads)
            query_codes = torch.einsum("qhd,hdr->qhr", queries, head_factors)
            weighted_target = fisher_multiply(probabilities, target)
            coefficients = torch.einsum(
                "qhc,qhr->cr",
                weighted_target,
                query_codes,
            )
            rhs.add_(scale * features.mT @ coefficients)
            weights = torch.einsum(
                "qhc,qhr->cr",
                probabilities,
                query_codes.square(),
            )
            diagonal.add_(scale**2 * features.square().mT @ weights)
            terms.append((features, probabilities, query_codes))

        def operator(value: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(value)
            for features, probabilities, query_codes in terms:
                latent = features @ value
                predicted = torch.einsum("qhr,cr->qhc", query_codes, latent)
                transformed = fisher_multiply(probabilities, predicted)
                coefficients = torch.einsum(
                    "qhc,qhr->cr",
                    transformed,
                    query_codes,
                )
                result.add_(scale**2 * features.mT @ coefficients)
            return result

        damping = _relative_damping(diagonal, relative_damping)
        initial = initial_encoders[group].to(rhs)
        delta_rhs = rhs - operator(initial) - damping * initial
        preconditioner = _preconditioner(diagonal, damping)
        if feature_eigensystems is not None:
            feature_values, feature_vectors = feature_eigensystems[group]
            feature_values = feature_values.to(diagonal)
            feature_vectors = feature_vectors.to(diagonal)
            rank_gram = torch.einsum("hnr,hns->rs", query_codes, query_codes)
            rank_values, rank_vectors = torch.linalg.eigh(
                0.5 * (rank_gram + rank_gram.mT)
            )
            rank_values.clamp_min_(0)
            approximate_diagonal = (
                torch.diagonal(
                    feature_vectors
                    @ torch.diag(feature_values)
                    @ feature_vectors.mT
                ).unsqueeze(1)
                * torch.diagonal(
                    rank_vectors @ torch.diag(rank_values) @ rank_vectors.mT
                ).unsqueeze(0)
            )
            ratio = float(diagonal.mean()) / max(
                float(approximate_diagonal.mean()),
                torch.finfo(diagonal.dtype).tiny,
            )
            denominator = (
                ratio * feature_values.unsqueeze(1) * rank_values.unsqueeze(0)
                + damping
            ).clamp_min(torch.finfo(diagonal.dtype).tiny)

            def preconditioner(value: torch.Tensor) -> torch.Tensor:
                transformed = feature_vectors.mT @ value @ rank_vectors
                return feature_vectors @ (transformed / denominator) @ rank_vectors.mT

        delta, record = conjugate_gradient_matrix(
            operator,
            delta_rhs,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            absolute_damping=damping,
            preconditioner=preconditioner,
        )
        encoders[group] = initial + delta
        diagnostics.append(record)
    return encoders, tuple(diagnostics)


def packed_chunk_feature_fisher_grams(
    dataset: PackedChunkFisherDataset,
) -> torch.Tensor:
    """Aggregate exact feature-side Fisher Grams for a reusable preconditioner."""

    dataset.validate()
    like = dataset.windows[0].queries
    mapping = dataset.head_to_group.to(device=like.device, dtype=torch.long)
    result = torch.zeros(
        dataset.groups,
        dataset.feature_dim,
        dataset.feature_dim,
        device=like.device,
        dtype=like.dtype,
    )
    for window_index, window in enumerate(dataset.windows):
        _, probabilities, _ = _packed_targets(window, like)
        for group in range(dataset.groups):
            heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
            probability = probabilities.index_select(1, heads).flatten(0, 1)
            features = window.features_by_group[group].to(like)
            mass = probability.sum(dim=0)
            means = probability @ features
            result[group].add_(features.mT @ (mass.unsqueeze(1) * features))
            result[group].sub_(means.mT @ means)
        print(
            f"Chunk-Fisher feature preconditioner "
            f"{window_index + 1}/{len(dataset.windows)}",
            flush=True,
        )
    return 0.5 * (result + result.mT)


def fit_packed_chunk_fisher_landmarks(
    train: PackedChunkFisherDataset,
    heldout: PackedChunkFisherDataset,
    *,
    initial_encoders: torch.Tensor,
    initial_query_factors: torch.Tensor,
    sweeps: int,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
) -> ChunkFisherFit:
    """Production packed-window direct Chunk-Fisher ALS."""

    train.validate()
    heldout.validate()
    assert train.head_to_group.tolist() == heldout.head_to_group.tolist()
    assert train.feature_dim == heldout.feature_dim
    assert sweeps > 0
    device = train.windows[0].queries.device
    encoders = initial_encoders.to(device=device, dtype=torch.float32).clone()
    queries = initial_query_factors.to(encoders).clone()
    half_steps = []
    encoder_diagnostics = []
    final_query_diagnostics = ()
    preconditioner_wall_seconds = 0.0
    feature_eigensystems = None

    def synchronize() -> None:
        if encoders.is_cuda:
            torch.cuda.synchronize(encoders.device)

    def record(
        sweep: int,
        boundary: str,
        diagnostics,
        solve_wall_seconds: float,
    ) -> None:
        synchronize()
        diagnostic_start = time.perf_counter()
        train_metrics = packed_chunk_fisher_metrics(train, encoders, queries)
        heldout_metrics = packed_chunk_fisher_metrics(heldout, encoders, queries)
        synchronize()
        diagnostic_wall_seconds = time.perf_counter() - diagnostic_start
        statistics = _cg_statistics(diagnostics, max_iterations)
        half_steps.append(
            ChunkFisherHalfStep(
                sweep=sweep,
                boundary=boundary,
                train=train_metrics,
                heldout=heldout_metrics,
                solve_wall_seconds=solve_wall_seconds,
                diagnostic_wall_seconds=diagnostic_wall_seconds,
                total_wall_seconds=solve_wall_seconds + diagnostic_wall_seconds,
                **statistics,
            )
        )

    if train.feature_dim > 128:
        synchronize()
        preconditioner_start = time.perf_counter()
        feature_grams = packed_chunk_feature_fisher_grams(train)
        feature_eigensystems = tuple(
            (
                values.clamp_min(0),
                vectors,
            )
            for values, vectors in (
                torch.linalg.eigh(feature_grams[group])
                for group in range(train.groups)
            )
        )
        synchronize()
        preconditioner_wall_seconds = time.perf_counter() - preconditioner_start
        print(
            f"Chunk-Fisher Kronecker preconditioner "
            f"wall={preconditioner_wall_seconds:.3f}s",
            flush=True,
        )

    for sweep in range(1, int(sweeps) + 1):
        synchronize()
        solve_start = time.perf_counter()
        queries, final_query_diagnostics = refit_packed_chunk_query_factors(
            train,
            encoders=encoders,
            initial_query_factors=queries,
            relative_damping=relative_damping,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
        )
        synchronize()
        record(
            sweep,
            "query",
            final_query_diagnostics,
            time.perf_counter() - solve_start,
        )
        synchronize()
        solve_start = time.perf_counter()
        encoders, encoder_step = refit_packed_chunk_encoders(
            train,
            query_factors=queries,
            initial_encoders=encoders,
            relative_damping=relative_damping,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            feature_eigensystems=feature_eigensystems,
        )
        encoders, queries = balance_chunk_factors(
            encoders,
            queries,
            train.head_to_group,
        )
        encoder_diagnostics.append(encoder_step)
        synchronize()
        record(
            sweep,
            "encoder",
            encoder_step,
            time.perf_counter() - solve_start,
        )
        query_record, encoder_record = half_steps[-2:]
        print(
            f"Packed Chunk-Fisher sweep {sweep}/{sweeps}: "
            f"train={encoder_record.train['fisher_nmse']:.6g} "
            f"heldout={encoder_record.heldout['fisher_nmse']:.6g} "
            f"query_cg={query_record.mean_iterations:.1f}/{query_record.maximum_iterations} "
            f"encoder_cg={encoder_record.mean_iterations:.1f}/{encoder_record.maximum_iterations} "
            f"wall={query_record.total_wall_seconds + encoder_record.total_wall_seconds:.3f}s",
            flush=True,
        )
    synchronize()
    solve_start = time.perf_counter()
    queries, final_query_diagnostics = refit_packed_chunk_query_factors(
        train,
        encoders=encoders,
        initial_query_factors=queries,
        relative_damping=relative_damping,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    synchronize()
    record(
        int(sweeps),
        "final_query_closure",
        final_query_diagnostics,
        time.perf_counter() - solve_start,
    )
    return ChunkFisherFit(
        encoders=encoders,
        query_factors=queries,
        half_steps=tuple(half_steps),
        final_query_diagnostics=final_query_diagnostics,
        encoder_diagnostics=tuple(encoder_diagnostics),
        preconditioner_wall_seconds=preconditioner_wall_seconds,
    )


__all__ = [
    "ChunkFisherDataset",
    "ChunkFisherExample",
    "ChunkFisherFit",
    "ChunkFisherHalfStep",
    "CompactChunkFisherDataset",
    "PackedChunkFisherDataset",
    "PackedChunkFisherWindow",
    "adapt_token_residual_initialization",
    "balance_chunk_factors",
    "chunk_fisher_metrics",
    "chunk_mean_logits",
    "chunk_teacher_logits",
    "compact_chunk_fisher_metrics",
    "compact_packed_chunk_fisher_dataset",
    "factor_logits",
    "fisher_multiply",
    "fit_chunk_fisher_landmarks",
    "fit_compact_chunk_fisher_landmarks",
    "fit_packed_chunk_fisher_landmarks",
    "packed_chunk_fisher_metrics",
    "packed_chunk_feature_fisher_grams",
    "packed_factor_logits",
    "refit_chunk_encoders",
    "refit_chunk_query_factors",
    "refit_compact_chunk_encoders",
    "refit_compact_chunk_query_factors",
    "refit_packed_chunk_encoders",
    "refit_packed_chunk_query_factors",
    "residual_chunk_features",
]
