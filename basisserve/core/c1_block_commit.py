"""Exact C1 block-schedule diagnostics and teacher-forced NLL evaluation.

These utilities isolate BF16 execution-order effects from Shadow-Key proposal
error.  Every token is fixed by the caller, all pending KV comes from the exact
C1 target, and a block schedule commits that pending target KV directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.core.c1_shadow_kv import C1ShadowKeyValueCache


@dataclass(frozen=True)
class TargetScheduleBlock:
    block_index: int
    relative_start: int
    absolute_start: int
    length: int
    aligned_logits: Tensor


@dataclass(frozen=True)
class TargetScheduleResult:
    block_length: int
    prefill_tokens: int
    evaluated_tokens: int
    block_count: int
    nll_sum: float
    token_nlls: tuple[float, ...]
    top1_token_ids: tuple[int, ...]
    top1_margins: tuple[float, ...]
    prediction_logits: Tensor | None

    @property
    def mean_nll(self) -> float:
        if self.evaluated_tokens == 0:
            raise ZeroDivisionError("target schedule contains no evaluated tokens")
        return self.nll_sum / self.evaluated_tokens


@dataclass(frozen=True)
class CacheDriftRecord:
    block_index: int
    layer_index: int
    relative_start: int
    length: int
    key_relative_l2: float
    key_maximum_absolute_error: float
    key_cosine_similarity: float
    c1_value_relative_l2: float
    c1_value_maximum_absolute_error: float
    c1_value_cosine_similarity: float


@dataclass(frozen=True)
class ExactBlockScheduleComparison:
    block_length: int
    evaluated_tokens: int
    block_count: int
    sequential_nll_sum: float
    block_nll_sum: float
    sequential_token_nlls: tuple[float, ...]
    block_token_nlls: tuple[float, ...]
    sequential_top1_margins: tuple[float, ...]
    block_top1_margins: tuple[float, ...]
    top1_agreements: int
    top1_comparisons: int
    first_top1_disagreement: int | None
    sequential_label_top1_matches: int
    block_label_top1_matches: int
    kl_sequential_to_block: tuple[float, ...]
    logit_relative_l2_by_block: tuple[float, ...]
    logit_maximum_absolute_error_by_block: tuple[float, ...]
    top5_overlap_fraction_sum: float
    cache_drift: tuple[CacheDriftRecord, ...]

    @property
    def top1_agreement(self) -> float:
        if self.top1_comparisons == 0:
            return 1.0
        return self.top1_agreements / self.top1_comparisons

    @property
    def mean_kl_sequential_to_block(self) -> float:
        if not self.kl_sequential_to_block:
            return 0.0
        return sum(self.kl_sequential_to_block) / len(self.kl_sequential_to_block)

    @property
    def mean_top5_overlap(self) -> float:
        if self.top1_comparisons == 0:
            return 1.0
        return self.top5_overlap_fraction_sum / self.top1_comparisons


@dataclass(frozen=True)
class TransactionalGreedyTrace:
    token_ids: Tensor
    schedule: TargetScheduleResult


BlockObserver = Callable[[TargetScheduleBlock, C1ShadowKeyValueCache], None]
PostPrefillCallback = Callable[[C1ShadowKeyValueCache], None]


def _extract_logits(outputs: Any, *, expected_tokens: int) -> Tensor:
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise TypeError("model output must expose logits shaped [batch, tokens, vocab]")
    if tuple(logits.shape[:2]) != (1, int(expected_tokens)):
        raise ValueError(
            "model logits do not match the target schedule input: "
            f"{tuple(logits.shape)} vs [1, {expected_tokens}, vocab]"
        )
    return logits


def _forward(
    model: nn.Module, input_ids: Tensor, cache: C1ShadowKeyValueCache
) -> Tensor:
    outputs = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=0,
    )
    return _extract_logits(outputs, expected_tokens=int(input_ids.shape[1]))


def _validate_schedule_inputs(
    input_ids: Tensor,
    *,
    prefill_length: int,
    block_length: int,
    cache: C1ShadowKeyValueCache,
) -> None:
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("block-scheduled C1 evaluation supports batch_size=1 only")
    if input_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("block-scheduled C1 token IDs must use an integer dtype")
    sequence_length = int(input_ids.shape[1])
    if not 0 < int(prefill_length) < sequence_length:
        raise ValueError("prefill length must leave at least one evaluated token")
    if int(block_length) <= 0:
        raise ValueError("block length must be positive")
    if cache.runtime_mode != "idle" or cache.committed_length != 0:
        raise ValueError("block-scheduled C1 evaluation requires a fresh idle cache")


@torch.inference_mode()
def block_scheduled_teacher_forced_nll(
    model: nn.Module,
    input_ids: Tensor,
    *,
    cache: C1ShadowKeyValueCache,
    prefill_length: int,
    block_length: int,
    capture_prediction_logits: bool = False,
    block_observer: BlockObserver | None = None,
    post_prefill_callback: PostPrefillCallback | None = None,
) -> TargetScheduleResult:
    """Score fixed tokens while directly committing exact target KV in blocks."""

    _validate_schedule_inputs(
        input_ids,
        prefill_length=prefill_length,
        block_length=block_length,
        cache=cache,
    )
    prefill_length = int(prefill_length)
    block_length = int(block_length)
    cache.begin_target_prefill()
    try:
        prefill = input_ids[:, :prefill_length]
        with cache.forward_pass(query_length=prefill_length):
            prefill_logits = _forward(model, prefill, cache)
        cache.finish_target_prefill()
        if post_prefill_callback is not None:
            post_prefill_callback(cache)
    except BaseException:
        cache.abort_transaction()
        raise

    next_logits = prefill_logits[:, -1, :]
    token_nlls: list[float] = []
    top1_token_ids: list[int] = []
    top1_margins: list[float] = []
    captured_logits: list[Tensor] = []
    block_count = 0
    cursor = prefill_length
    sequence_length = int(input_ids.shape[1])
    while cursor < sequence_length:
        end = min(cursor + block_length, sequence_length)
        block_tokens = input_ids[:, cursor:end]
        query_length = end - cursor
        cache.begin_verify()
        try:
            with cache.forward_pass(query_length=query_length):
                verifier_logits = _forward(model, block_tokens, cache)
            aligned_logits = torch.cat(
                (next_logits.unsqueeze(1), verifier_logits[:, :-1, :]),
                dim=1,
            )
            if int(aligned_logits.shape[1]) != query_length:
                raise RuntimeError(
                    "target block logits are misaligned with fixed tokens"
                )
            labels = block_tokens.to(aligned_logits.device)
            losses = F.cross_entropy(
                aligned_logits.float().reshape(-1, aligned_logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            )
            if not bool(torch.isfinite(losses).all().cpu()):
                raise FloatingPointError("non-finite block-scheduled target NLL")
            top2 = aligned_logits.float().topk(k=2, dim=-1)
            token_nlls.extend(float(value) for value in losses.detach().cpu())
            top1_token_ids.extend(
                int(value) for value in top2.indices[..., 0].cpu().view(-1)
            )
            margins = top2.values[..., 0] - top2.values[..., 1]
            top1_margins.extend(float(value) for value in margins.cpu().view(-1))
            if capture_prediction_logits:
                captured_logits.append(aligned_logits.float().cpu())

            cache.commit_pending_target_prefix(query_length)
            observation = TargetScheduleBlock(
                block_index=block_count,
                relative_start=cursor - prefill_length,
                absolute_start=cursor,
                length=query_length,
                aligned_logits=aligned_logits,
            )
            if block_observer is not None:
                block_observer(observation, cache)
            next_logits = verifier_logits[:, -1, :]
        except BaseException:
            cache.abort_transaction()
            raise
        cursor = end
        block_count += 1

    evaluated_tokens = sequence_length - prefill_length
    if len(token_nlls) != evaluated_tokens or cache.committed_length != sequence_length:
        raise RuntimeError(
            "block-scheduled target accounting is inconsistent: "
            f"losses={len(token_nlls)}, cache={cache.committed_length}, "
            f"sequence={sequence_length}"
        )
    prediction_logits = (
        torch.cat(captured_logits, dim=1) if capture_prediction_logits else None
    )
    return TargetScheduleResult(
        block_length=block_length,
        prefill_tokens=prefill_length,
        evaluated_tokens=evaluated_tokens,
        block_count=block_count,
        nll_sum=sum(token_nlls),
        token_nlls=tuple(token_nlls),
        top1_token_ids=tuple(top1_token_ids),
        top1_margins=tuple(top1_margins),
        prediction_logits=prediction_logits,
    )


@torch.inference_mode()
def transactional_exact_c1_greedy_trace(
    model: nn.Module,
    input_ids: Tensor,
    *,
    cache: C1ShadowKeyValueCache,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> TransactionalGreedyTrace:
    """Generate one-token exact-C1 proposals and retain their logits and KV."""

    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("transactional exact C1 generation supports batch_size=1 only")
    if int(input_ids.shape[1]) <= 0:
        raise ValueError("transactional exact C1 generation requires a prompt")
    if int(max_new_tokens) <= 0:
        raise ValueError("transactional exact C1 trace requires generated tokens")
    if cache.runtime_mode != "idle" or cache.committed_length != 0:
        raise ValueError(
            "transactional exact C1 generation requires a fresh idle cache"
        )
    prompt_length = int(input_ids.shape[1])
    cache.begin_target_prefill()
    try:
        with cache.forward_pass(query_length=prompt_length):
            prompt_logits = _forward(model, input_ids, cache)
        cache.finish_target_prefill()
    except BaseException:
        cache.abort_transaction()
        raise

    next_logits = prompt_logits[:, -1, :]
    generated: list[int] = []
    token_nlls: list[float] = []
    top1_margins: list[float] = []
    prediction_logits: list[Tensor] = []
    while len(generated) < int(max_new_tokens):
        float_logits = next_logits.float()
        top2 = float_logits.topk(k=2, dim=-1)
        token = int(top2.indices[0, 0])
        generated.append(token)
        prediction_logits.append(float_logits.unsqueeze(1).cpu())
        label = torch.tensor([token], device=float_logits.device)
        token_nlls.append(float(F.cross_entropy(float_logits, label)))
        top1_margins.append(float(top2.values[0, 0] - top2.values[0, 1]))

        token_input = torch.tensor(
            [[token]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        cache.begin_verify()
        try:
            with cache.forward_pass(query_length=1):
                logits = _forward(model, token_input, cache)
            cache.commit_pending_target_prefix(1)
        except BaseException:
            cache.abort_transaction()
            raise
        next_logits = logits[:, -1, :]
        if eos_token_id is not None and token == int(eos_token_id):
            break

    token_ids = torch.cat(
        (
            input_ids,
            torch.tensor(
                [generated],
                dtype=input_ids.dtype,
                device=input_ids.device,
            ),
        ),
        dim=1,
    )
    if cache.committed_length != int(token_ids.shape[1]):
        raise RuntimeError("transactional greedy trace cache/output lengths diverged")
    return TransactionalGreedyTrace(
        token_ids=token_ids,
        schedule=TargetScheduleResult(
            block_length=1,
            prefill_tokens=prompt_length,
            evaluated_tokens=len(generated),
            block_count=len(generated),
            nll_sum=sum(token_nlls),
            token_nlls=tuple(token_nlls),
            top1_token_ids=tuple(generated),
            top1_margins=tuple(top1_margins),
            prediction_logits=torch.cat(prediction_logits, dim=1),
        ),
    )


def _tensor_drift(reference: Tensor, observed: Tensor) -> tuple[float, float, float]:
    reference_float = reference.float().reshape(-1)
    observed_float = observed.float().reshape(-1)
    difference = observed_float - reference_float
    denominator = max(float(torch.linalg.vector_norm(reference_float)), 1e-12)
    relative_l2 = float(torch.linalg.vector_norm(difference)) / denominator
    maximum_absolute_error = (
        float(difference.abs().max()) if difference.numel() else 0.0
    )
    if reference_float.numel() == 0:
        cosine = 1.0
    elif not bool(reference_float.any()) and not bool(observed_float.any()):
        cosine = 1.0
    else:
        cosine = float(F.cosine_similarity(reference_float, observed_float, dim=0))
    return relative_l2, maximum_absolute_error, cosine


@torch.inference_mode()
def compare_exact_block_and_sequential_schedules(
    model: nn.Module,
    input_ids: Tensor,
    *,
    sequential_cache: C1ShadowKeyValueCache,
    block_cache: C1ShadowKeyValueCache,
    prefill_length: int,
    block_length: int,
    sequential_reference: TargetScheduleResult | None = None,
) -> ExactBlockScheduleComparison:
    """Compare exact-proposal block commit against one-token target execution."""

    sequential = sequential_reference
    if sequential is None:
        sequential = block_scheduled_teacher_forced_nll(
            model,
            input_ids,
            cache=sequential_cache,
            prefill_length=prefill_length,
            block_length=1,
            capture_prediction_logits=True,
        )
    elif (
        sequential.block_length != 1
        or sequential.prefill_tokens != int(prefill_length)
        or sequential.evaluated_tokens != int(input_ids.shape[1]) - int(prefill_length)
        or sequential_cache.committed_length != int(input_ids.shape[1])
    ):
        raise ValueError(
            "provided sequential reference is incompatible with comparison"
        )
    if sequential.prediction_logits is None:
        raise AssertionError("sequential schedule did not retain comparison logits")
    sequential_logits = sequential.prediction_logits
    cache_drift: list[CacheDriftRecord] = []
    kl_values: list[float] = []
    logit_relative_l2: list[float] = []
    logit_maximum_absolute_error: list[float] = []
    top1_agreements = 0
    top1_comparisons = 0
    top5_overlap_sum = 0.0
    first_top1_disagreement: int | None = None

    def observe(
        observation: TargetScheduleBlock,
        committed_block_cache: C1ShadowKeyValueCache,
    ) -> None:
        nonlocal top1_agreements, top1_comparisons, top5_overlap_sum
        nonlocal first_top1_disagreement
        start = observation.relative_start
        end = start + observation.length
        reference_logits = sequential_logits[:, start:end, :].to(
            observation.aligned_logits.device
        )
        observed_logits = observation.aligned_logits.float()
        reference_log_probabilities = torch.log_softmax(reference_logits, dim=-1)
        observed_log_probabilities = torch.log_softmax(observed_logits, dim=-1)
        probabilities = reference_log_probabilities.exp()
        token_kl = torch.sum(
            probabilities * (reference_log_probabilities - observed_log_probabilities),
            dim=-1,
        )
        kl_values.extend(float(value) for value in token_kl.cpu().view(-1))
        difference = observed_logits - reference_logits
        denominator = max(float(torch.linalg.vector_norm(reference_logits)), 1e-12)
        logit_relative_l2.append(
            float(torch.linalg.vector_norm(difference)) / denominator
        )
        logit_maximum_absolute_error.append(float(difference.abs().max()))

        reference_top1 = torch.argmax(reference_logits, dim=-1)
        observed_top1 = torch.argmax(observed_logits, dim=-1)
        agreement = reference_top1 == observed_top1
        top1_agreements += int(agreement.sum())
        top1_comparisons += observation.length
        if first_top1_disagreement is None and not bool(agreement.all()):
            local = int(torch.nonzero(~agreement, as_tuple=False)[0, 1])
            first_top1_disagreement = start + local
        topk = min(5, int(reference_logits.shape[-1]))
        reference_topk = reference_logits.topk(k=topk, dim=-1).indices
        observed_topk = observed_logits.topk(k=topk, dim=-1).indices
        overlap = (
            (reference_topk.unsqueeze(-1) == observed_topk.unsqueeze(-2))
            .any(dim=-1)
            .sum(dim=-1)
        )
        top5_overlap_sum += float((overlap.float() / topk).sum())

        absolute_start = observation.absolute_start
        absolute_end = absolute_start + observation.length
        for layer_index, (reference_layer, observed_layer) in enumerate(
            zip(
                sequential_cache.shadow_layers,
                committed_block_cache.shadow_layers,
                strict=True,
            )
        ):
            if (
                reference_layer.exact_key_committed is None
                or observed_layer.exact_key_committed is None
                or reference_layer.c1_value_committed is None
                or observed_layer.c1_value_committed is None
            ):
                raise RuntimeError(
                    "exact schedule comparison found incomplete committed KV"
                )
            reference_key = reference_layer.exact_key_committed[
                ..., absolute_start:absolute_end, :
            ]
            observed_key = observed_layer.exact_key_committed[
                ..., absolute_start:absolute_end, :
            ]
            reference_value = reference_layer.c1_value_committed[
                ..., absolute_start:absolute_end, :
            ]
            observed_value = observed_layer.c1_value_committed[
                ..., absolute_start:absolute_end, :
            ]
            key_metrics = _tensor_drift(reference_key, observed_key)
            value_metrics = _tensor_drift(reference_value, observed_value)
            cache_drift.append(
                CacheDriftRecord(
                    block_index=observation.block_index,
                    layer_index=layer_index,
                    relative_start=start,
                    length=observation.length,
                    key_relative_l2=key_metrics[0],
                    key_maximum_absolute_error=key_metrics[1],
                    key_cosine_similarity=key_metrics[2],
                    c1_value_relative_l2=value_metrics[0],
                    c1_value_maximum_absolute_error=value_metrics[1],
                    c1_value_cosine_similarity=value_metrics[2],
                )
            )

    block = block_scheduled_teacher_forced_nll(
        model,
        input_ids,
        cache=block_cache,
        prefill_length=prefill_length,
        block_length=block_length,
        block_observer=observe,
    )
    if sequential.evaluated_tokens != block.evaluated_tokens:
        raise RuntimeError(
            "sequential and block schedules scored different token counts"
        )
    labels = input_ids[0, int(prefill_length) :].detach().cpu()
    sequential_label_matches = sum(
        int(prediction == int(label))
        for prediction, label in zip(sequential.top1_token_ids, labels, strict=True)
    )
    block_label_matches = sum(
        int(prediction == int(label))
        for prediction, label in zip(block.top1_token_ids, labels, strict=True)
    )
    return ExactBlockScheduleComparison(
        block_length=int(block_length),
        evaluated_tokens=block.evaluated_tokens,
        block_count=block.block_count,
        sequential_nll_sum=sequential.nll_sum,
        block_nll_sum=block.nll_sum,
        sequential_token_nlls=sequential.token_nlls,
        block_token_nlls=block.token_nlls,
        sequential_top1_margins=sequential.top1_margins,
        block_top1_margins=block.top1_margins,
        top1_agreements=top1_agreements,
        top1_comparisons=top1_comparisons,
        first_top1_disagreement=first_top1_disagreement,
        sequential_label_top1_matches=sequential_label_matches,
        block_label_top1_matches=block_label_matches,
        kl_sequential_to_block=tuple(kl_values),
        logit_relative_l2_by_block=tuple(logit_relative_l2),
        logit_maximum_absolute_error_by_block=tuple(logit_maximum_absolute_error),
        top5_overlap_fraction_sum=top5_overlap_sum,
        cache_drift=tuple(cache_drift),
    )


__all__ = [
    "CacheDriftRecord",
    "ExactBlockScheduleComparison",
    "TargetScheduleBlock",
    "TargetScheduleResult",
    "TransactionalGreedyTrace",
    "block_scheduled_teacher_forced_nll",
    "compare_exact_block_and_sequential_schedules",
    "transactional_exact_c1_greedy_trace",
]
