"""Greedy shadow-Key speculative verification for a deployed C1 model.

``strict_replay`` preserves ordinary one-token BF16 C1 greedy semantics.
``direct_block`` instead commits accepted exact block-verifier KV and performs
an additional one-token target forward only for a correction token.  The
second policy is mathematically the same C1 model but has block-scheduled BF16
execution semantics and is intentionally not described as lossless.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from types import SimpleNamespace
from typing import Any, Literal

import torch
from torch import Tensor, nn

from basisserve.core.c1_shadow_kv import C1ShadowKeyValueCache


CommitPolicy = Literal["strict_replay", "direct_block"]


@dataclass(frozen=True)
class BlockCommitConfig:
    policy: CommitPolicy = "strict_replay"

    def __post_init__(self) -> None:
        if self.policy not in ("strict_replay", "direct_block"):
            raise ValueError(f"unsupported block commit policy {self.policy!r}")


@dataclass(frozen=True)
class GreedyVerificationResult:
    accepted_length: int
    proposed_length: int
    full_block_accepted: bool
    first_rejection_index: int | None
    emitted_length: int
    forced_target_seed_length: int
    shadow_continuation_accepted: int
    directly_committed_length: int
    sequentially_committed_length: int


@dataclass(frozen=True)
class C1SpeculativeDecodeMetrics:
    commit_policy: CommitPolicy
    generated_target_tokens: int
    target_verification_calls: int
    target_commit_sync_calls: int
    target_accepted_replay_calls: int
    target_correction_sync_calls: int
    directly_committed_target_tokens: int
    sequentially_committed_target_tokens: int
    draft_forward_calls: int
    rollback_count: int
    rejected_suffix_tokens: int
    verifier_top1_agreements: int
    verifier_top1_comparisons: int
    shadow_logit_kl_sum: float
    shadow_logit_kl_comparisons: int
    block_sequential_top1_agreements: int
    block_sequential_top1_comparisons: int
    block_sequential_logit_kl_sum: float
    block_sequential_logit_kl_comparisons: int
    draft_seconds: float
    verification_seconds: float
    commit_sync_seconds: float
    correction_sync_seconds: float
    cache_storage: dict[str, int]

    @property
    def verifier_top1_agreement(self) -> float:
        if self.verifier_top1_comparisons == 0:
            return 1.0
        return self.verifier_top1_agreements / self.verifier_top1_comparisons

    @property
    def mean_shadow_logit_kl(self) -> float | None:
        if self.shadow_logit_kl_comparisons == 0:
            return None
        return self.shadow_logit_kl_sum / self.shadow_logit_kl_comparisons

    @property
    def block_sequential_top1_agreement(self) -> float | None:
        if self.block_sequential_top1_comparisons == 0:
            return None
        return (
            self.block_sequential_top1_agreements
            / self.block_sequential_top1_comparisons
        )

    @property
    def mean_block_sequential_logit_kl(self) -> float | None:
        if self.block_sequential_logit_kl_comparisons == 0:
            return None
        return (
            self.block_sequential_logit_kl_sum
            / self.block_sequential_logit_kl_comparisons
        )


@dataclass(frozen=True)
class C1SpeculativeDecodeResult:
    commit_policy: CommitPolicy
    token_ids: Tensor
    generated_token_ids: tuple[int, ...]
    rounds: tuple[GreedyVerificationResult, ...]
    metrics: C1SpeculativeDecodeMetrics


def _validate_generation_inputs(
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    draft_length: int,
    attention_mask: Tensor | None,
) -> None:
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("C1 speculative decoding supports batch_size=1 only")
    if int(input_ids.shape[1]) <= 0:
        raise ValueError("C1 speculative decoding requires a non-empty prompt")
    if input_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("input token IDs must use an integer dtype")
    if int(max_new_tokens) < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if int(draft_length) <= 0:
        raise ValueError("draft_length must be positive")
    if attention_mask is not None:
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError("attention mask must match the batch-one prompt shape")
        if not bool(torch.all(attention_mask != 0)):
            raise NotImplementedError(
                "padded prompts are not supported by the first oracle"
            )


def _extract_logits(outputs: Any, *, expected_tokens: int) -> Tensor:
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise TypeError("model output must expose logits shaped [batch, tokens, vocab]")
    if int(logits.shape[0]) != 1 or int(logits.shape[1]) != int(expected_tokens):
        raise ValueError(
            "model logits do not match the requested batch/token shape: "
            f"{tuple(logits.shape)} vs [1, {expected_tokens}, vocab]"
        )
    return logits


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_forward(
    model: nn.Module,
    input_ids: Tensor,
    *,
    past_key_values: Any,
) -> tuple[Tensor, float]:
    device = input_ids.device
    _synchronize(device)
    started = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=0,
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return _extract_logits(outputs, expected_tokens=int(input_ids.shape[1])), elapsed


def _target_to_draft_kl(target_logits: Tensor, draft_logits: Tensor) -> float:
    target_log_probabilities = torch.log_softmax(target_logits.float(), dim=-1)
    draft_log_probabilities = torch.log_softmax(draft_logits.float(), dim=-1)
    probabilities = target_log_probabilities.exp()
    return float(
        torch.sum(
            probabilities * (target_log_probabilities - draft_log_probabilities),
            dim=-1,
        ).mean()
    )


@torch.inference_mode()
def c1_shadow_greedy_decode(
    model: nn.Module,
    input_ids: Tensor,
    *,
    cache: C1ShadowKeyValueCache,
    max_new_tokens: int,
    draft_length: int,
    block_commit: BlockCommitConfig = BlockCommitConfig(),
    eos_token_id: int | None = None,
    attention_mask: Tensor | None = None,
) -> C1SpeculativeDecodeResult:
    """Generate greedy C1 tokens through shadow-Key proposals.

    The first proposal in every round is seeded by the carried exact-target
    logits.  Acceptance statistics expose this forced seed separately from the
    subsequent shadow-generated continuation. ``strict_replay`` is the exact
    sequential reference; ``direct_block`` uses block-scheduled target state.
    """

    _validate_generation_inputs(
        input_ids,
        max_new_tokens=max_new_tokens,
        draft_length=draft_length,
        attention_mask=attention_mask,
    )
    if cache.committed_length != 0 or cache.runtime_mode != "idle":
        raise ValueError("C1 speculative decoding requires a fresh, idle cache")
    if max_new_tokens == 0:
        return C1SpeculativeDecodeResult(
            commit_policy=block_commit.policy,
            token_ids=input_ids.detach().clone(),
            generated_token_ids=(),
            rounds=(),
            metrics=C1SpeculativeDecodeMetrics(
                commit_policy=block_commit.policy,
                generated_target_tokens=0,
                target_verification_calls=0,
                target_commit_sync_calls=0,
                target_accepted_replay_calls=0,
                target_correction_sync_calls=0,
                directly_committed_target_tokens=0,
                sequentially_committed_target_tokens=0,
                draft_forward_calls=0,
                rollback_count=0,
                rejected_suffix_tokens=0,
                verifier_top1_agreements=0,
                verifier_top1_comparisons=0,
                shadow_logit_kl_sum=0.0,
                shadow_logit_kl_comparisons=0,
                block_sequential_top1_agreements=0,
                block_sequential_top1_comparisons=0,
                block_sequential_logit_kl_sum=0.0,
                block_sequential_logit_kl_comparisons=0,
                draft_seconds=0.0,
                verification_seconds=0.0,
                commit_sync_seconds=0.0,
                correction_sync_seconds=0.0,
                cache_storage=cache.storage_summary(),
            ),
        )

    cache.begin_target_prefill()
    try:
        with cache.forward_pass(query_length=int(input_ids.shape[1])):
            prefill_logits, _ = _timed_forward(
                model,
                input_ids,
                past_key_values=cache,
            )
        cache.finish_target_prefill()
    except BaseException:
        cache.abort_transaction()
        raise

    next_target_logits = prefill_logits[:, -1, :]
    generated: list[int] = []
    round_results: list[GreedyVerificationResult] = []
    target_verification_calls = 0
    target_commit_sync_calls = 0
    target_accepted_replay_calls = 0
    target_correction_sync_calls = 0
    directly_committed_target_tokens = 0
    sequentially_committed_target_tokens = 0
    draft_forward_calls = 0
    rollback_count = 0
    rejected_suffix_tokens = 0
    verifier_top1_agreements = 0
    verifier_top1_comparisons = 0
    shadow_logit_kl_sum = 0.0
    shadow_logit_kl_comparisons = 0
    block_sequential_top1_agreements = 0
    block_sequential_top1_comparisons = 0
    block_sequential_logit_kl_sum = 0.0
    block_sequential_logit_kl_comparisons = 0
    draft_seconds = 0.0
    verification_seconds = 0.0
    commit_sync_seconds = 0.0
    correction_sync_seconds = 0.0

    while len(generated) < max_new_tokens:
        requested = min(int(draft_length), max_new_tokens - len(generated))
        forced_seed = int(torch.argmax(next_target_logits, dim=-1).item())
        proposals = [forced_seed]
        draft_prediction_logits: list[Tensor | None] = [None]
        cache.begin_draft()
        try:
            while len(proposals) < requested and (
                eos_token_id is None or proposals[-1] != int(eos_token_id)
            ):
                draft_input = torch.tensor(
                    [[proposals[-1]]],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                with cache.forward_pass(query_length=1):
                    logits, elapsed = _timed_forward(
                        model,
                        draft_input,
                        past_key_values=cache,
                    )
                draft_seconds += elapsed
                draft_forward_calls += 1
                prediction_logits = logits[:, -1, :]
                draft_prediction_logits.append(prediction_logits)
                proposals.append(int(torch.argmax(prediction_logits, dim=-1).item()))

            proposal_tensor = torch.tensor(
                [proposals],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            cache.begin_verify()
            with cache.forward_pass(query_length=len(proposals)):
                verify_logits, elapsed = _timed_forward(
                    model,
                    proposal_tensor,
                    past_key_values=cache,
                )
            verification_seconds += elapsed
            target_verification_calls += 1

            aligned_block_logits = torch.cat(
                (
                    next_target_logits.unsqueeze(1),
                    verify_logits[:, :-1, :],
                ),
                dim=1,
            )
            emitted: list[int] = []
            accepted_length = 0
            rejection_index: int | None = None
            round_directly_committed = 0
            round_sequentially_committed = 0
            if block_commit.policy == "strict_replay":
                # Pending block KV are diagnostic in strict mode. Sequential
                # target forwards decide acceptance and commit every emitted token.
                cache.commit_pending_target_prefix(0)
                for index, proposal in enumerate(proposals):
                    sequential_logits = next_target_logits
                    target_token = int(torch.argmax(sequential_logits, dim=-1).item())
                    block_logits = aligned_block_logits[:, index, :]
                    block_token = int(torch.argmax(block_logits, dim=-1).item())
                    block_sequential_top1_agreements += int(block_token == target_token)
                    block_sequential_top1_comparisons += 1
                    block_sequential_logit_kl_sum += _target_to_draft_kl(
                        sequential_logits,
                        block_logits,
                    )
                    block_sequential_logit_kl_comparisons += 1

                    if index > 0:
                        draft_logits = draft_prediction_logits[index]
                        if draft_logits is None:
                            raise AssertionError(
                                "shadow proposal is missing its draft logits"
                            )
                        verifier_top1_agreements += int(proposal == target_token)
                        verifier_top1_comparisons += 1
                        shadow_logit_kl_sum += _target_to_draft_kl(
                            sequential_logits,
                            draft_logits,
                        )
                        shadow_logit_kl_comparisons += 1

                    if proposal == target_token:
                        accepted_length += 1
                        emitted.append(proposal)
                        committed_token = proposal
                        target_accepted_replay_calls += 1
                    else:
                        rejection_index = index
                        emitted.append(target_token)
                        committed_token = target_token
                        rejected_suffix_tokens += len(proposals) - accepted_length
                        rollback_count += 1

                    commit_input = torch.tensor(
                        [[committed_token]],
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                    cache.begin_verify()
                    with cache.forward_pass(query_length=1):
                        commit_logits, elapsed = _timed_forward(
                            model,
                            commit_input,
                            past_key_values=cache,
                        )
                    cache.commit_pending_target_prefix(1)
                    next_target_logits = commit_logits[:, -1, :]
                    commit_sync_seconds += elapsed
                    target_commit_sync_calls += 1
                    sequentially_committed_target_tokens += 1
                    round_sequentially_committed += 1
                    if rejection_index is not None:
                        correction_sync_seconds += elapsed
                        target_correction_sync_calls += 1
                        break
            else:
                correction_token: int | None = None
                for index, proposal in enumerate(proposals):
                    block_logits = aligned_block_logits[:, index, :]
                    target_token = int(torch.argmax(block_logits, dim=-1).item())
                    if index > 0:
                        draft_logits = draft_prediction_logits[index]
                        if draft_logits is None:
                            raise AssertionError(
                                "shadow proposal is missing its draft logits"
                            )
                        verifier_top1_agreements += int(proposal == target_token)
                        verifier_top1_comparisons += 1
                        shadow_logit_kl_sum += _target_to_draft_kl(
                            block_logits,
                            draft_logits,
                        )
                        shadow_logit_kl_comparisons += 1
                    if proposal == target_token:
                        accepted_length += 1
                        continue
                    rejection_index = index
                    correction_token = target_token
                    rejected_suffix_tokens += len(proposals) - accepted_length
                    rollback_count += 1
                    break

                emitted.extend(proposals[:accepted_length])
                cache.commit_pending_target_prefix(accepted_length)
                directly_committed_target_tokens += accepted_length
                round_directly_committed = accepted_length
                if rejection_index is None:
                    next_target_logits = verify_logits[:, -1, :]
                else:
                    if correction_token is None:
                        raise AssertionError(
                            "rejected block is missing a correction token"
                        )
                    emitted.append(correction_token)
                    correction_input = torch.tensor(
                        [[correction_token]],
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                    cache.begin_verify()
                    with cache.forward_pass(query_length=1):
                        correction_logits, elapsed = _timed_forward(
                            model,
                            correction_input,
                            past_key_values=cache,
                        )
                    cache.commit_pending_target_prefix(1)
                    next_target_logits = correction_logits[:, -1, :]
                    commit_sync_seconds += elapsed
                    correction_sync_seconds += elapsed
                    target_commit_sync_calls += 1
                    target_correction_sync_calls += 1
                    sequentially_committed_target_tokens += 1
                    round_sequentially_committed = 1

            full_block_accepted = rejection_index is None
            expected_emitted = accepted_length + (0 if full_block_accepted else 1)
            if len(emitted) != expected_emitted:
                raise RuntimeError(
                    f"emitted length {len(emitted)} disagrees with accepted/correction "
                    f"accounting {expected_emitted}"
                )
            expected_cache_length = (
                int(input_ids.shape[1]) + len(generated) + len(emitted)
            )
            if cache.committed_length != expected_cache_length:
                raise RuntimeError(
                    "C1 cache/output lengths diverged within a speculative round: "
                    f"cache={cache.committed_length}, output={expected_cache_length}"
                )

            generated.extend(emitted)
            round_results.append(
                GreedyVerificationResult(
                    accepted_length=accepted_length,
                    proposed_length=len(proposals),
                    full_block_accepted=full_block_accepted,
                    first_rejection_index=rejection_index,
                    emitted_length=len(emitted),
                    forced_target_seed_length=1,
                    shadow_continuation_accepted=max(accepted_length - 1, 0),
                    directly_committed_length=round_directly_committed,
                    sequentially_committed_length=round_sequentially_committed,
                )
            )
        except BaseException:
            cache.abort_transaction()
            raise

        if eos_token_id is not None and int(eos_token_id) in emitted:
            eos_index = emitted.index(int(eos_token_id))
            if eos_index != len(emitted) - 1:
                raise RuntimeError(
                    "EOS appeared before the end of an emitted speculative block"
                )
            break

    if cache.committed_length != int(input_ids.shape[1]) + len(generated):
        raise RuntimeError(
            "C1 cache/output lengths diverged after decoding: "
            f"cache={cache.committed_length}, output={input_ids.shape[1] + len(generated)}"
        )
    generated_tensor = torch.tensor(
        [generated],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    token_ids = torch.cat((input_ids, generated_tensor), dim=1)
    return C1SpeculativeDecodeResult(
        commit_policy=block_commit.policy,
        token_ids=token_ids,
        generated_token_ids=tuple(generated),
        rounds=tuple(round_results),
        metrics=C1SpeculativeDecodeMetrics(
            commit_policy=block_commit.policy,
            generated_target_tokens=len(generated),
            target_verification_calls=target_verification_calls,
            target_commit_sync_calls=target_commit_sync_calls,
            target_accepted_replay_calls=target_accepted_replay_calls,
            target_correction_sync_calls=target_correction_sync_calls,
            directly_committed_target_tokens=directly_committed_target_tokens,
            sequentially_committed_target_tokens=sequentially_committed_target_tokens,
            draft_forward_calls=draft_forward_calls,
            rollback_count=rollback_count,
            rejected_suffix_tokens=rejected_suffix_tokens,
            verifier_top1_agreements=verifier_top1_agreements,
            verifier_top1_comparisons=verifier_top1_comparisons,
            shadow_logit_kl_sum=shadow_logit_kl_sum,
            shadow_logit_kl_comparisons=shadow_logit_kl_comparisons,
            block_sequential_top1_agreements=block_sequential_top1_agreements,
            block_sequential_top1_comparisons=block_sequential_top1_comparisons,
            block_sequential_logit_kl_sum=block_sequential_logit_kl_sum,
            block_sequential_logit_kl_comparisons=(
                block_sequential_logit_kl_comparisons
            ),
            draft_seconds=draft_seconds,
            verification_seconds=verification_seconds,
            commit_sync_seconds=commit_sync_seconds,
            correction_sync_seconds=correction_sync_seconds,
            cache_storage=cache.storage_summary(),
        ),
    )


@torch.inference_mode()
def exact_c1_greedy_decode(
    model: nn.Module,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Independent ordinary exact-cache greedy baseline."""

    _validate_generation_inputs(
        input_ids,
        max_new_tokens=max_new_tokens,
        draft_length=1,
        attention_mask=attention_mask,
    )
    if max_new_tokens == 0:
        return input_ids.detach().clone()
    from transformers.cache_utils import DynamicCache

    model_config = getattr(model, "config", None)
    cache = DynamicCache(config=model_config)
    logits, _ = _timed_forward(model, input_ids, past_key_values=cache)
    next_logits = logits[:, -1, :]
    generated: list[int] = []
    while len(generated) < max_new_tokens:
        token = int(torch.argmax(next_logits, dim=-1).item())
        generated.append(token)
        if eos_token_id is not None and token == int(eos_token_id):
            break
        if len(generated) == max_new_tokens:
            break
        token_input = torch.tensor(
            [[token]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        logits, _ = _timed_forward(model, token_input, past_key_values=cache)
        next_logits = logits[:, -1, :]
    return torch.cat(
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


@torch.inference_mode()
def transactional_exact_c1_greedy_decode(
    model: nn.Module,
    input_ids: Tensor,
    *,
    cache: C1ShadowKeyValueCache,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Sequential exact-Key C1 greedy decoding through transactional cache modes."""
    if max_new_tokens == 0:
        return input_ids.detach().clone()
    _validate_generation_inputs(
        input_ids,
        max_new_tokens=max_new_tokens,
        draft_length=1,
        attention_mask=attention_mask,
    )
    from basisserve.core.c1_block_commit import transactional_exact_c1_greedy_trace

    return transactional_exact_c1_greedy_trace(
        model,
        input_ids,
        cache=cache,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    ).token_ids


def result_as_namespace(result: C1SpeculativeDecodeResult) -> SimpleNamespace:
    """Small compatibility helper for scripts that prefer attribute payloads."""

    return SimpleNamespace(
        commit_policy=result.commit_policy,
        token_ids=result.token_ids,
        generated_token_ids=result.generated_token_ids,
        rounds=result.rounds,
        metrics=result.metrics,
    )


__all__ = [
    "BlockCommitConfig",
    "C1SpeculativeDecodeMetrics",
    "C1SpeculativeDecodeResult",
    "CommitPolicy",
    "GreedyVerificationResult",
    "c1_shadow_greedy_decode",
    "exact_c1_greedy_decode",
    "result_as_namespace",
    "transactional_exact_c1_greedy_decode",
]
