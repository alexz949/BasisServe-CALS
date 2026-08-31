"""Streaming sufficient statistics for the shared S80 GQA latent.

S80 uses one group encoder for the joint token feature ``X = [V, K_post]``.
Payload fitting consumes the routed features ``P @ X`` while routing fitting
consumes paired query and token Grams.  Routing shards deliberately remain
paired: multiplying independently aggregated Q and X Grams would introduce
cross-document score terms that are absent from the calibration objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from basisserve.calibration.gqa_routed_ov_stats import (
    flatten_covariance_blocks,
)


def _floating_statistics_dtype(dtype: torch.dtype) -> None:
    return None


def joint_token_features(
    value_states: torch.Tensor,
    post_rope_key_states: torch.Tensor,
) -> torch.Tensor:
    """Return ``[V, K_post]`` without changing the leading tensor layout."""

    return torch.cat((value_states, post_rope_key_states), dim=-1)


@dataclass(frozen=True)
class S80PayloadStatistics:
    covariance_blocks: torch.Tensor
    row_count: int
    dense_output_energy: float
    value_dim: int
    key_dim: int

    @property
    def num_query_heads(self) -> int:
        return int(self.covariance_blocks.shape[0])

    @property
    def joint_dim(self) -> int:
        return int(self.covariance_blocks.shape[2])

    def flat_covariance(self) -> torch.Tensor:
        return flatten_covariance_blocks(self.covariance_blocks)

    def validate(self) -> None:
        return None


@dataclass(frozen=True)
class S80DirectResidualData:
    """Paired routing operands used only by the joint BF update."""

    routing_queries: torch.Tensor
    routing_joint_rows: torch.Tensor

    def validate(
        self,
        *,
        num_query_heads: int,
        num_kv_heads: int,
        key_dim: int,
        joint_dim: int,
    ) -> None:
        return None


class S80PayloadAccumulator:
    """Accumulate the full covariance of headwise ``[P V, P K_post]`` rows."""

    def __init__(
        self,
        *,
        num_query_heads: int,
        value_dim: int,
        key_dim: int,
        accumulation_dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        _floating_statistics_dtype(accumulation_dtype)
        self.num_query_heads = int(num_query_heads)
        self.value_dim = int(value_dim)
        self.key_dim = int(key_dim)
        self.joint_dim = self.value_dim + self.key_dim
        self.width = self.num_query_heads * self.joint_dim
        self.accumulation_dtype = accumulation_dtype
        self.device = None if device is None else torch.device(device)
        self._gram: torch.Tensor | None = None
        self._output_energy: torch.Tensor | None = None
        self._rows = 0

    @torch.no_grad()
    def update(
        self,
        routed_value: torch.Tensor,
        routed_post_rope_key: torch.Tensor,
        dense_output: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        joint = joint_token_features(routed_value, routed_post_rope_key)
        rows = joint.reshape(-1, self.width)
        outputs = dense_output.detach().reshape(-1, dense_output.shape[-1])
        if valid_mask is not None:
            selected = valid_mask.detach().reshape(-1).to(dtype=torch.bool)
            rows = rows[selected]
            outputs = outputs[selected]
        if rows.shape[0] == 0:
            return
        target_device = self.device or rows.device
        rows = rows.detach().to(target_device, self.accumulation_dtype)
        outputs = outputs.detach().to(target_device, self.accumulation_dtype)
        if self._gram is None:
            self._gram = torch.zeros(
                self.width,
                self.width,
                device=target_device,
                dtype=self.accumulation_dtype,
            )
            self._output_energy = torch.zeros(
                (), device=target_device, dtype=self.accumulation_dtype
            )
        self._gram.addmm_(rows.mT, rows)
        self._output_energy.add_(outputs.square().sum())
        self._rows += int(rows.shape[0])

    @property
    def rows(self) -> int:
        return self._rows

    def finalize(
        self,
        *,
        output_dtype: torch.dtype = torch.float64,
        output_device: torch.device | str = "cpu",
    ) -> S80PayloadStatistics:
        _floating_statistics_dtype(output_dtype)
        covariance = self._gram / self._rows
        covariance = 0.5 * (covariance + covariance.mT)
        blocks = (
            covariance.reshape(
                self.num_query_heads,
                self.joint_dim,
                self.num_query_heads,
                self.joint_dim,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
            .to(device=output_device, dtype=output_dtype)
        )
        result = S80PayloadStatistics(
            covariance_blocks=blocks,
            row_count=self._rows,
            dense_output_energy=float(self._output_energy.double() / self._rows),
            value_dim=self.value_dim,
            key_dim=self.key_dim,
        )
        result.validate()
        return result


@dataclass(frozen=True)
class S80RoutingShard:
    query_grams: torch.Tensor
    joint_grams: torch.Tensor
    query_sums: torch.Tensor
    joint_sums: torch.Tensor
    query_row_counts: torch.Tensor
    key_row_counts: torch.Tensor
    target_score_energy: float
    metadata: Mapping[str, Any]

    def validate(self, head_to_kv_group: torch.Tensor | Sequence[int]) -> None:
        return None

    @property
    def num_query_heads(self) -> int:
        return int(self.query_grams.shape[0])

    @property
    def num_kv_heads(self) -> int:
        return int(self.joint_grams.shape[0])

    @property
    def key_dim(self) -> int:
        return int(self.query_grams.shape[-1])

    @property
    def joint_dim(self) -> int:
        return int(self.joint_grams.shape[-1])


@dataclass(frozen=True)
class S80RoutingStatistics:
    shards: tuple[S80RoutingShard, ...]
    head_to_kv_group: torch.Tensor
    value_dim: int
    key_dim: int

    @property
    def target_score_energy(self) -> float:
        return sum(float(shard.target_score_energy) for shard in self.shards)

    @property
    def num_query_heads(self) -> int:
        return int(self.head_to_kv_group.numel())

    @property
    def num_kv_heads(self) -> int:
        return int(self.head_to_kv_group.max().item()) + 1

    @property
    def joint_dim(self) -> int:
        return self.value_dim + self.key_dim

    def validate(self) -> None:
        return None


class S80RoutingAccumulator:
    """Retain paired causal-shard Grams for the S80 routing objective."""

    def __init__(
        self,
        *,
        head_to_kv_group: torch.Tensor | Sequence[int],
        value_dim: int,
        key_dim: int,
        storage_dtype: torch.dtype = torch.float64,
        storage_device: torch.device | str = "cpu",
    ) -> None:
        _floating_statistics_dtype(storage_dtype)
        mapping = torch.as_tensor(head_to_kv_group, dtype=torch.long).clone()
        self.head_to_kv_group = mapping
        self.value_dim = int(value_dim)
        self.key_dim = int(key_dim)
        self.storage_dtype = storage_dtype
        self.storage_device = torch.device(storage_device)
        self._shards: list[S80RoutingShard] = []

    @torch.no_grad()
    def update_shard(
        self,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        post_rope_key_states: torch.Tensor,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Add one shard whose query rows share exactly the supplied key range.

        Inputs use ``[rows, heads/groups, channels]`` layout.  Callers must
        split causal queries with different visible prefixes into distinct
        calls; this method never merges independently accumulated Grams.
        """

        heads = self.head_to_kv_group.numel()
        groups = int(self.head_to_kv_group.max()) + 1
        joint = joint_token_features(value_states, post_rope_key_states)
        query = query_states.detach().permute(1, 0, 2).to(self.storage_dtype)
        by_group = joint.detach().permute(1, 0, 2).to(self.storage_dtype)
        query_grams = torch.bmm(query.mT, query)
        joint_grams = torch.bmm(by_group.mT, by_group)
        target_energy = query_grams.new_zeros(())
        for head, group in enumerate(self.head_to_kv_group.tolist()):
            key_gram = joint_grams[group, self.value_dim :, self.value_dim :]
            target_energy.add_(torch.sum(query_grams[head] * key_gram.mT))
        shard = S80RoutingShard(
            query_grams=query_grams.to(self.storage_device, self.storage_dtype),
            joint_grams=joint_grams.to(self.storage_device, self.storage_dtype),
            query_sums=query.sum(dim=1).to(
                self.storage_device,
                self.storage_dtype,
            ),
            joint_sums=by_group.sum(dim=1).to(
                self.storage_device,
                self.storage_dtype,
            ),
            query_row_counts=torch.full(
                (heads,),
                int(query_states.shape[0]),
                dtype=torch.int64,
                device=self.storage_device,
            ),
            key_row_counts=torch.full(
                (groups,),
                int(value_states.shape[0]),
                dtype=torch.int64,
                device=self.storage_device,
            ),
            target_score_energy=float(target_energy),
            metadata=dict(metadata or {}),
        )
        shard.validate(self.head_to_kv_group)
        self._shards.append(shard)

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    def finalize(self) -> S80RoutingStatistics:
        result = S80RoutingStatistics(
            shards=tuple(self._shards),
            head_to_kv_group=self.head_to_kv_group.clone(),
            value_dim=self.value_dim,
            key_dim=self.key_dim,
        )
        result.validate()
        return result


def routing_proxy_score_statistics(
    statistics: S80RoutingStatistics,
    effective_maps: torch.Tensor,
    *,
    scaling: float,
) -> dict[str, torch.Tensor]:
    """Return exact first and second moments implied by paired routing shards."""

    device = effective_maps.device
    dtype = effective_maps.dtype
    mapping = statistics.head_to_kv_group.to(device=device)
    heads = int(mapping.numel())
    score_sums = torch.zeros(heads, device=device, dtype=dtype)
    score_square_sums = torch.zeros_like(score_sums)
    score_counts = torch.zeros(heads, device=device, dtype=torch.int64)
    for shard in statistics.shards:
        query_grams = shard.query_grams.to(device=device, dtype=dtype)
        joint_grams = shard.joint_grams.to(device=device, dtype=dtype)
        query_sums = shard.query_sums.to(device=device, dtype=dtype)
        joint_sums = shard.joint_sums.to(device=device, dtype=dtype)
        query_counts = shard.query_row_counts.to(device=device)
        key_counts = shard.key_row_counts.to(device=device)
        for head, group in enumerate(mapping.tolist()):
            effective = effective_maps[head]
            score_sums[head].add_(query_sums[head] @ effective @ joint_sums[group])
            score_square_sums[head].add_(
                torch.sum(
                    (query_grams[head] @ effective @ joint_grams[group]) * effective
                )
            )
            score_counts[head].add_(query_counts[head] * key_counts[group])
    counts = score_counts.to(dtype=dtype)
    raw_mean = score_sums / counts
    raw_variance = torch.clamp(
        score_square_sums / counts - raw_mean.square(),
        min=0,
    )
    total_count = score_counts.sum()
    total_count_float = total_count.to(dtype=dtype)
    aggregate_mean = score_sums.sum() / total_count_float
    aggregate_variance = torch.clamp(
        score_square_sums.sum() / total_count_float - aggregate_mean.square(),
        min=0,
    )
    scale = torch.as_tensor(scaling, device=device, dtype=dtype)
    return {
        "score_counts_by_head": score_counts,
        "raw_mean_by_head": raw_mean,
        "raw_variance_by_head": raw_variance,
        "scaled_mean_by_head": raw_mean * scale,
        "scaled_variance_by_head": raw_variance * scale.square(),
        "score_count": total_count,
        "raw_mean": aggregate_mean,
        "raw_variance": aggregate_variance,
        "scaled_mean": aggregate_mean * scale,
        "scaled_variance": aggregate_variance * scale.square(),
    }


__all__ = [
    "S80DirectResidualData",
    "S80PayloadAccumulator",
    "S80PayloadStatistics",
    "S80RoutingAccumulator",
    "S80RoutingShard",
    "S80RoutingStatistics",
    "joint_token_features",
    "routing_proxy_score_statistics",
]
