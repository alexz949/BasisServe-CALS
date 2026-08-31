"""Portable low-rank KV metadata used by BASISServe integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LowRankKVConfig:
    """Shape contract for a low-rank latent KV cache.

    This is intentionally framework-neutral. SGLang-specific config parsing
    should map into this object, not become the portable BASISServe API.
    """

    format: str
    rank_k: int
    rank_v: int
    num_kv_groups: int
    q_heads_per_kv_group: int
    full_head_dim: int
    dtype: str = "bfloat16"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LowRankKVConfig":
        return cls(
            format=str(data["format"]),
            rank_k=int(data["rank_k"]),
            rank_v=int(data["rank_v"]),
            num_kv_groups=int(data["num_kv_groups"]),
            q_heads_per_kv_group=int(data["q_heads_per_kv_group"]),
            full_head_dim=int(data["full_head_dim"]),
            dtype=str(data.get("dtype", "bfloat16")),
        )

    def __post_init__(self) -> None:
        fields = {
            "rank_k": self.rank_k,
            "rank_v": self.rank_v,
            "num_kv_groups": self.num_kv_groups,
            "q_heads_per_kv_group": self.q_heads_per_kv_group,
            "full_head_dim": self.full_head_dim,
        }
        for name, value in fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if self.rank_k > self.full_head_dim:
            raise ValueError("rank_k cannot exceed full_head_dim")
        if self.rank_v > self.full_head_dim:
            raise ValueError("rank_v cannot exceed full_head_dim")

    @property
    def latent_k_width(self) -> int:
        return self.num_kv_groups * self.rank_k

    @property
    def latent_v_width(self) -> int:
        return self.num_kv_groups * self.rank_v

    @property
    def full_kv_width(self) -> int:
        return self.num_kv_groups * self.full_head_dim * 2

    @property
    def latent_kv_width(self) -> int:
        return self.latent_k_width + self.latent_v_width

    @property
    def compression_ratio_vs_full_kv(self) -> float:
        return self.latent_kv_width / self.full_kv_width
