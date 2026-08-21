"""Attention-layout checks for MHA/GQA/MQA tensor parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


KVPartitionMode = Literal["sharded", "replicated"]


@dataclass(frozen=True)
class AttentionTPLayout:
    """Describe the standard head-parallel layout before ``o_proj``.

    The low-rank ``o_proj`` communication primitive depends on the query-head
    output shard, not on whether K/V use MHA, GQA, or MQA.  This object makes
    that distinction explicit and validates the common layouts used by serving
    runtimes.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    tp_size: int

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("tp_size", self.tp_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_attention_heads % self.tp_size != 0:
            raise ValueError(
                "head-parallel o_proj requires num_attention_heads divisible by tp_size"
            )
        if not (
            self.num_key_value_heads % self.tp_size == 0
            or self.tp_size % self.num_key_value_heads == 0
        ):
            raise ValueError(
                "standard KV sharding/replication requires either num_key_value_heads "
                "divisible by tp_size or tp_size divisible by num_key_value_heads"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def attention_type(self) -> str:
        if self.num_key_value_heads == self.num_attention_heads:
            return "MHA"
        if self.num_key_value_heads == 1:
            return "MQA"
        return "GQA"

    @property
    def local_query_heads(self) -> int:
        return self.num_attention_heads // self.tp_size

    @property
    def local_o_input_width(self) -> int:
        # o_proj consumes concatenated *query-head* outputs.  This remains true
        # for GQA/MQA; only K/V ownership changes.
        return self.local_query_heads * self.head_dim

    @property
    def kv_partition_mode(self) -> KVPartitionMode:
        if self.num_key_value_heads >= self.tp_size:
            return "sharded"
        return "replicated"

    @property
    def local_kv_heads(self) -> int:
        if self.kv_partition_mode == "sharded":
            return self.num_key_value_heads // self.tp_size
        return 1

    @property
    def kv_replication_factor(self) -> int:
        if self.kv_partition_mode == "sharded":
            return 1
        return self.tp_size // self.num_key_value_heads

    @property
    def supports_low_rank_o_communication(self) -> bool:
        # The method only requires equal query-head shards and row-parallel
        # o_proj.  It is therefore not MHA-specific.
        return self.local_o_input_width * self.tp_size == self.hidden_size
