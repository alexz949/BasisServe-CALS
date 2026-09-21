#!/usr/bin/env python3
"""Qwen3-32B V96 LRQK RULER evaluation with one TP2 replica; key codes stay on GPU.

The single-GPU LRQK baseline keeps a full model per device, which leaves no room
for the rank-32 key codes, so it streams every layer's codes (536 MB) from
pinned host memory on every decode step. Tensor parallelism halves the model
and halves each rank's share of the codes, so the codes are resident and the
per-step routing scan never touches PCIe. Only the selected exact-Key rows are
fetched from pinned host memory, as in the baseline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.distributed import DistributedConfig
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from basisserve.core.c1_k_offload import (
    PinnedCPUExactKeyPageStore,
    PreparedQueryKeyFetch,
)
from basisserve.core.c1_lrqk import LRQKConfig, _solve, select_tokens
from basisserve.kernels.compressed_v_decode_attention import (
    compressed_v_prefill_attention,
)
from basisserve.kernels.indexed_sparse_decode_attention import (
    gqa_indexed_sparse_decode_attention_triton,
)
from evaluation.chunked_prefill_mlp import (
    install_chunked_prefill_mlps,
    install_chunked_prefill_norms,
)
from evaluation.deterministic_evaluation import (
    NUMERICAL_POLICY,
    configure_deterministic_evaluation,
)
from evaluation.ruler_v1 import sample_score
from evaluation.v96kl_common import sha256

TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multiquery",
    "niah_multivalue",
    "vt",
    "fwe",
    "qa_1",
    "qa_2",
)
SEQUENCE_LENGTH = 131072
SAMPLES_PER_TASK = 100
YARN_FACTOR = 4.0
TP_SIZE = 2
GLOBAL_QUERY_HEADS = 64
GLOBAL_KV_HEADS = 8
LOCAL_QUERY_HEADS = GLOBAL_QUERY_HEADS // TP_SIZE
LOCAL_KV_HEADS = GLOBAL_KV_HEADS // TP_SIZE
HEAD_DIM = 128
VALUE_DIM = 96
# Same routing configuration as the single-GPU baseline entry point.
LRQK = LRQKConfig(
    rank=32,
    topk=832,
    recent=64,
    prefill_iterations=2,
    decode_iterations=2,
    tolerance=0.01,
    seed=0,
)
SOURCES = (
    "basisserve/core/c1_lrqk.py",
    "basisserve/core/c1_k_offload.py",
    "basisserve/kernels/compressed_v_decode_attention.py",
    "basisserve/kernels/indexed_sparse_decode_attention.py",
    "basisserve/kernels/selected_key_compaction.py",
    "evaluation/chunked_prefill_mlp.py",
)


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".rank{dist.get_rank()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if hasattr(value, "to_local") else value


def _mean_across_ranks(value: torch.Tensor) -> torch.Tensor:
    """Average a per-rank mean so both ranks see the value the full-head run would."""
    total = value.detach().clone()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total / dist.get_world_size()


class PinnedSequence:
    def __init__(self, value: torch.Tensor, capacity: int):
        assert value.ndim == 4 and value.shape[2] <= capacity
        self.storage = torch.empty(
            (*value.shape[:2], capacity, value.shape[-1]),
            dtype=value.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.length = 0
        self.append(value)

    def append(self, value: torch.Tensor) -> None:
        stop = self.length + value.shape[2]
        assert value.shape[:2] == self.storage.shape[:2]
        assert value.shape[-1] == self.storage.shape[-1]
        assert stop <= self.storage.shape[2]
        self.storage[:, :, self.length:stop].copy_(value.detach())
        self.length = stop

    def view(self) -> torch.Tensor:
        return self.storage[:, :, :self.length]


def prefill_factors_tp2(q, k, aq, ak, iterations, tolerance):
    """basisserve.core.c1_lrqk.prefill_factors on one rank's heads.

    Every operation is a per-head batched matmul or solve, so the local result
    equals the corresponding slice of the full-head result. Only the convergence
    test averages over heads; the per-rank means are averaged across ranks so
    both ranks stop on the same sweep the single-GPU run would.
    """
    assert q.dtype == k.dtype == aq.dtype == ak.dtype == torch.float32
    for _ in range(iterations):
        qq = aq.transpose(-1, -2) @ aq
        kk = ak.transpose(-1, -2) @ ak
        bq = _solve(qq, aq.transpose(-1, -2) @ q)
        bk = _solve(kk, ak.transpose(-1, -2) @ k)
        new_aq = q @ _solve(
            kk + bq @ bq.transpose(-1, -2),
            k.transpose(-1, -2) @ ak + bq.transpose(-1, -2),
            left=False,
        )
        qq = new_aq.transpose(-1, -2) @ new_aq
        new_ak = k @ _solve(
            qq + bk @ bk.transpose(-1, -2),
            q.transpose(-1, -2) @ new_aq + bk.transpose(-1, -2),
            left=False,
        )
        local = torch.stack((F.mse_loss(aq, new_aq), F.mse_loss(ak, new_ak)))
        delta = float(_mean_across_ranks(local).max())
        aq, ak = new_aq, new_ak
        if delta < tolerance:
            break
    bq = _solve(aq.transpose(-1, -2) @ aq, aq.transpose(-1, -2) @ q)
    bk = _solve(ak.transpose(-1, -2) @ ak, ak.transpose(-1, -2) @ k)
    return aq, bq, ak, bk


def _gradient_step(db, code):
    return db.square().sum(dim=(2, 3), keepdim=True) / (
        (code @ db).square().sum(dim=(2, 3), keepdim=True) + 1e-6
    )


def decode_factors_tp2(bq, ak, bk, active_k, q, k, iterations, tolerance):
    """basisserve.core.c1_lrqk.decode_factors with the convergence mean averaged across ranks."""
    kb = k @ bk.transpose(-1, -2)
    bbk = bk @ bk.transpose(-1, -2)
    rhs = q @ bq.transpose(-1, -2) + (q @ active_k.transpose(-1, -2)) @ ak
    normal = bq @ bq.transpose(-1, -2) + ak.transpose(-1, -2) @ ak
    qk = q @ k.transpose(-1, -2)
    new_k = _solve(bbk, kb, left=False)
    for _ in range(max(iterations, 1)):
        new_q = _solve(
            normal + new_k.transpose(-1, -2) @ new_k, rhs + qk @ new_k, left=False
        )
        updated_k = _solve(
            bbk + new_q.transpose(-1, -2) @ new_q, kb + qk @ new_q, left=False
        )
        delta = float(_mean_across_ranks(F.mse_loss(new_k, updated_k)))
        new_k = updated_k
        if delta < tolerance:
            break
    dbq = new_q.transpose(-1, -2) @ (new_q @ bq - q)
    dbk = new_k.transpose(-1, -2) @ (new_k @ bk - k)
    return (
        bq - _gradient_step(dbq, new_q) * dbq,
        bk - _gradient_step(dbk, new_k) * dbk,
        new_q,
        new_k,
    )


class TP2LRQKState:
    """One rank's slice of the LRQK routing state, with the key codes resident on the GPU."""

    @torch.inference_mode()
    def __init__(self, q, k, config: LRQKConfig, layer: int, tp_rank: int, capacity: int):
        assert q.shape[2] == k.shape[2] and q.shape[2] >= config.rank
        assert q.shape[1] == LOCAL_QUERY_HEADS and k.shape[1] == LOCAL_KV_HEADS
        self.config = config
        self.length = k.shape[2]
        self.steps = 0
        # The baseline draws its ALS initialisation for all 64 heads from one
        # generator stream. Draw the same full tensors and keep this rank's heads,
        # so the sharded fit starts from exactly the values the baseline used.
        generator = torch.Generator(device=q.device).manual_seed(config.seed + layer)
        full = (q.shape[0], GLOBAL_QUERY_HEADS, q.shape[2], config.rank)
        start = tp_rank * LOCAL_QUERY_HEADS
        stop = start + LOCAL_QUERY_HEADS
        aq = torch.randn(full, device=q.device, dtype=torch.float32, generator=generator)
        aq = aq[:, start:stop].contiguous()
        ak = torch.randn(full, device=q.device, dtype=torch.float32, generator=generator)
        ak = ak[:, start:stop].contiguous()
        repeated_k = k.repeat_interleave(LOCAL_QUERY_HEADS // LOCAL_KV_HEADS, dim=1)
        factors = prefill_factors_tp2(
            q.float(), repeated_k.float(), aq, ak,
            config.prefill_iterations, config.tolerance,
        )
        aq, self.bq, ak, self.bk = [t.to(q.dtype) for t in factors]
        assert all(torch.isfinite(t).all() for t in (aq, self.bq, ak, self.bk))
        del repeated_k
        # Preallocated at capacity so decode steps append in place, mirroring the
        # baseline's shared workspace but without the host round trip.
        self.ak = torch.empty(
            q.shape[0], LOCAL_QUERY_HEADS, capacity, config.rank,
            dtype=q.dtype, device=q.device,
        )
        self.ak[:, :, : self.length].copy_(ak)
        self.selected = select_tokens(aq[:, :, -1:], self.codes(), config)

    def codes(self) -> torch.Tensor:
        return self.ak[:, :, : self.length]

    def statistics(self, kv_heads: int):
        batch, heads, count = self.selected.shape
        grouped = self.selected.reshape(batch, kv_heads, -1)
        ordered = grouped.sort(dim=-1).values
        union = 1 + (ordered[..., 1:] != ordered[..., :-1]).sum(-1)
        codes = self.codes()
        return dict(
            length=self.length,
            decode_steps=self.steps,
            selected_per_query_head=count,
            physical_union_per_kv_group=union.cpu().tolist(),
            key_code_shape=list(codes.shape),
            key_code_bytes=codes.numel() * codes.element_size(),
        )


class TP2LRQKCache(DynamicCache):
    def __init__(self, config, capacity: int):
        super().__init__(config=config)
        self.capacity = int(capacity)
        self.groups = LOCAL_QUERY_HEADS // LOCAL_KV_HEADS
        self.lengths = {}
        self.lrqk_states = {}
        self.host_keys = {}
        self.resident_values = {}
        self.key_fetch = None
        self.statistics = {}

    def get_seq_length(self, layer_idx: int = 0):
        return self.lengths.get(layer_idx, 0)

    def store_prefill(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.host_keys[layer] = PinnedSequence(key, self.capacity)
        resident = torch.empty(
            value.shape[0], value.shape[1], self.capacity, value.shape[-1],
            dtype=value.dtype, device=value.device,
        )
        resident[:, :, : value.shape[2]].copy_(value)
        self.resident_values[layer] = resident
        self.lengths[layer] = key.shape[2]
        self.layers[layer].keys = self.host_keys[layer].view()
        self.layers[layer].values = resident[:, :, : self.lengths[layer]]

    def append_key_and_value(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.host_keys[layer].append(key)
        previous = self.lengths[layer]
        self.resident_values[layer][:, :, previous : previous + 1].copy_(value)
        self.lengths[layer] = previous + 1
        self.layers[layer].keys = self.host_keys[layer].view()
        self.layers[layer].values = self.resident_values[layer][:, :, : previous + 1]

    def value(self, layer: int) -> torch.Tensor:
        return self.resident_values[layer][:, :, : self.lengths[layer]]

    def fetch_exact(self, layer: int, ids: torch.Tensor, device):
        batch, query_heads, count = ids.shape
        kv_heads = self.host_keys[layer].storage.shape[1]
        assert query_heads == kv_heads * self.groups
        grouped = ids.reshape(batch, kv_heads, self.groups, count).contiguous()
        if self.key_fetch is None or self.key_fetch.shape[-1] != count:
            source = PinnedCPUExactKeyPageStore(
                self.host_keys[layer].storage, layer_idx=layer
            )
            self.key_fetch = PreparedQueryKeyFetch(
                source, groups=self.groups, tokens_per_query=count, device=device
            )
        self.key_fetch.bank = self.host_keys[layer].storage.reshape(
            batch * kv_heads * self.capacity, -1
        )
        self.key_fetch(grouped)
        return self.key_fetch.destination, self.key_fetch.inverse_ids.reshape(
            batch, query_heads, count
        )


def _validate_mask(attention_mask, query: torch.Tensor, previous: int) -> None:
    if attention_mask is None:
        return
    assert attention_mask.ndim == 4
    length = query.shape[2]
    expected = torch.arange(previous + length, device=query.device)[None, :] <= (
        previous + torch.arange(length, device=query.device)[:, None]
    )
    valid = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    assert torch.equal(
        valid.expand(query.shape[0], 1, length, previous + length)[0, 0], expected
    )


class TP2C1LRQKAttention(nn.Module):
    def __init__(
        self,
        base_attention: nn.Module,
        compressed_v_weight: torch.Tensor,
        local_decoder_weight: torch.Tensor,
        tp_rank: int,
    ):
        super().__init__()
        self.layer_idx = int(base_attention.layer_idx)
        self.head_dim = HEAD_DIM
        self.value_head_dim = VALUE_DIM
        self.num_attention_heads = LOCAL_QUERY_HEADS
        self.num_key_value_heads = LOCAL_KV_HEADS
        self.num_key_value_groups = LOCAL_QUERY_HEADS // LOCAL_KV_HEADS
        self.scaling = float(base_attention.scaling)
        self.attention_dropout = float(base_attention.attention_dropout)
        self.is_causal = bool(base_attention.is_causal)
        self.sliding_window = base_attention.sliding_window
        self.tp_rank = int(tp_rank)
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        device = _local_tensor(base_attention.v_proj.weight).device
        dtype = _local_tensor(base_attention.v_proj.weight).dtype
        self.v_proj = nn.Linear(
            5120, LOCAL_KV_HEADS * VALUE_DIM, bias=False, device=device, dtype=dtype
        )
        self.v_proj.weight.copy_(compressed_v_weight.to(device=device, dtype=dtype))
        self.register_buffer(
            "local_decoder_weight",
            local_decoder_weight.to(device=device, dtype=dtype).contiguous(),
        )

    def _decode_output(self, output: torch.Tensor) -> torch.Tensor:
        flattened = output.transpose(1, 2).contiguous().reshape(
            output.shape[0], output.shape[2], LOCAL_QUERY_HEADS * VALUE_DIM
        )
        hidden = F.linear(flattened, self.local_decoder_weight)
        dist.all_reduce(hidden, op=dist.ReduceOp.SUM)
        return hidden

    @torch.inference_mode()
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        assert isinstance(past_key_values, TP2LRQKCache)
        batch, length, _ = hidden_states.shape
        assert batch > 0 and not kwargs.get("output_attentions", False)
        query = self.q_norm(
            self.q_proj(hidden_states).view(batch, length, LOCAL_QUERY_HEADS, HEAD_DIM)
        ).transpose(1, 2)
        pre_key = self.k_norm(
            self.k_proj(hidden_states).view(batch, length, LOCAL_KV_HEADS, HEAD_DIM)
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, length, LOCAL_KV_HEADS, VALUE_DIM
        ).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, pre_key, cos, sin)
        previous = past_key_values.get_seq_length(self.layer_idx)
        assert previous == 0 or length == 1
        _validate_mask(attention_mask, query, previous)
        config = LRQK
        if previous == 0:
            state = TP2LRQKState(
                query, key, config, self.layer_idx, self.tp_rank, past_key_values.capacity
            )
            past_key_values.lrqk_states[self.layer_idx] = state
            output = compressed_v_prefill_attention(query, key, value, scale=self.scaling)
            past_key_values.store_prefill(self.layer_idx, key, value)
        else:
            past_key_values.append_key_and_value(self.layer_idx, key, value)
            state = past_key_values.lrqk_states[self.layer_idx]
            assert state.length == previous and state.selected.max() < previous
            prior_key, prior_rows = past_key_values.fetch_exact(
                self.layer_idx, state.selected, query.device
            )
            active_key = prior_key[prior_rows].reshape(
                batch, LOCAL_QUERY_HEADS, state.selected.shape[-1], HEAD_DIM
            )
            active_code = state.codes().gather(
                2, state.selected[..., None].expand(-1, -1, -1, config.rank)
            )
            current_key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            factors = decode_factors_tp2(
                state.bq.float(),
                active_code.float(),
                state.bk.float(),
                active_key.float(),
                query.float(),
                current_key.float(),
                config.decode_iterations,
                config.tolerance,
            )
            state.bq, state.bk, query_code, key_code = [
                tensor.to(query.dtype) for tensor in factors
            ]
            state.ak[:, :, previous : previous + 1].copy_(key_code)
            state.length = previous + 1
            state.steps += 1
            state.selected = select_tokens(query_code, state.codes(), config)
            selected_key, selected_rows = past_key_values.fetch_exact(
                self.layer_idx, state.selected, query.device
            )
            output = gqa_indexed_sparse_decode_attention_triton(
                query,
                selected_key,
                past_key_values.value(self.layer_idx),
                state.selected,
                selected_key_rows=selected_rows,
                scale=self.scaling,
            )
            past_key_values.statistics[self.layer_idx] = {
                **state.statistics(LOCAL_KV_HEADS),
                **past_key_values.key_fetch.traffic(),
                "exact_key_storage": "pinned_cpu",
                "value_storage": "cuda",
                "key_code_storage": "cuda",
            }
        return self._decode_output(output), None


def effective_config(model_path: Path):
    native = AutoConfig.from_pretrained(model_path, local_files_only=True)
    assert (
        native.model_type,
        native.num_hidden_layers,
        native.num_attention_heads,
        native.num_key_value_heads,
        native.head_dim,
    ) == ("qwen3", 64, GLOBAL_QUERY_HEADS, GLOBAL_KV_HEADS, HEAD_DIM)
    maximum = int(round(native.max_position_embeddings * YARN_FACTOR))
    rope = {
        **dict(native.rope_parameters),
        "rope_type": "yarn",
        "factor": YARN_FACTOR,
        "original_max_position_embeddings": native.max_position_embeddings,
    }
    return AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        max_position_embeddings=maximum,
        rope_parameters=rope,
    )


def factor_identity(model_path: Path, factors: Path):
    result = read_json(factors / "results.json")
    fit = result["fit_config"]
    assert result["format"] == "basisserve.qwen3_32b.gqa_c1_v96_joint.v1"
    assert result["status"] == "complete"
    assert result["layers"] == list(range(64))
    assert fit["cache_rank_per_head"] == VALUE_DIM
    assert fit["model_config_sha256"] == sha256(model_path / "config.json")
    artifact_hashes = {}
    for layer in range(64):
        record = result["artifacts"][str(layer)]
        path = factors / record["file"]
        assert sha256(path) == record["sha256"]
        artifact_hashes[str(layer)] = record["sha256"]
    return {
        "results_sha256": sha256(factors / "results.json"),
        "artifact_sha256": artifact_hashes,
    }


@torch.no_grad()
def install_tp2_v96_lrqk(model, factors: Path, tp_rank: int) -> None:
    manifest = read_json(factors / "results.json")
    kv_start = tp_rank * LOCAL_KV_HEADS
    kv_stop = kv_start + LOCAL_KV_HEADS
    query_start = tp_rank * LOCAL_QUERY_HEADS
    query_stop = query_start + LOCAL_QUERY_HEADS
    for layer_index, layer in enumerate(model.model.layers):
        base = layer.self_attn
        record = manifest["artifacts"][str(layer_index)]
        payload = load_file(str(factors / record["file"]), device="cpu")
        encoders = payload["value_coordinate_encoders"][kv_start:kv_stop].float()
        decoders = payload["head_output_decoders"][query_start:query_stop].float()
        dense_v = _local_tensor(base.v_proj.weight).detach().float()
        assert dense_v.shape == (LOCAL_KV_HEADS * HEAD_DIM, 5120)
        compressed_v = torch.bmm(
            encoders.to(dense_v.device).transpose(1, 2),
            dense_v.reshape(LOCAL_KV_HEADS, HEAD_DIM, 5120),
        ).reshape(LOCAL_KV_HEADS * VALUE_DIM, 5120)
        decoder_weight = (
            decoders.to(dense_v.device)
            .permute(2, 0, 1)
            .reshape(5120, LOCAL_QUERY_HEADS * VALUE_DIM)
        )
        layer.self_attn = TP2C1LRQKAttention(base, compressed_v, decoder_weight, tp_rank)
    install_chunked_prefill_mlps(model)
    install_chunked_prefill_norms(model)


def load_model(args, mesh, tp_rank: int):
    config = effective_config(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        distributed_config=DistributedConfig(tp_size=TP_SIZE),
        device_mesh=mesh,
        local_files_only=True,
    ).eval()
    install_tp2_v96_lrqk(model, args.factors, tp_rank)
    return model


def load_inputs(args):
    metadata = read_json(args.prompts_json)
    assert metadata["status"] == "complete"
    assert metadata["model_config_sha256"] == sha256(args.model / "config.json")
    assert metadata["tokens_sha256"] == sha256(args.prompts)
    rows = metadata["rows"]
    assert len(rows) == len(TASKS) * SAMPLES_PER_TASK
    assert Counter(row["task"] for row in rows) == dict.fromkeys(TASKS, SAMPLES_PER_TASK)
    assert [row["index"] for row in rows] == list(range(len(rows)))
    return metadata, rows


def protocol(args, metadata):
    return {
        "format": "basisserve.qwen3_32b_v96_ruler128k_lrqk_tp2.v1",
        "model_config_sha256": sha256(args.model / "config.json"),
        "factor_identity": factor_identity(args.model, args.factors),
        "prompts_sha256": sha256(args.prompts),
        "prompts_metadata_sha256": sha256(args.prompts_json),
        "dtype": "bfloat16",
        "sequence_length": SEQUENCE_LENGTH,
        "tasks": list(TASKS),
        "samples_per_task": SAMPLES_PER_TASK,
        "arm": "lrqk",
        "tensor_parallel_size": TP_SIZE,
        "replicas": args.shards,
        "batch_size": args.batch_size,
        "parallelism": (
            f"{args.shards} independent TP2 replicas; up to {args.batch_size} "
            "equal-length prompts per replica"
        ),
        "value_cache": "uniform C1 V96 with TP-local value heads and output all-reduce",
        "lrqk": {
            **asdict(LRQK),
            "head_sharding": (
                f"{LOCAL_QUERY_HEADS} query heads and {LOCAL_KV_HEADS} KV heads per rank; "
                "ALS initialisation drawn from the full 64-head generator stream and "
                "sliced; convergence means averaged across ranks"
            ),
        },
        "memory_policy": (
            "half of the BF16 model per rank; rank-32 key codes and V96 values resident "
            "on the GPU; exact Keys in pinned host memory with only selected rows fetched"
        ),
        "rope": {
            "rope_type": "yarn",
            "factor": YARN_FACTOR,
            "original_max_position_embeddings": 40960,
            "effective_max_position_embeddings": 163840,
        },
        "generation": "greedy; native EOS; official task caps",
        "generation_seed": 0,
        "numerical_policy": NUMERICAL_POLICY,
        "script_sha256": sha256(Path(__file__)),
        "source_sha256": {name: sha256(REPO / name) for name in SOURCES},
        "input_template": metadata["input_template"],
    }


def tensor_for_row(path: Path, row, limit: int | None):
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(str(row["index"])).to(torch.int64)
    assert len(tensor) == row["input_tokens"]
    assert hashlib.sha256(tensor.to(torch.int32).numpy().tobytes()).hexdigest() == row[
        "input_sha256"
    ]
    if limit is not None:
        tensor = tensor[:limit].contiguous()
    return tensor


def agreed_tokens(value: torch.Tensor) -> list[int]:
    current = value.to(dtype=torch.int64)
    values = [torch.empty_like(current) for _ in range(TP_SIZE)]
    dist.all_gather(values, current)
    assert all(torch.equal(values[0], other) for other in values[1:])
    return [int(item) for item in values[0].tolist()]


@torch.inference_mode()
def generate(model, tokenizer, rows, input_ids, cap: int):
    torch.manual_seed(0)
    assert input_ids.ndim == 2 and input_ids.shape[0] == len(rows)
    cache = TP2LRQKCache(model.config, input_ids.shape[1] + cap)
    device = model.get_input_embeddings().weight.device
    tokens = input_ids.to(device=device, dtype=torch.long)
    output = model(
        input_ids=tokens,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    assert torch.isfinite(output.logits).all()
    first_tokens = agreed_tokens(output.logits[:, -1].argmax(dim=-1))
    ids = [[token] for token in first_tokens]
    del output, tokens
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    active = [tokens[-1] not in eos for tokens in ids]
    steps = 1
    fallback_eos = min(eos)
    while steps < cap and any(active):
        mask = torch.ones(
            len(rows), 1, 1, cache.get_seq_length() + 1, dtype=torch.bool, device=device
        )
        current = [tokens[-1] if keep else fallback_eos for tokens, keep in zip(ids, active)]
        output = model(
            input_ids=torch.tensor(current, device=device).unsqueeze(1),
            past_key_values=cache,
            attention_mask=mask,
            use_cache=True,
            logits_to_keep=1,
        )
        assert torch.isfinite(output.logits).all()
        next_tokens = agreed_tokens(output.logits[:, -1].argmax(dim=-1))
        for index, token in enumerate(next_tokens):
            if active[index]:
                ids[index].append(token)
                active[index] = token not in eos
        steps += 1
        del output
    routing = []
    expected_length = input_ids.shape[1] + steps - 1
    for layer_index in range(64):
        assert cache.lengths[layer_index] == expected_length
        state = cache.lrqk_states[layer_index]
        routing.append(
            cache.statistics.get(
                layer_index,
                {
                    **state.statistics(LOCAL_KV_HEADS),
                    "unique_key_tokens": 0,
                    "logical_k_bytes": 0,
                    "h2d_dma_payload_bytes": 0,
                    "d2h_index_payload_bytes": 0,
                    "measured_pcie_bus_bytes": None,
                    "exact_key_storage": "pinned_cpu",
                    "value_storage": "cuda",
                    "key_code_storage": "cuda",
                },
            )
        )
    stopped = [tokens[-1] in eos for tokens in ids]
    del cache
    torch.cuda.empty_cache()
    return ids, routing, stopped


def audit(saved, row, spec, tokenizer, cap: int, prompt_tokens: int):
    assert saved["status"] == "complete"
    assert saved["protocol"] == spec
    assert saved["sample"] == row
    result = saved["result"]
    ids = result["ids"]
    assert result["input_tokens"] == prompt_tokens
    assert 0 < len(ids) <= cap
    eos = set(spec.get("eos_ids", []))
    assert not any(token in eos for token in ids[:-1])
    assert result["stopped"] == (ids[-1] in eos)
    assert result["stopped"] or len(ids) == cap
    assert tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ) == result["prediction"]
    assert len(result["routing_tp_ranks"]) == TP_SIZE
    assert all(len(values) == 64 for values in result["routing_tp_ranks"])
    return result


LRQK_FIELDS = ("rank", "topk", "recent", "prefill_iterations", "decode_iterations", "tolerance", "seed")


def audit_resumed(saved, row, spec, tokenizer):
    """Accept an earlier TP2 LRQK sample whose algorithmic protocol matches ours."""
    assert saved["status"] == "complete"
    assert saved["sample"] == row
    prior = saved["protocol"]
    assert saved.get("arm", prior.get("arm")) == "lrqk"
    assert prior["format"] == spec["format"]
    assert prior["model_config_sha256"] == spec["model_config_sha256"]
    assert prior["factor_identity"] == spec["factor_identity"]
    assert all(prior["lrqk"][name] == spec["lrqk"][name] for name in LRQK_FIELDS)
    assert prior["numerical_policy"] == spec["numerical_policy"]
    assert prior["tasks"] == spec["tasks"]
    assert prior["rope"] == spec["rope"]
    assert prior["generation"] == spec["generation"]
    assert prior["generation_seed"] == spec["generation_seed"]
    assert prior["source_sha256"]["basisserve/core/c1_lrqk.py"] == spec["source_sha256"]["basisserve/core/c1_lrqk.py"]
    result = saved["result"]
    ids = result["ids"]
    cap = row["maximum_tokens"]
    assert 0 < len(ids) <= cap
    eos = set(spec["eos_ids"])
    assert not any(token in eos for token in ids[:-1])
    assert result["stopped"] == (ids[-1] in eos)
    assert result["stopped"] or len(ids) == cap
    prediction = tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assert prediction == result["prediction"]
    assert result["score"] == sample_score(prediction, row["answers"], row["match_type"])
    return result


def load_resumed(args, rows, spec, tokenizer):
    if not args.resume_from:
        return {}
    resumed = {}
    for directory in args.resume_from:
        for row in rows:
            path = directory / f"sample_{row['index']:04d}.json"
            if path.exists():
                saved = read_json(path)
                audit_resumed(saved, row, spec, tokenizer)
                if row["index"] in resumed:
                    assert resumed[row["index"]]["result"]["ids"] == saved["result"]["ids"]
                else:
                    resumed[row["index"]] = saved
    return resumed


def compare_to_reference(directory: Path, row, ids: list[int], score: float):
    """Report agreement with a single-GPU LRQK sample, when one exists."""
    path = directory / f"sample_{row['index']:04d}.json"
    if not path.exists():
        return None
    reference = read_json(path)["result"]
    reference_ids = reference["ids"]
    common = min(len(ids), len(reference_ids))
    divergence = next(
        (position for position in range(common) if ids[position] != reference_ids[position]),
        None,
    )
    if divergence is None and len(ids) != len(reference_ids):
        divergence = common
    return {
        "reference_sample": str(path),
        "identical": ids == reference_ids,
        "first_divergence": divergence,
        "reference_length": len(reference_ids),
        "reference_score": reference["score"],
        "score": score,
        "score_agrees": reference["score"] == score,
    }


def summarize(args, rows, spec, tokenizer, resumed):
    results = []
    for row in rows:
        if row["index"] in resumed:
            results.append(audit_resumed(resumed[row["index"]], row, spec, tokenizer))
        else:
            path = args.output / "evaluate" / f"sample_{row['index']:04d}.json"
            results.append(
                audit(read_json(path), row, spec, tokenizer, row["maximum_tokens"], row["input_tokens"])
            )
    tasks = {
        task: 100 * sum(
            result["score"] for result, row in zip(results, rows, strict=True) if row["task"] == task
        ) / SAMPLES_PER_TASK
        for task in TASKS
    }
    summary = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": spec,
        "tasks": tasks,
        "mean": sum(tasks.values()) / len(tasks),
        "verified_predictions": len(results),
        "resumed_predictions": len(resumed),
        "new_predictions": len(results) - len(resumed),
    }
    write_json_atomic(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("dev", "smoke", "evaluate", "summarize"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--prompt-token-limit", type=int)
    parser.add_argument("--generation-token-limit", type=int)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume-from", type=Path, action="append", default=[])
    parser.add_argument(
        "--compare-to", type=Path,
        help="Directory of single-GPU LRQK samples to report token agreement against",
    )
    return parser.parse_args()


def make_batches(rows, batch_size: int):
    if batch_size == 1:
        return [[row] for row in rows]
    grouped = {}
    for row in rows:
        grouped.setdefault((row["input_tokens"], row["maximum_tokens"]), []).append(row)
    batches = []
    singletons = []
    for group in grouped.values():
        for start in range(0, len(group) - 1, 2):
            batches.append(group[start:start + 2])
        if len(group) % 2:
            singletons.append([group[-1]])
    batches.extend(singletons)
    return sorted(batches, key=lambda batch: min(row["index"] for row in batch))


def main():
    args = parse_args()
    configure_deterministic_evaluation()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    mesh = init_device_mesh("cuda", (TP_SIZE,), mesh_dim_names=("tp",))
    tp_rank = dist.get_rank()
    metadata, rows = load_inputs(args)
    spec = protocol(args, metadata)
    native_eos = read_json(args.model / "config.json")["eos_token_id"]
    spec["eos_ids"] = sorted(native_eos if isinstance(native_eos, list) else [native_eos])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    resumed = load_resumed(args, rows, spec, tokenizer)
    if tp_rank == 0 and resumed:
        print(
            "RESUME", len(resumed), "verified prior results;",
            len(rows) - len(resumed), "samples remain", flush=True,
        )
    if args.stage == "summarize":
        if tp_rank == 0:
            summarize(args, rows, spec, tokenizer, resumed)
        dist.barrier()
        dist.destroy_process_group()
        return

    model = load_model(args, mesh, tp_rank)
    batch_rows = rows
    if args.stage == "evaluate" and args.sample_index is None:
        batch_rows = [row for row in rows if row["index"] not in resumed]
    batches = make_batches(batch_rows, args.batch_size)
    if args.sample_index is not None:
        selected = [next(batch for batch in batches if any(
            row["index"] == args.sample_index for row in batch
        ))]
    elif args.stage in ("dev", "smoke"):
        selected = [batches[0]]
    else:
        assert 0 <= args.shard < args.shards
        selected = batches[args.shard::args.shards]
    directory = (
        f"{args.stage}-b{args.batch_size}" if args.stage in ("dev", "smoke") else "evaluate"
    )
    for batch in selected:
        paths = [args.output / directory / f"sample_{row['index']:04d}.json" for row in batch]
        prompt_tokens = (
            batch[0]["input_tokens"]
            if args.prompt_token_limit is None
            else min(batch[0]["input_tokens"], args.prompt_token_limit)
        )
        cap = (
            batch[0]["maximum_tokens"]
            if args.generation_token_limit is None
            else min(batch[0]["maximum_tokens"], args.generation_token_limit)
        )
        if all(path.exists() for path in paths):
            if tp_rank == 0:
                for path, row in zip(paths, batch, strict=True):
                    audit(read_json(path), row, spec, tokenizer, cap, prompt_tokens)
                print("SKIP", [row["index"] for row in batch], flush=True)
            dist.barrier()
            continue
        tensors = [tensor_for_row(args.prompts, row, args.prompt_token_limit) for row in batch]
        assert len({len(tensor) for tensor in tensors}) == 1
        tensor = torch.stack(tensors)
        torch.cuda.reset_peak_memory_stats()
        if tp_rank == 0:
            print(
                "START", [row["index"] for row in batch], [row["task"] for row in batch],
                tensor.shape[1], cap, datetime.now(timezone.utc).isoformat(), flush=True,
            )
        started = time.monotonic()
        ids, local_routing, stopped = generate(model, tokenizer, batch, tensor, cap)
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.monotonic() - started, device=f"cuda:{local_rank}")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        peak = torch.tensor(
            torch.cuda.max_memory_allocated(), dtype=torch.int64, device=f"cuda:{local_rank}"
        )
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        routing_by_rank = [None] * TP_SIZE
        dist.all_gather_object(routing_by_rank, local_routing)
        if tp_rank == 0:
            scores = []
            for path, row, row_ids, row_stopped in zip(paths, batch, ids, stopped, strict=True):
                prediction = tokenizer.decode(
                    row_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                score = sample_score(prediction, row["answers"], row["match_type"])
                scores.append(score)
                result = {
                    "ids": row_ids,
                    "prediction": prediction,
                    "stopped": row_stopped,
                    "score": score,
                    "input_tokens": tensor.shape[1],
                    "seconds": float(elapsed.item()),
                    "batch_size": len(batch),
                    "peak_gib_per_rank": int(peak.item()) / 2**30,
                    "routing_tp_ranks": routing_by_rank,
                }
                if args.compare_to is not None:
                    comparison = compare_to_reference(args.compare_to, row, row_ids, score)
                    if comparison is not None:
                        result["reference_comparison"] = comparison
                        print("COMPARE", row["index"], json.dumps(comparison), flush=True)
                saved = {
                    "status": "complete",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "protocol": spec,
                    "sample": row,
                    "arm": "lrqk",
                    "result": result,
                    "environment": {
                        "hostname": platform.node(),
                        "python": sys.version,
                        "torch": torch.__version__,
                        "torch_cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(),
                    },
                }
                audit(saved, row, spec, tokenizer, cap, tensor.shape[1])
                write_json_atomic(path, saved)
            print(
                "COMPLETE", [row["index"] for row in batch], scores,
                float(elapsed.item()), int(peak.item()) / 2**30,
                [row_ids[:8] for row_ids in ids], flush=True,
            )
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
