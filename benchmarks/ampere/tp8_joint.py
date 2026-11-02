"""Llama-3.1-8B-Instruct TP8 decode smoke and Ampere benchmark runner."""

import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, rotate_half

from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.mapped_host_paged_attention import (
    append_mapped_host_key,
    mapped_host_bf16_empty,
    mapped_host_device_pointer,
)
from basisserve.kernels.ragged_allgather import StaticRaggedPlan
from basisserve.kernels.slot_indexed_attention import slot_indexed_attention
from basisserve.kernels.compressed_v_decode_attention import compressed_v_decode_attention_triton
from benchmarks.ampere.build_kernels import configure_local_caches, load_kernels
from benchmarks.system.fused_candidates import candidates
from benchmarks.system.two_stage_router import Metadata


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "benchmarks/ampere/tp8_joint.py",
    "benchmarks/ampere/build_kernels.py",
    "benchmarks/ampere/kernel_smoke.py",
    "benchmarks/ampere/two_stage.py",
    "benchmarks/system/fused_candidates.py",
    "benchmarks/system/two_stage_router.py",
    "basisserve/kernels/slot_indexed_attention.py",
    "basisserve/kernels/compressed_v_decode_attention.py",
    "basisserve/kernels/feature_ragged_allgather.py",
    "basisserve/kernels/ragged_allgather.py",
    "basisserve/kernels/mapped_host_paged_attention.py",
    "basisserve/kernels/csrc/mapped_host_paged_attention.cpp",
    "basisserve/kernels/csrc/mapped_host_paged_attention.cu",
    "basisserve/kernels/csrc/feature_ragged_allgather.cpp",
    "basisserve/kernels/csrc/feature_ragged_allgather_pack.cu",
    "basisserve/kernels/csrc/feature_uniform_allgather_ipc.cu",
    "basisserve/kernels/csrc/conditional_router_page32.cu",
    "basisserve/kernels/csrc/fused_decode_postprocess.cu",
    "basisserve/kernels/csrc/persistent_key_slots.cu",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flash(query, key, value, *, causal):
    mask = causal_lower_right(query.shape[2], key.shape[2]) if causal else None
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=False,
            enable_gqa=True, scale=128**-0.5,
        )


def compact_prefill(query, key, value):
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(
            query, key, value, is_causal=True, enable_gqa=True,
            scale=128**-0.5,
        )


class TP8Attention(nn.Module):
    def __init__(self, original, *, rank, batch, capacity, mode, factor_path,
                 router_path, shared_rope, communicator, kernels):
        super().__init__()
        self.mode = mode
        self.rank = rank
        self.layer_idx = original.layer_idx
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.scaling = original.scaling
        self.length = 0
        self.capacity = capacity
        self.shared_rope = shared_rope
        self.dispatch = Counter()
        if mode == "dense":
            self.v_proj = original.v_proj
            self.o_proj = original.o_proj
            self.key_cache = torch.empty(batch, 1, capacity, 128, device="cuda", dtype=torch.bfloat16)
            self.value_cache = torch.empty_like(self.key_cache)
            return

        factor = load_file(str(factor_path))
        encoder = factor["value_coordinate_encoders"][rank:rank+1].cuda().bfloat16()
        weight = original.v_proj.weight.detach()
        local_weight = weight.to_local().reshape(1, 128, 4096)
        folded = torch.einsum("hdr,hdm->hrm", encoder.float(), local_weight.float())
        self.v_proj = nn.Linear(4096, 96, bias=False, device="cuda", dtype=torch.bfloat16)
        self.v_proj.weight = nn.Parameter(folded.reshape(96, 4096).bfloat16(), requires_grad=False)
        self.register_buffer(
            "global_decoder",
            factor["head_output_decoders"].reshape(32 * 96, 4096).cuda().bfloat16(),
        )
        self.value_cache = torch.empty(batch, 1, capacity, 96, device="cuda", dtype=torch.bfloat16)
        self.communicator = communicator
        self.plan = StaticRaggedPlan.from_source_widths((4 * 96,) * 8)
        self.decode_ag = communicator.prepare_uniform(
            self.plan, tokens=batch, dtype=torch.bfloat16, backend="uniform_nccl"
        )
        if mode == "als_full":
            self.key_cache = torch.empty(batch, 1, capacity, 128, device="cuda", dtype=torch.bfloat16)
            self.valid_length = torch.empty((), device="cuda", dtype=torch.int32)
            return

        assert mode == "joint" and router_path is not None
        router = load_file(str(router_path))
        self.register_buffer("base_left", router["base_left_b16"][rank:rank+1].cuda().bfloat16().contiguous())
        self.register_buffer("base_right", router["base_right_b16"][rank:rank+1].cuda().bfloat16().contiguous())
        self.register_buffer("base_bias", router["base_bias_b16"][rank:rank+1].cuda().bfloat16().contiguous())
        self.register_buffer("residual_encoder", router["residual_encoder_b16_r16"][rank:rank+1].cuda().bfloat16().contiguous())
        self.register_buffer("residual_query", router["residual_query_b16_r16"][rank*4:(rank+1)*4].cuda().bfloat16().contiguous())
        self.base_cache = torch.empty(batch, 1, capacity, 16, device="cuda", dtype=torch.bfloat16)
        self.residual_cache = torch.empty_like(self.base_cache)
        self.host_key = mapped_host_bf16_empty(batch=batch, kv_heads=1, capacity=capacity)
        self.host_pointer = mapped_host_device_pointer(self.host_key)
        self.metadata = None
        self.kernels = dict(kernels)
        self.query_code = torch.empty(batch, 1, 4, 16, device="cuda", dtype=torch.bfloat16)
        self.candidate_ids = torch.empty(batch, 1, 512, device="cuda", dtype=torch.long)
        self.fine_scores = torch.empty(batch, 1, 4, 512, device="cuda")
        self.selected_pages = torch.empty(batch, 1, 62, device="cuda", dtype=torch.long)
        self.support_ids = torch.empty(batch, 1, 2048, device="cuda", dtype=torch.long)
        self.slot_keys = torch.empty(batch, 1, 2048, 128, device="cuda", dtype=torch.bfloat16)
        self.slot_resident = torch.full((batch, 1, 2048), -1, device="cuda", dtype=torch.long)
        self.slot_lookup = torch.full((batch, 1, capacity), -1, device="cuda", dtype=torch.int32)
        self.selected_slots = torch.empty_like(self.support_ids)
        self.slot_missing = torch.empty_like(self.support_ids, dtype=torch.int32)
        self.slot_counts = torch.empty(batch, 1, 2, device="cuda", dtype=torch.int32)
        self.slot_workspace = (
            torch.empty(batch, 4, 16, 96, device="cuda"),
            torch.empty(batch, 4, 16, device="cuda"),
            torch.empty(batch, 4, 1, 96, device="cuda", dtype=torch.bfloat16),
        )

    def reset_request(self):
        self.length = 0
        self.dispatch.clear()
        if self.mode == "joint":
            self.metadata = None
            self.kernels["prefill_key"] = None
            self.slot_resident.fill_(-1)
            self.slot_lookup.fill_(-1)

    def project(self, attention, *, decode):
        batch, heads, tokens, width = attention.shape
        assert heads == 4 and width == 96
        prepared = self.decode_ag if decode else self.communicator.prepare_uniform(
            self.plan, tokens=batch*tokens, dtype=attention.dtype, backend="uniform_nccl"
        )
        prepared.local_feature_major_view_fast().copy_(
            attention.permute(1, 3, 0, 2).reshape(heads*width, batch*tokens)
        )
        full = prepared.gather_inplace_fast()
        self.dispatch["prepared_allgather"] += 1
        return (full.T @ self.global_decoder).reshape(batch, tokens, 4096)

    def store(self, key, value, cos, sin, start):
        stop = start + key.shape[2]
        self.value_cache[:, :, start:stop] = value
        if self.mode == "joint":
            base = torch.einsum("bhtd,hdr->bhtr", value, self.base_left)
            predicted = (torch.einsum("bhtr,hrd->bhtd", base, self.base_right)
                         + self.base_bias[None, :, None]).bfloat16()
            rotated = (predicted*cos[:, None] + rotate_half(predicted)*sin[:, None]).bfloat16()
            residual = torch.einsum("bhtd,hdr->bhtr", (key-rotated).bfloat16(), self.residual_encoder)
            self.base_cache[:, :, start:stop] = base
            self.residual_cache[:, :, start:stop] = residual
        else:
            self.key_cache[:, :, start:stop] = key

    def route_and_attend(self, query, key, end):
        historical = end - 64
        self.metadata.advance(key)
        coarse = self.metadata.scores(query)
        candidate_ids = candidates(coarse, historical, output=self.candidate_ids)
        n = candidate_ids.shape[-1]
        scores = self.fine_scores[:, :, :, :n]
        self.kernels["fine"].conditional_router_page_lse(
            query, self.base_cache[:, :, :historical], self.residual_cache[:, :, :historical],
            self.base_right, self.base_bias, self.residual_query,
            self.shared_rope["cos"][:historical], self.shared_rope["sin"][:historical],
            self.query_code, scores, self.scaling, False, candidate_ids,
        )
        self.dispatch["two_stage_router"] += 1
        self.kernels["postprocess"].select_and_pack(
            scores, candidate_ids, self.selected_pages, self.support_ids, historical, end
        )
        self.dispatch["select_pack"] += 1
        self.kernels["slots"].refresh(
            self.host_pointer, self.slot_keys, self.support_ids,
            self.slot_resident, self.slot_lookup, self.selected_slots,
            self.slot_missing, self.slot_counts, True,
        )
        self.dispatch["persistent_slots"] += 1
        result = slot_indexed_attention(
            query, self.slot_keys, self.value_cache, self.support_ids,
            self.selected_slots, self.slot_workspace, scale=self.scaling,
        )
        self.dispatch["compressed_v_attention"] += 1
        return result

    @torch.inference_mode()
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        assert past_key_values is None and attention_mask is None
        batch, tokens, _ = hidden_states.shape
        start = self.length
        end = start + tokens
        assert end <= self.capacity and (start == 0 or tokens == 1)
        cos, sin = position_embeddings
        outputs = torch.empty_like(hidden_states)
        if self.mode == "joint" and start == 0:
            self.kernels["prefill_key"] = torch.empty(
                batch, 1, tokens, 128, device="cuda", dtype=torch.bfloat16
            )
        for left in range(0, tokens, 2048):
            right = min(tokens, left + 2048)
            block = hidden_states[:, left:right]
            raw_key = self.k_proj(block).view(batch, right-left, 1, 128).transpose(1, 2)
            _, key = apply_rotary_pos_emb(raw_key, raw_key, cos[:, left:right], sin[:, left:right])
            width = 128 if self.mode == "dense" else 96
            value = self.v_proj(block).view(batch, right-left, 1, width).transpose(1, 2)
            self.store(key, value, cos[:, left:right], sin[:, left:right], start+left)
            if self.mode == "joint" and start == 0:
                self.kernels["prefill_key"][:, :, left:right] = key
            if self.layer_idx == 0:
                self.shared_rope["cos"][start+left:start+right] = cos[0, left:right, :64]
                self.shared_rope["sin"][start+left:start+right] = sin[0, left:right, :64]
        if self.mode == "joint":
            if start == 0:
                append_mapped_host_key(self.host_key, self.kernels["prefill_key"], start=0)
                self.metadata = Metadata(self.kernels["prefill_key"], self.capacity)
            else:
                append_mapped_host_key(self.host_key, key, start=start)
        if start == 0 and self.mode != "dense":
            query_full = torch.empty(batch, 4, tokens, 128, device="cuda", dtype=torch.bfloat16)
            for left in range(0, tokens, 2048):
                right = min(tokens, left + 2048)
                raw_query = self.q_proj(hidden_states[:, left:right]).view(
                    batch, right-left, 4, 128
                ).transpose(1, 2)
                query_full[:, :, left:right], _ = apply_rotary_pos_emb(
                    raw_query, raw_query, cos[:, left:right], sin[:, left:right]
                )
            full_key = self.kernels["prefill_key"] if self.mode == "joint" else self.key_cache[:, :, :end]
            attention_full = compact_prefill(
                query_full, full_key, self.value_cache[:, :, :end]
            )
            self.dispatch["prefill_attention"] += 1
            for left in range(0, tokens, 2048):
                right = min(tokens, left + 2048)
                outputs[:, left:right] = self.project(
                    attention_full[:, :, left:right], decode=False
                )
            self.length = end
            if self.mode == "joint":
                self.kernels["prefill_key"] = None
            return outputs, None
        for left in range(0, tokens, 2048):
            right = min(tokens, left + 2048)
            block = hidden_states[:, left:right]
            raw_query = self.q_proj(block).view(batch, right-left, 4, 128).transpose(1, 2)
            query, _ = apply_rotary_pos_emb(raw_query, raw_query, cos[:, left:right], sin[:, left:right])
            if start == 0:
                result = flash(
                    query, self.key_cache[:, :, :right],
                    self.value_cache[:, :, :right], causal=True,
                )
                self.dispatch["prefill_attention"] += 1
            elif self.mode == "joint":
                result = self.route_and_attend(query, key, end)
            elif self.mode == "als_full":
                self.valid_length.fill_(end)
                result = compressed_v_decode_attention_triton(
                    query, self.key_cache, self.value_cache,
                    scale=self.scaling, valid_sequence_length=self.valid_length,
                )
                self.dispatch["full_decode_attention"] += 1
            else:
                result = flash(query, self.key_cache[:, :, :end],
                               self.value_cache[:, :, :end], causal=False)
                self.dispatch["full_decode_attention"] += 1
            if self.mode == "dense":
                outputs[:, left:right] = self.o_proj(result.transpose(1, 2).reshape(batch, right-left, 512))
            else:
                outputs[:, left:right] = self.project(result, decode=start != 0)
        self.length = end
        return outputs, None


def choose(logits, vocab_size):
    values, ids = logits.max(-1)
    if logits.shape[-1] != vocab_size:
        assert logits.shape[-1] * 8 == vocab_size
        ids = ids + dist.get_rank() * logits.shape[-1]
        values_all = [torch.empty_like(values) for _ in range(8)]
        ids_all = [torch.empty_like(ids) for _ in range(8)]
        dist.all_gather(values_all, values)
        dist.all_gather(ids_all, ids)
        winner = torch.stack(values_all).argmax(0, keepdim=True)
        ids = torch.stack(ids_all).gather(0, winner).squeeze(0)
    return ids


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--mode", choices=("dense", "als_full", "joint"), default="joint")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--discard-first", type=int, default=2)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--cohorts", type=int, default=1)
    parser.add_argument("--engine-warmup", action="store_true")
    parser.add_argument("--warmup-offset", type=int, default=47)
    parser.add_argument("--checkpoint", choices=("retrievalmix_v96", "original_v96"), default="retrievalmix_v96")
    args = parser.parse_args()
    assert args.mode != "joint" or args.checkpoint == "retrievalmix_v96"
    assert args.length >= 4096 and args.batch >= 1 and args.steps >= 2
    assert 0 <= args.discard_first < args.steps - 1 and args.prompt_offset >= 0
    assert args.cohorts >= 1 and args.warmup_offset >= 0
    rank = int(os.environ["LOCAL_RANK"])
    configure_local_caches(rank)
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() in ((8, 0), (8, 6))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    assert dist.get_world_size() == 8
    assets = json.loads(args.assets.read_text())
    assert assets["schema"] == "basisserve.ampere_tp8_assets.v1"
    if rank == 0:
        print(json.dumps({"rank": rank, "phase": "verify_assets"}), flush=True)
        assert all(sha256(Path(x["path"])) == x["sha256"]
                   for x in assets[args.checkpoint]["layers"])
        if args.mode == "joint":
            assert all(sha256(Path(x["path"])) == x["sha256"]
                       for x in assets["router_als40_pcg100_b16r16"]["layers"])
        assert sha256(Path(assets["input_windows"]["path"])) == assets["input_windows"]["sha256"]
    dist.barrier(device_ids=[rank])
    args.output.mkdir(parents=True, exist_ok=True)
    build_dir = args.build_dir or args.output.parent / "build_sm86"
    args.build_dir = build_dir
    kernels = {}
    if args.mode == "joint":
        if rank == 0:
            kernels = load_kernels(build_dir)
        dist.barrier(device_ids=[rank])
        if rank != 0:
            kernels = load_kernels(build_dir)
    communicator = None if args.mode == "dense" else FeatureRaggedCommunicator.from_distributed(
        device=torch.device("cuda", rank)
    )
    print(json.dumps({"rank": rank, "phase": "model_load", "mode": args.mode}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        assets["model"]["path"], torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        tp_plan="auto", local_files_only=True,
    ).eval()
    assert args.length + args.steps - 1 <= model.config.max_position_embeddings
    shared_rope = {
        name: torch.empty(args.length+args.steps, 64, device="cuda", dtype=torch.bfloat16)
        for name in ("cos", "sin")
    }
    layers = []
    for index, layer in enumerate(model.model.layers):
        factor_path = None if args.mode == "dense" else Path(assets[args.checkpoint]["layers"][index]["path"])
        router_path = None if args.mode != "joint" else Path(assets["router_als40_pcg100_b16r16"]["layers"][index]["path"])
        replacement = TP8Attention(
            layer.self_attn, rank=rank, batch=args.batch,
            capacity=args.length+args.steps, mode=args.mode, factor_path=factor_path,
            router_path=router_path, shared_rope=shared_rope,
            communicator=communicator, kernels=kernels,
        )
        layer.self_attn = replacement
        layers.append(replacement)
    windows = load_file(assets["input_windows"]["path"])["input_ids"]
    measured_stop = args.prompt_offset + args.batch * args.cohorts
    assert windows.shape[1] >= args.length and windows.shape[0] >= measured_stop
    if args.engine_warmup:
        assert args.warmup_offset + args.batch <= windows.shape[0]
        assert (args.warmup_offset >= measured_stop
                or args.warmup_offset + args.batch <= args.prompt_offset)
    trials = []
    trial_ids = ([-1] if args.engine_warmup else []) + list(range(args.cohorts))
    for cohort in trial_ids:
        for layer in layers:
            layer.reset_request()
        warmup = cohort == -1
        separate = args.engine_warmup or args.cohorts > 1
        trial_args = argparse.Namespace(**{
            **vars(args), "cohort_id": cohort, "is_engine_warmup": warmup,
            "prompt_offset": args.warmup_offset if warmup else args.prompt_offset + cohort * args.batch,
            "output": args.output / ("warmup" if warmup else f"cohort_{cohort}") if separate else args.output,
        })
        print(json.dumps({"rank": rank, "phase": "trial_start", "cohort_id": cohort,
                          "engine_warmup": warmup, "prompt_offset": trial_args.prompt_offset}), flush=True)
        result = run_trial(trial_args, assets, model, layers, rank, windows)
        if not warmup:
            trials.append(result)
    if rank == 0 and (args.engine_warmup or args.cohorts > 1):
        summary = {
            "status": "complete", "mode": args.mode, "length": args.length, "batch": args.batch,
            "cohorts": args.cohorts, "separate_engine_warmup": args.engine_warmup,
            "conditioning_steps": args.discard_first,
            "measured_decode_steps": args.steps - 1 - args.discard_first,
            "cohort_mean_ms": [row["decode_mean_ms"] for row in trials],
            "median_cohort_mean_ms": statistics.median(row["decode_mean_ms"] for row in trials),
            "cohort_tokens_per_second": [row["steady_tokens_per_second"] for row in trials],
            "prompt_sha256": [row["prompt_sha256"] for row in trials],
            "asset_manifest_sha256": sha256(args.assets), "source_sha256": trials[0]["source_sha256"],
            "router_training_manifest_matches_current": assets["router_training_manifest_matches_current"],
        }
        (args.output / "cohorts.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    dist.barrier(device_ids=[rank])
    if communicator is not None:
        communicator.close()
    dist.destroy_process_group()


@torch.inference_mode()
def run_trial(args, assets, model, layers, rank, windows):
    args.output.mkdir(parents=True, exist_ok=True)
    assert windows.shape[1] >= args.length and windows.shape[0] >= args.prompt_offset + args.batch
    prompt = windows[args.prompt_offset:args.prompt_offset+args.batch, :args.length].long().cuda()
    prompt_hash = hashlib.sha256(prompt.cpu().numpy().tobytes()).hexdigest()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[rank])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print(json.dumps({"rank": rank, "phase": "prefill"}), flush=True)
    output = model(input_ids=prompt, use_cache=False, logits_to_keep=1)
    assert bool(torch.isfinite(output.logits).all())
    token = choose(output.logits, model.config.vocab_size)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    prefill_peak = torch.cuda.max_memory_allocated()
    prefill_reserved = torch.cuda.max_memory_reserved()
    del output, prompt
    decode_ready_allocated = torch.cuda.memory_allocated()
    generated = [token.clone()]
    events = []
    finite = []
    steady_started = None
    torch.cuda.reset_peak_memory_stats()
    dist.barrier(device_ids=[rank])
    print(json.dumps({"rank": rank, "phase": "decode"}), flush=True)
    for step in range(args.steps - 1):
        if step == args.discard_first:
            torch.cuda.synchronize()
            steady_started = time.perf_counter()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        position = torch.full((args.batch, 1), args.length+step, device="cuda", dtype=torch.long)
        output = model(input_ids=token, position_ids=position, use_cache=False, logits_to_keep=1)
        token = choose(output.logits, model.config.vocab_size)
        finite.append(torch.isfinite(output.logits).all())
        generated.append(token.clone())
        end.record()
        events.append((begin, end))
        del output
    torch.cuda.synchronize()
    assert steady_started is not None
    steady_seconds = torch.tensor(time.perf_counter() - steady_started, device="cuda")
    dist.all_reduce(steady_seconds, op=dist.ReduceOp.MAX)
    elapsed = torch.tensor([start.elapsed_time(stop) for start, stop in events], device="cuda")
    local_elapsed = elapsed.cpu().tolist()
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    generated_cuda = torch.cat(generated, -1)
    assert generated_cuda.shape == (args.batch, args.steps) and bool(torch.stack(finite).all())
    all_tokens = [torch.empty_like(generated_cuda) for _ in range(8)]
    dist.all_gather(all_tokens, generated_cuda)
    assert all(bool(torch.equal(generated_cuda, item)) for item in all_tokens)
    generated = generated_cuda.cpu()
    dispatch = Counter()
    for layer in layers:
        dispatch.update(layer.dispatch)
    expected_prefill = 32 * (math.ceil(args.length / 2048) if args.mode == "dense" else 1)
    assert dispatch["prefill_attention"] == expected_prefill
    if args.mode == "joint":
        for name in ("two_stage_router", "select_pack", "persistent_slots", "compressed_v_attention"):
            assert dispatch[name] == 32 * (args.steps - 1)
    else:
        assert dispatch["full_decode_attention"] == 32 * (args.steps - 1)
    if args.mode != "dense":
        assert dispatch["prepared_allgather"] == 32 * math.ceil(args.length / 2048) + 32 * (args.steps - 1)
    measured = elapsed[args.discard_first:].cpu().tolist()
    result = {
        "status": "complete", "rank": rank, "world_size": 8,
        "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "argv": sys.argv,
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "mode": args.mode, "checkpoint": args.checkpoint,
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "length": args.length, "batch": args.batch, "steps": args.steps,
        "build_dir": str(args.build_dir),
        "torch_extensions_dir": os.environ["TORCH_EXTENSIONS_DIR"],
        "prompt_offset": args.prompt_offset,
        "cohort_id": args.cohort_id, "is_engine_warmup": args.is_engine_warmup,
        "separate_engine_warmup": args.engine_warmup,
        "discard_first_decode_steps": args.discard_first,
        "measured_decode_steps": len(measured),
        "measurement_cache_length_before_first_forward": args.length + args.discard_first,
        "measurement_cache_length_after_last_forward": args.length + args.steps - 1,
        "asset_manifest_sha256": sha256(args.assets),
        "prompt_sha256": prompt_hash, "generated_token_ids": generated.tolist(),
        "dispatch": dict(dispatch), "all_logits_finite": bool(torch.stack(finite).all()),
        "prefill_seconds": prefill_seconds, "decode_max_rank_ms": elapsed.cpu().tolist(),
        "decode_local_rank_ms": local_elapsed,
        "decode_mean_ms": statistics.fmean(measured),
        "decode_p50_ms": statistics.median(measured),
        "decode_p95_ms": sorted(measured)[math.ceil(0.95 * len(measured))-1],
        "steady_tokens_per_second": args.batch * len(measured) / float(steady_seconds),
        "prefill_peak_allocated_bytes": prefill_peak,
        "prefill_peak_reserved_bytes": prefill_reserved,
        "decode_ready_allocated_bytes": decode_ready_allocated,
        "decode_final_allocated_bytes": torch.cuda.memory_allocated(),
        "decode_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "decode_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "host_exact_k_capacity_bytes": sum(layer.host_key.numel() * layer.host_key.element_size()
                                           for layer in layers) if args.mode == "joint" else 0,
        "final_step_slot_counts_hits_valid": [layer.slot_counts.cpu().tolist() for layer in layers]
                                             if args.mode == "joint" else None,
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "router_training_manifest_matches_current": assets["router_training_manifest_matches_current"],
    }
    (args.output / f"rank_{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rank": rank, "phase": "complete", "tokens": generated.tolist(),
                      "decode_mean_ms": result["decode_mean_ms"]}), flush=True)
    dist.barrier(device_ids=[rank])
    if rank == 0:
        records = [json.loads((args.output / f"rank_{other}.json").read_text()) for other in range(8)]
        assert {tuple(row["generated_token_ids"][0]) for row in records} == {tuple(generated[0].tolist())}
        assert len({row["prompt_sha256"] for row in records}) == 1
        assert len({json.dumps(row["dispatch"], sort_keys=True) for row in records}) == 1
        assert len({json.dumps(row["source_sha256"], sort_keys=True) for row in records}) == 1
        summary = {"status": "complete", "ranks": 8, "mode": args.mode,
                   "generated_token_ids": generated.tolist(),
                   "decode_mean_ms_max_rank": max(row["decode_mean_ms"] for row in records),
                   "asset_manifest_sha256": sha256(args.assets)}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    dist.barrier(device_ids=[rank])
    return result


if __name__ == "__main__":
    main()
