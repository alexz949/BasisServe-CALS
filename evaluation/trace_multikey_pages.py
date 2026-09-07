"""Three-arm, same-state selected-page audit of fixed RULER regressions."""

import argparse
import json
import math
from pathlib import Path
import re
import shlex
import sys
import time
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import install_qwen3_gqa_vo_als_export, _memory_bounded_gqa_sdpa
from basisserve.core.c1_conditional_page_attention import (
    c1_conditional_page_topk_attention, _selected_pages, C1ConditionalPageAttentionResult,
)
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.residual_kl_replay import prefix_signature
from evaluation.eval_qwen3_8b_residual_rank_ruler import (
    TASKS, protocol, prepare_arm, greedy_decode, full_attention, load_bank,
    _build_work, _eos_ids, parse_tasks, ruler_prompt, sample_score,
)
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json

ARMS = ("full_exact_k", "exact_pages", "q32_proxy")


def page_snapshot(scores, valid, *, kv_heads, page_size, page_budget, pinned_prefix_pages):
    """Native selected IDs plus full per-head mass and routed-score metadata."""
    assert scores.shape[0] == scores.shape[2] == 1 and valid.all()
    heads, length = scores.shape[1], scores.shape[-1]
    pages = math.ceil(length / page_size)
    ids, chosen_valid = _selected_pages(scores, valid, kv_heads=kv_heads, page_size=page_size,
                                       page_budget=page_budget, pinned_prefix_pages=pinned_prefix_pages)
    assert chosen_valid.all() and ids.shape[-1] == page_budget
    assert (ids[..., 0] == 0).all() and (ids.sort(-1).values.diff(dim=-1) > 0).all()
    padded = torch.nn.functional.pad(scores.float(), (0, pages * page_size - length), value=-torch.inf)
    lse = padded.reshape(kv_heads, heads // kv_heads, pages, page_size).logsumexp(-1)
    mass = lse.softmax(-1)
    routed = lse.clone()
    routed[..., :pinned_prefix_pages] = -torch.inf
    group_score, owner = routed.softmax(-1).max(1)
    group_score[:, :pinned_prefix_pages] = -torch.inf
    sorted_score = group_score[:, pinned_prefix_pages:].sort(-1).values.contiguous()
    rank = torch.zeros_like(group_score, dtype=torch.int32)
    rank[:, pinned_prefix_pages:] = (sorted_score.shape[-1] - torch.searchsorted(
        sorted_score, group_score[:, pinned_prefix_pages:].contiguous(), right=True).int() + 1)
    return dict(ids=ids[0, :, 0], full_mass=mass, group_score=group_score,
                owner=owner.short(), rank_min=rank.short(),
                cutoff=sorted_score[:, -(page_budget-pinned_prefix_pages)])


def support_spans(tokenizer, source, old):
    prompt = ruler_prompt(source)
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    assert len(encoded["input_ids"]) == old["prompt_tokens"]
    spans = []
    answer = source["outputs"][0]
    wrong = re.search(r"\d+", old["arms"]["uniform_r8"]["prediction"]).group()
    for kind, text in (("answer", answer), ("old_q32_wrong_answer", wrong)):
        for match in re.finditer(re.escape(text), prompt):
            start = prompt.rfind("\n", 0, match.start()) + 1
            stop = prompt.find("\n", match.end())
            assert stop >= match.end()
            for span_kind, a, b in ((kind, match.start(), match.end()), (kind + "_sentence", start, stop)):
                tokens = [i for i, (x, y) in enumerate(encoded["offset_mapping"]) if y > a and x < b]
                spans.append({"kind": span_kind, "text": prompt[a:b], "token_start": min(tokens),
                              "token_stop": max(tokens)+1, "pages": sorted({i//32 for i in tokens})})
    assert sum(s["kind"] == "answer" for s in spans) == 1
    return spans


class PageTrace:
    """Diagnostic wrapper; only exact_pages replaces the selector's returned IDs."""
    def __init__(self, arm, spans):
        assert arm in ARMS
        self.arm, self.spans = arm, spans
        self.calls = 0
        self.tensors = {}
        self.rows = []

    def __call__(self, query, key, value, sidecar, projector, **options):
        assert query.shape == (1, 32, 1, 128) and key.shape[1] == 8
        assert options["attention_mask"] is not None and options["attention_mask"].all()
        assert options["page_size"] == 32 and options["exact_token_budget"] == 2048
        assert options["pinned_prefix_pages"] == 1 and not options["collect_statistics"]
        layer = self.calls % 36
        step = self.calls // 36 + 1
        self.calls += 1
        exact_scores = torch.einsum("ghd,gtd->ght", query[0, :, 0].float().reshape(8, 4, 128),
                                    key[0].float()).reshape(1, 32, 1, -1) * options["scale"]
        observed = []

        def select(proxy_scores, valid, **selection_options):
            assert not observed
            exact = page_snapshot(exact_scores, valid, **selection_options)
            proxy = page_snapshot(proxy_scores, valid, **selection_options)
            observed.append(True)
            exact_ids, proxy_ids = exact["ids"], proxy["ids"]
            overlap = (exact_ids[:, :, None] == proxy_ids[:, None, :]).any(-1).sum(-1)
            key_prefix = f"step{step:03d}.layer{layer:02d}"
            for selector, snapshot in (("exact", exact), ("proxy", proxy)):
                for name, tensor in snapshot.items():
                    self.tensors[f"{key_prefix}.{selector}.{name}"] = tensor.detach().contiguous().cpu()
            self.tensors[f"{key_prefix}.intersection_count"] = overlap.cpu().short()
            row = {"layer": layer, "decode_forward": step, "predicts_generated_token_1based": step+1,
                   "query_position": key.shape[-2]-1, "mean_page_overlap": float(overlap.float().mean()/64),
                   "support": []}
            for span in self.spans:
                for page in span["pages"]:
                    row["support"].append({"kind": span["kind"], "page": page,
                        "exact_selected_groups": torch.where((exact_ids == page).any(-1))[0].cpu().tolist(),
                        "proxy_selected_groups": torch.where((proxy_ids == page).any(-1))[0].cpu().tolist(),
                        "teacher_mass_by_head": exact["full_mass"][:, :, page].cpu().tolist(),
                        "exact_rank_min": exact["rank_min"][:, page].cpu().tolist(),
                        "proxy_rank_min": proxy["rank_min"][:, page].cpu().tolist()})
            self.rows.append(row)
            chosen = exact_ids if self.arm == "exact_pages" else proxy_ids
            return chosen[None, :, None], torch.ones_like(chosen[None, :, None], dtype=torch.bool)

        with patch("basisserve.core.c1_conditional_page_attention._selected_pages", new=select):
            output = c1_conditional_page_topk_attention(query, key, value, sidecar, projector, **options)
        assert observed == [True]
        if self.arm == "full_exact_k":
            full = _memory_bounded_gqa_sdpa(query, key, value, attention_mask=options["attention_mask"],
                                           dropout_p=0.0, is_causal=False, scale=options["scale"])
            return C1ConditionalPageAttentionResult(output=full, statistics={})
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "evaluate"), required=True)
    parser.add_argument("--sample-index", type=int, choices=(33, 37, 38), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--c1-checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8")
    parser.add_argument("--old-results", type=Path, default=ROOT / "results/evaluation/mse_base_q32_ruler32k")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.samples_per_task, args.sequence_length = 8, 32768
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA L40S"
    output_dir = args.output_dir / args.stage / f"sample_{args.sample_index:03d}"
    assert not output_dir.exists() or not any(output_dir.iterdir())
    old_file = args.old_results / "evaluate" / f"sample_{args.sample_index:03d}.json"
    previous = json.loads(old_file.read_text())
    assert previous["status"] == "complete"
    settings = protocol(args)
    for field in ("c1_layer_sha256", "bank_sha256", "dataset_manifest_sha256", "model", "c1_checkpoint"):
        assert previous["protocol"][field] == settings[field], field
    work = _build_work(args.data_dir, parse_tasks(TASKS), 8)
    index, task, ordinal, source = work[args.sample_index]
    old = previous["result"]
    assert index == args.sample_index and old["key"] == f"{task.name}:{ordinal}"
    assert source["outputs"] == old["references"] and source["index"] == old["source_index"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    spans = support_spans(tokenizer, source, old)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True,
              low_cpu_mem_usage=True, attn_implementation="sdpa").to("cuda:0").eval()
    install_qwen3_gqa_vo_als_export(model, args.c1_checkpoint, attention_backend="triton")
    bank = load_bank(args.bank)
    modules = [layer.self_attn for layer in model.model.layers]
    assert len(modules) == 36
    full_attention(modules, "triton")
    tokens = tokenizer(ruler_prompt(source), add_special_tokens=True, return_tensors="pt")["input_ids"].to("cuda:0")
    assert tokens.shape[-1] == old["prompt_tokens"]
    prefix = RoutingDynamicCache()
    prefill = model(input_ids=tokens, past_key_values=prefix, use_cache=True, logits_to_keep=1)
    first = int(prefill.logits[0, -1].argmax())
    assert first == old["first_token_from_shared_prefill"]
    del prefill, tokens
    signature = prefix_signature(prefix)
    maximum = 8 if args.stage == "smoke" else task.tokens_to_generate
    eos = _eos_ids(tokenizer, model)
    smoke_reference = {}
    if args.stage == "smoke":
        for arm, ranks in (("full_exact_k", None), ("q32_proxy", [8]*36)):
            cache = prepare_arm(model, bank, prefix, ranks)
            ids, cache, logits = greedy_decode(model, cache, first, maximum_tokens=maximum, eos_ids=eos, trace=True)
            smoke_reference[arm] = (ids, logits)
            del cache
            assert prefix_signature(prefix) == signature
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for arm in ARMS:
        cache = prepare_arm(model, bank, prefix, [8]*36)
        audit = PageTrace(arm, spans)
        arm_start = time.monotonic()
        with patch("basisserve.checkpoint.gqa_vo_qwen3.c1_conditional_page_topk_attention", new=audit):
            ids, cache, logits = greedy_decode(model, cache, first, maximum_tokens=maximum, eos_ids=eos, trace=True)
        assert audit.calls == 36*(len(ids)-1)
        assert prefix_signature(prefix) == signature
        assert cache.get_seq_length() == prefix.get_seq_length()+len(ids)-1
        matches_old = None
        if arm != "exact_pages":
            old_arm = "c1_exact_k" if arm == "full_exact_k" else "uniform_r8"
            expected = old["arms"][old_arm]["generated_token_ids"]
            assert ids == expected[:maximum], (arm, ids, expected)
            matches_old = True
        if arm in smoke_reference:
            reference_ids, reference_logits = smoke_reference[arm]
            assert ids == reference_ids and len(logits) == len(reference_logits)
            for a, b in zip(logits, reference_logits, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        audit.tensors["generated_token_ids"] = torch.tensor(ids, dtype=torch.int32)
        audit.tensors["decode_logits"] = torch.stack(logits)
        path = output_dir / f"{arm}.safetensors"
        save_file(audit.tensors, str(path))
        prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        record = {"arm": arm, "generated_token_ids": ids, "prediction": prediction,
                  "score": sample_score(prediction, source["outputs"], task.match_type),
                  "stopped_on_eos": ids[-1] in eos, "matches_old_tokens": matches_old,
                  "untraced_logits_bitwise_equal": True if arm in smoke_reference else None,
                  "calls": audit.calls, "rows": audit.rows, "tensor_sha256": sha256(path),
                  "elapsed_seconds": time.monotonic()-arm_start}
        write_json(output_dir / f"{arm}.json", record)
        records[arm] = {k:v for k,v in record.items() if k != "rows"}
        print(f"[{index} {arm}] score={record['score']} prediction={prediction!r} calls={audit.calls}", flush=True)
        del cache, audit, logits
    write_json(output_dir / "result.json", {"status": "complete", "stage": args.stage,
        "sample_index": index, "source_index": source["index"], "references": source["outputs"],
        "prompt_tokens": old["prompt_tokens"], "support_spans": spans, "arms": records,
        "source_protocol": settings, "old_result_sha256": sha256(old_file),
        "reference_precision": "FP32 exact QK selector; native BF16 proxy and selected-attention payload",
        "scope": "each arm follows its own greedy trajectory; exact/proxy sets within a row share Q/K; no refit",
        "full_arm": "actual full SDPA output; sparse output discarded after observational tracing",
        "smoke": "8 generated tokens, not official accuracy" if args.stage == "smoke" else None,
        "command": shlex.join(sys.argv), "python": sys.executable, "gpu": torch.cuda.get_device_name(0),
        "code_sha256": sha256(Path(__file__)), "wall_seconds": time.monotonic()-started})


if __name__ == "__main__":
    with torch.inference_mode():
        main()
