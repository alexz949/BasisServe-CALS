"""Does a V2Q8 TP1 timing runtime compute what its quality reference computed?

Teacher-forced decode of the same prompt and forced tokens through the reference and the runtime, each in its
own process (--side reference / --side runtime; the BasisKV runtime imports the frozen Table 3 kernel tree, the
reference the repository's quality code), then --side compare. Records, per decode step and layer, each KV head's
selected token set, plus the logits.

  basis_v96*   runtime: v2q8_basis CUDA router (TP1).  reference: paper Table 2 quality path
               (eval_k_routing_ruler.install arm=ours + llama_b16r16_k_offload_p1, Page4, no sink, B2112)
  *_v96 rest   runtime: v2q8_tp1_v96 (preallocated storage).  reference: the same code with DynamicCache and
               the reference LRQKState (selections must be identical; logits bitwise up to kernel nondeterminism)
"""

import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/system"))


def teacher_forced(prefill, decode, prompt, forced, read_selection, steps):
    """Feeds `forced` tokens, or, when it is None, decodes greedily from the prompt; returns the fed tokens."""
    logits, selections, fed = [], [], []
    token = prefill(prompt)[0, -1].argmax().view(1, 1)
    for step in range(steps):
        if forced is not None:
            token = forced[step].view(1, 1)
        fed.append(int(token))
        output = decode(token)[0, -1].float()
        logits.append(output.cpu())
        selections.append(read_selection())
        token = output.argmax().view(1, 1)
    return torch.stack(logits), selections, fed


def reference_basis(args, prompt, forced):
    from transformers import AutoModelForCausalLM
    from evaluation import eval_k_routing_ruler as ruler
    from evaluation import llama_b16r16_k_offload_p1 as offload
    from evaluation import llama_sink_recent_routing_p16 as routing
    from evaluation.chunked_prefill_mlp import install_chunked_prefill_mlps
    from benchmarks.system import audit_tp1_request as frozen

    offload.PAGE_SIZE, offload.PINNED_PAGES, offload.BUDGET = 4, 0, 2112
    model = AutoModelForCausalLM.from_pretrained(frozen.MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 local_files_only=True).cuda().eval()
    install_chunked_prefill_mlps(model)
    manifest = json.loads((args.value_checkpoint / "manifest.json").read_text())
    bank = {entry["layer"]: load_file(str(args.bank_v96 / f"layer_{entry['layer']:03d}.safetensors"))
            for entry in manifest["layers"]}
    ruler.install(model, args.value_checkpoint, manifest, "ours", bank)
    offload.install(model)
    cache = ruler.RoutingCache(model.config)
    step = []
    original = routing.page_support

    def recorded(scores, **keywords):
        ids, valid = original(scores, **keywords)
        step.append(ids.masked_fill(~valid, -1).cpu())
        return ids, valid

    routing.page_support = recorded

    def prefill(tokens):
        return model(input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1).logits

    def decode(token):
        mask = torch.ones(1, 1, 1, cache.get_seq_length() + 1, device="cuda", dtype=torch.bool)
        return model(input_ids=token, past_key_values=cache, attention_mask=mask, use_cache=True,
                     logits_to_keep=1).logits

    def read():
        layers = list(step)
        step.clear()
        return layers

    result = teacher_forced(prefill, decode, prompt, forced, read, args.steps)
    routing.page_support = original
    del model, cache
    return result


def runtime_basis(args, prompt, forced):
    from bench_v2q8_tp1 import build_runtime, fresh

    runtime = build_runtime(args)
    fresh(runtime)

    def read():
        return [layer.support_ids.clone().cpu() for layer in runtime.state()]

    result = teacher_forced(runtime.prefill, runtime.decode, prompt, forced, read, args.steps)
    del runtime
    return result


def v96(args, prompt, forced, preallocated):
    from benchmarks.system import v2q8_tp1_v96
    from benchmarks.system.audit_tp1_request import MODEL

    runtime = v2q8_tp1_v96.build(args.arm, model_path=MODEL, value_checkpoint=args.value_checkpoint,
                                 capacity=args.length + args.steps, preallocated=preallocated)
    runtime.reset()

    def read():
        # LRQK selects per query head (32 rows); ShadowKV per KV head (8 rows); Full-K has no selection.
        name = {"lrqk_v96": "selected", "shadowkv_v96": "selected_ids"}.get(args.arm)
        return [getattr(state, name).cpu() for _, state in sorted(runtime.states().items())] if name else []

    result = teacher_forced(runtime.prefill, runtime.decode, prompt, forced, read, args.steps)
    del runtime
    return result


def compare(reference, runtime):
    (expected_logits, expected), (actual_logits, actual) = reference, runtime
    rows, exact, jaccard = [], 0, []
    for step, (layers_expected, layers_actual) in enumerate(zip(expected, actual, strict=True)):
        for layer, (left, right) in enumerate(zip(layers_expected, layers_actual, strict=True)):
            for head in range(left.shape[1]):
                a = set(left[0, head][left[0, head] >= 0].tolist())
                b = set(right[0, head][right[0, head] >= 0].tolist())
                exact += a == b
                jaccard.append(len(a & b) / max(1, len(a | b)))
                if a != b and len(rows) < 64:
                    rows.append(dict(step=step, layer=layer, head=head, reference=len(a), runtime=len(b),
                                     shared=len(a & b)))
    difference = (expected_logits - actual_logits).abs()
    total = len(jaccard)
    return dict(sets_compared=total, identical_sets=exact, identical_fraction=exact / total if total else None,
        mean_jaccard=sum(jaccard) / total if total else None, min_jaccard=min(jaccard) if total else None,
        first_mismatches=rows, logits_max_abs_difference=float(difference.max()),
        logits_mean_abs_difference=float(difference.mean()),
        argmax_agreement=float((expected_logits.argmax(-1) == actual_logits.argmax(-1)).float().mean()))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True,
                        choices=("basis_v96", "basis_v96_reuse", "basis_v96_resident", "fullk_v96", "lrqk_v96", "shadowkv_v96"))
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--cohort", type=int, default=0)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--bank-v96", type=Path, required=True)
    parser.add_argument("--bank-v128", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--side", choices=("reference", "runtime", "compare"), required=True)
    parser.add_argument("--ruler-prompts", type=Path,
                        help="paper Table 2 RULER-128K prompts.safetensors; with --ruler-index the prompt is that "
                             "sample, the reference decodes greedily and the runtime is forced with its tokens")
    parser.add_argument("--ruler-index", type=int)
    args = parser.parse_args()
    args.mode, args.output_tokens = "check", args.steps
    args.output.mkdir(parents=True, exist_ok=True)
    if args.side == "compare":
        sides = [torch.load(args.output / f"{side}.pt") for side in ("reference", "runtime")]
        scope = ("runtime CUDA router vs paper Table 2 quality path (Python page_support, float32 scores)"
                 if args.arm.startswith("basis") else
                 "timing runtime (preallocated storage, FlashAttention prefill) vs the quality code path "
                 "(DynamicCache, reference LRQKState, Triton prefill)")
        result = dict(arm=args.arm, context_tokens=sides[0]["length"], forced_steps=args.steps, cohort=args.cohort,
                      ruler_index=args.ruler_index, scope=scope, fed_tokens=sides[0]["tokens"],
                      **compare(*[(side["logits"], side["selections"]) for side in sides]))
        (args.output / "selection_check.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "first_mismatches"}), flush=True)
        return
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    if args.ruler_prompts:
        prompt = load_file(str(args.ruler_prompts))[str(args.ruler_index)][None].long().cuda()
        args.length = prompt.shape[1]
        forced = (None if args.side == "reference" else
                  torch.tensor(torch.load(args.output / "reference.pt")["tokens"], device="cuda"))
    else:
        inputs = load_file(str(args.inputs))
        prompt = inputs[f"cohort_{args.cohort}"][:args.length][None].long().cuda()
        forced = inputs[f"cohort_{(args.cohort + 1) % 3}"][:args.steps].long().cuda()
    if args.arm.startswith("basis"):
        logits, selections, tokens = (reference_basis if args.side == "reference" else runtime_basis)(args, prompt, forced)
    else:
        logits, selections, tokens = v96(args, prompt, forced, preallocated=args.side == "runtime")
    torch.save(dict(logits=logits, selections=selections, tokens=tokens, length=args.length),
               args.output / f"{args.side}.pt")
    print(json.dumps(dict(side=args.side, steps=len(selections), layers=len(selections[0]) if selections else 0)),
          flush=True)


if __name__ == "__main__":
    main()
