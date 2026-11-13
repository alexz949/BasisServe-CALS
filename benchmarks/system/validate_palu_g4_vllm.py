"""Numerical check of the vLLM PaLU G-LRD4 deployments against a single-GPU reference.

reference: HF Transformers Llama with V_k = D_k E_g x (dense or PaLU) or the BasisKV
           coordinates E_k^T V_k with per-head output decoders; teacher-forced
           log-softmax at the tail positions plus a greedy continuation.
serve:     vLLM TP run of an arm; prompt log-probabilities at the same positions (teacher
           forced by construction) plus the same greedy continuation.
compare:   top-1 agreement, log-probability differences and continuation agreement. Dense
           versus dense calibrates the ordinary HF/vLLM BF16 gap; PaLU versus PaLU must sit in
           the same range.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PROMPT, TAIL, CONTINUATION = 512, 32, 32
MODEL = Path(os.environ.get("BENCH_MODEL", "/home/yl1139/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct"
                                           "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"))


def windows(path, count):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_slice("input_ids")[32:32 + count, :PROMPT + TAIL].long()


def palu_value_projection(factors, manifest, layer):
    """Dense V rows represented by the group writers and decoders of one layer."""
    ranks = manifest["compression"]["layer_ranks"][layer]
    writer = factors[f"layers.{layer}.v_writer.weight"].float()
    decoder = factors[f"layers.{layer}.v_decoder.weight"].float()
    rows, offset = [], 0
    for group, rank in enumerate(ranks):
        rows.append(decoder[group, :, :rank] @ writer[offset:offset + rank])
        offset += rank
    return torch.cat(rows)


def basis_projections(root, record, value):
    """Dense-shaped V and O weights of one BasisKV layer: rank-r coordinates in the first
    r value features of every KV head, the output decoder in the matching o_proj columns."""
    factors = load_file(str(root / record["file"]))
    ranks = record.get("ranks") or factors["source_ranks"].tolist()
    assert len(set(ranks)) == 1
    rank = ranks[0]
    encoders = factors["value_coordinate_encoders"].float()[..., :rank]
    decoders = factors["head_output_decoders"].float()[:, :rank]
    folded_value = torch.zeros(8, 128, 4096)
    folded_value[:, :rank] = encoders.transpose(1, 2) @ value.float().cpu().view(8, 128, 4096)
    folded_output = torch.zeros(4096, 32, 128)
    folded_output[:, :, :rank] = decoders.permute(2, 0, 1)
    return folded_value.view(1024, 4096), folded_output.view(4096, 4096)


def reference(args):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.bfloat16,
                                                 local_files_only=True).cuda().eval()
    record = dict(variant=args.variant, prompt_tokens=PROMPT, tail=TAIL)
    if args.variant == "palu":
        manifest = json.loads((args.factors / "manifest.json").read_text())
        factors = load_file(str(args.factors / manifest["artifact"]["file"]))
        record["factor_sha256"] = manifest["artifact"]["sha256"]
        for index, layer in enumerate(model.model.layers):
            layer.self_attn.v_proj.weight.data.copy_(palu_value_projection(factors, manifest, index))
    if args.variant == "basis":
        manifest_bytes = (args.factors / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        record["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        record["layer_ranks"] = [layer["ranks"] for layer in manifest["layers"]]
        for index, layer in enumerate(model.model.layers):
            value, output = basis_projections(args.factors, manifest["layers"][index],
                                              layer.self_attn.v_proj.weight.data)
            layer.self_attn.v_proj.weight.data.copy_(value)
            layer.self_attn.o_proj.weight.data.copy_(output)
    prompts = windows(args.windows, args.prompts).cuda()
    logprobs, continuations = [], []
    with torch.inference_mode():
        for prompt in prompts:
            logits = model(prompt[None]).logits[0, -TAIL - 1:-1].float()
            logprobs.append(torch.log_softmax(logits, dim=-1))
            generated = model.generate(prompt[None], max_new_tokens=CONTINUATION,
                                       min_new_tokens=CONTINUATION, do_sample=False,
                                       pad_token_id=0)[0, prompt.numel():]
            continuations.append(generated.cpu())
    save_file(dict(logprobs=torch.stack(logprobs).cpu(), continuations=torch.stack(continuations),
                   prompts=prompts.cpu()), str(args.output.with_suffix(".safetensors")))
    record["status"] = "complete"
    args.output.write_text(json.dumps(record, indent=2) + "\n")


def serve(args):
    from vllm import LLM, SamplingParams
    from basisserve.vllm import register
    register()
    overrides = {}
    if args.arm.startswith("palu_g4"):
        manifest_bytes = (args.factors / "manifest.json").read_bytes()
        architecture = {"palu_g4": "BasisServeLlamaPaluG4ForCausalLM",
                        "palu_g4_sharded": "BasisServeLlamaPaluG4ShardedForCausalLM"}[args.arm]
        overrides = dict(architectures=[architecture],
                         basisserve_palu_factor_dir=str(args.factors),
                         basisserve_palu_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
    if args.arm == "als_full":
        manifest_bytes = (args.factors / "manifest.json").read_bytes()
        overrides = dict(architectures=["BasisServeLlamaJointALSForCausalLM"],
                         basisserve_joint_factor_dir=str(args.factors),
                         basisserve_joint_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
    llm = LLM(model=str(MODEL), tensor_parallel_size=args.tp, dtype="bfloat16",
              distributed_executor_backend="mp", max_model_len=PROMPT + TAIL + CONTINUATION + 16,
              max_num_seqs=4, max_num_batched_tokens=8192, enable_chunked_prefill=True,
              enable_prefix_caching=False, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
              disable_custom_all_reduce=True, seed=0, hf_overrides=overrides, block_size=32,
              worker_extension_cls="benchmarks.system.vllm_smoke_worker.LlamaSmokeWorker")
    prompts = windows(args.windows, args.prompts)
    tail_logprobs = torch.full((len(prompts), TAIL), float("nan"))
    tail_top1 = torch.full((len(prompts), TAIL), -1, dtype=torch.long)
    continuations = []
    scoring = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1, detokenize=False)
    greedy = SamplingParams(temperature=0, max_tokens=CONTINUATION, ignore_eos=True, detokenize=False)
    for index, prompt in enumerate(prompts):
        tokens = prompt.tolist()
        scored = llm.generate([{"prompt_token_ids": tokens}], scoring, use_tqdm=False)[0]
        for offset in range(TAIL):
            position = PROMPT + offset  # log-probabilities of token `position` given its prefix
            entry = scored.prompt_logprobs[position]
            tail_logprobs[index, offset] = entry[tokens[position]].logprob
            tail_top1[index, offset] = min(entry.items(), key=lambda item: item[1].rank)[0]
        generated = llm.generate([{"prompt_token_ids": tokens}], greedy, use_tqdm=False)[0]
        continuations.append(torch.tensor(list(generated.outputs[0].token_ids)))
    workers = llm.collective_rpc("llama_smoke_statistics")
    save_file(dict(tail_logprobs=tail_logprobs, tail_top1=tail_top1,
                   continuations=torch.stack(continuations), prompts=prompts),
              str(args.output.with_suffix(".safetensors")))
    args.output.write_text(json.dumps(dict(arm=args.arm, tp=args.tp, status="complete", workers=workers,
                                           gpu_memory_utilization=args.gpu_memory_utilization,
                                           hf_overrides=overrides), indent=2) + "\n")


def compare(args):
    ref = load_file(str(args.reference.with_suffix(".safetensors")))
    run = load_file(str(args.served.with_suffix(".safetensors")))
    prompts = ref["prompts"]
    actual = prompts[:, PROMPT:PROMPT + TAIL]
    ref_logprob = torch.gather(ref["logprobs"], 2, actual[..., None])[..., 0]
    ref_top1 = ref["logprobs"].argmax(-1)
    same_continuation = (ref["continuations"] == run["continuations"])
    first_divergence = [int(row.numel()) if row.all() else int((~row).nonzero()[0]) for row in same_continuation]
    result = dict(reference=str(args.reference), served=str(args.served),
        all_finite=bool(torch.isfinite(run["tail_logprobs"]).all() and torch.isfinite(ref_logprob).all()),
        same_prompts=bool(torch.equal(prompts, run["prompts"])), prompts=int(prompts.shape[0]),
        tail_positions=TAIL, teacher_forced_top1_agreement=float((ref_top1 == run["tail_top1"]).float().mean()),
        teacher_forced_logprob_max_abs_difference=float((ref_logprob - run["tail_logprobs"]).abs().max()),
        teacher_forced_logprob_mean_abs_difference=float((ref_logprob - run["tail_logprobs"]).abs().mean()),
        greedy_continuation_tokens=CONTINUATION,
        greedy_continuation_agreement=float(same_continuation.float().mean()),
        greedy_first_divergence_per_prompt=first_divergence)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("reference", "serve", "compare"))
    parser.add_argument("--variant", choices=("dense", "palu", "basis"), default="dense")
    parser.add_argument("--arm", choices=("dense", "als_full", "palu_g4", "palu_g4_sharded"), default="dense")
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.6)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--served", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    {"reference": reference, "serve": serve, "compare": compare}[args.stage](args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
