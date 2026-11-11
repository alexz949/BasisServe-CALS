"""Dense Llama-3.1-8B under HF tensor parallelism vs one GPU: relative logits error at the positions the ShadowKV
model smoke scores (teacher-forced on its reference tokens). This is the TP numerical noise floor that the ShadowKV
TP adapter's logits error is judged against. Run with torchrun --nproc-per-node 4 (or 8).
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, DistributedConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--model-check", type=Path, required=True, help="model_reference.json of the ShadowKV smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    generated = json.loads(args.model_check.read_text())["reference_generated_ids"]
    ids = torch.cat((load_file(args.tokens)["input_ids"].to(device), torch.tensor([generated[:-1]], device=device)), 1)
    with torch.inference_mode():
        tp = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
            distributed_config=DistributedConfig(tp_size=dist.get_world_size()), local_files_only=True).eval()
        tp_logits = tp(input_ids=ids, use_cache=False, logits_to_keep=len(generated)).logits[0].float()
        tp_logits = tp_logits.full_tensor() if hasattr(tp_logits, "full_tensor") else tp_logits
        del tp
        torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            one = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                       device_map={"": str(device)}, local_files_only=True).eval()
            reference = one(input_ids=ids, use_cache=False, logits_to_keep=len(generated)).logits[0].float()
            relative = ((tp_logits - reference).norm(dim=-1) / reference.norm(dim=-1)).tolist()
            result = dict(status="complete", world_size=dist.get_world_size(), positions=len(generated),
                          relative_error_per_position=relative, maximum_relative_error=max(relative),
                          top1_equal=bool(torch.equal(tp_logits.argmax(-1), reference.argmax(-1))),
                          scope="Dense HF TP vs one GPU, BF16, SDPA, teacher-forced on the ShadowKV smoke tokens")
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
