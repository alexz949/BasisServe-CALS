"""Compare the full STAR fused checkpoint's single-GPU output with TP8 smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from safetensors.torch import load_file
import torch
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.core.qwen3_tp8_v_only import StarQwen3ForCausalLM
from benchmarks.system.bench_qwen3_8b_tp8_v_only_memory import DEFAULT_STAR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STAR)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--tp8-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    config.star_value_ranks = json.loads((args.checkpoint / "result.json").read_text())["budget"]["ranks"]
    state = torch.load(args.checkpoint / "fused.pt", map_location="cpu", mmap=True, weights_only=True)
    model, loading_info = StarQwen3ForCausalLM.from_pretrained(
        None, config=config, state_dict=state, dtype=torch.bfloat16,
        device_map="cuda:0", output_loading_info=True,
    )
    assert not loading_info["missing_keys"] and not loading_info["unexpected_keys"]
    model.eval()
    del state
    ids = load_file(str(args.tokens))["input_ids"][:1].long().cuda()
    with torch.inference_mode():
        first = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        token0 = first.logits.argmax(dim=-1)
        second = model(input_ids=token0, past_key_values=first.past_key_values,
                       use_cache=True, logits_to_keep=1)
        token1 = second.logits.argmax(dim=-1)
    reference = [token0.flatten().cpu().tolist(), token1.flatten().cpu().tolist()]
    tp8 = json.loads(args.tp8_result.read_text())["generated_token_ids"]
    payload = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "prompt_tokens": ids.shape[1],
        "reference_tokens": reference,
        "tp8_tokens": tp8,
        "tokens_match": reference == tp8,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)
    assert payload["tokens_match"]


if __name__ == "__main__":
    main()
