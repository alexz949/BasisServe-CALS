"""RULER-128K quality of a V2Q8 TP1 *timing* runtime, paired with the paper Table 2 archive.

Runs the exact runtime that the TP1 benchmark times (bench_v2q8_tp1.build_runtime) on the paper Table 2 prompts
(HF evaluation/llama31-8b-instruct-ruler128k-1100-seed42; prompts.json sha256 1f6389dd... = Table 2
prompts_sha256), with Table 2's generation rule: greedy, stop at a native EOS or the task's token cap, decode with
skip_special_tokens, official RULER substring scoring. Each sample is written as it completes (resumable), with
the archived Table 2 record for the same sample alongside for pairing.

  python benchmarks/system/bench_v2q8_tp1_ruler.py --arm basis_v96 --shard 0 --shards 4 --output DIR ...
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/system"))

ARCHIVE = {"basis_v96": "b16r16_page4_fisher", "basis_v96_reuse": "b16r16_page4_fisher",
           "basis_v96_resident": "b16r16_page4_fisher", "fullk_v96": "full", "lrqk_v96": "lrqk",
           "shadowkv_v96": "shadowkv"}
EOS = (128001, 128008, 128009)


@torch.inference_mode()
def main():
    from transformers import AutoTokenizer
    from evaluation.ruler_v1 import sample_score
    from bench_v2q8_tp1 import build_runtime, fresh
    from benchmarks.system.audit_tp1_request import MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARCHIVE), required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="directory with prompts.json/.safetensors")
    parser.add_argument("--archive", type=Path, default=ROOT / "results/l31-v96-retrievalmix-ruler1100/predictions")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-ordinal", type=int, help="keep the first N samples of every task")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--bank-v96", type=Path, required=True)
    parser.add_argument("--bank-v128", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.prompts / "prompts.json").read_text())["rows"]
    rows = [r for r in rows if args.max_ordinal is None or r["ordinal"] < args.max_ordinal][args.shard::args.shards]
    archive = {}
    for line in (args.archive / f"{ARCHIVE[args.arm]}.jsonl").read_text().splitlines():
        record = json.loads(line)
        archive[record["sample"]["index"]] = record
    tokens = load_file(str(args.prompts / "prompts.safetensors"))
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    args.mode, args.length = "ruler", max(r["input_tokens"] for r in rows)
    args.output_tokens = max(r["maximum_tokens"] for r in rows)
    runtime = build_runtime(args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = args.output / f"sample_{row['index']:04d}.json"
        if path.exists():
            continue
        prompt = tokens[str(row["index"])][None].long().cuda()
        assert prompt.shape[1] == row["input_tokens"]
        fresh(runtime)
        begin = time.perf_counter()
        logits = runtime.prefill(prompt)
        ids = [int(logits[0, -1].argmax())]
        while len(ids) < row["maximum_tokens"] and ids[-1] not in EOS:
            ids.append(int(runtime.decode(torch.tensor([[ids[-1]]], device="cuda"))[0, -1].argmax()))
        torch.cuda.synchronize()
        prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        archived = archive[row["index"]]["result"]
        result = dict(sample=row, arm=args.arm, ids=ids, prediction=prediction, stopped=ids[-1] in EOS,
                      score=sample_score(prediction, row["answers"], row["match_type"]),
                      seconds=time.perf_counter() - begin, configuration=runtime.configuration,
                      archived=dict(arm=ARCHIVE[args.arm], ids=archived["ids"], score=archived["score"]),
                      same_ids_as_archive=ids == archived["ids"], same_first_token=ids[0] == archived["ids"][0])
        path.write_text(json.dumps(result) + "\n")
        print(json.dumps(dict(index=row["index"], task=row["task"], score=result["score"],
                              archived=result["archived"]["score"], same_ids=result["same_ids_as_archive"],
                              seconds=round(result["seconds"], 1))), flush=True)


if __name__ == "__main__":
    main()
