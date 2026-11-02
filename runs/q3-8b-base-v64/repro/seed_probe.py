"""Find whether a seed reproduces the paper's qwen3_8b_c4_512f128h_s4096 document order.

The paper's sampler (evaluation/build_llama31_8b_palu_m_checkpoint._prepare_windows): stream allenai/c4 en train at revision
1588ec4, shuffle(seed, buffer 10000), keep documents with >= L tokens (no special tokens), dedupe by url, one window each.
Known fingerprint: the urls of usable documents 320..463 (profile and confirmation windows).
Usage: seed_probe.py SEED [SEQUENCE_LENGTH]
"""
import json
import sys
import time

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer

seed, length = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 4096
model = "/zpool-00/home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
known = json.load(open("/zpool-00/home/lz299/BasisServe-CALS/ICLR-results/qwen3-8b/c1/two-sided-kl/profile-s6-a1/shard_00.json"))["configuration"]["windows_provenance"]["document_ids"]
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
stream = load_dataset("allenai/c4", "en", split="train", streaming=True, revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2").shuffle(seed=seed, buffer_size=10_000)
seen, usable, started = set(), [], time.time()
for index, row in enumerate(stream):
    doc = f"url:{row['url']}" if row.get("url") else f"timestamp:{row['timestamp']}"
    if doc in seen:
        continue
    if len(tokenizer(row["text"], add_special_tokens=False)["input_ids"]) < length:
        continue
    seen.add(doc)
    usable.append(doc)
    if len(usable) == 321:
        hit = usable[320] == known[0]
        print(f"seed {seed} L{length} datasets {datasets.__version__}: usable[320] {'MATCHES' if hit else 'differs'} ({usable[320][:70]}) after {index + 1} stream rows, {time.time() - started:.0f}s", flush=True)
        if not hit:
            print("first usable docs:", usable[:3], flush=True)
            break
    if len(usable) == 464:
        matches = sum(a == b for a, b in zip(usable[320:464], known))
        print(f"seed {seed}: {matches}/144 fingerprint documents match in order; stream rows read {index + 1}", flush=True)
        break
