"""Reproduce a paper window bank's document order: stream allenai/c4 en SPLIT at revision 1588ec4, shuffle(seed, 10000),
keep documents with >= L tokens (no special tokens), dedupe by url; compare usable[offset:offset+n] with known urls.
Usage: stream_probe.py SPLIT SEED L KNOWN_JSON OFFSET   (KNOWN_JSON: a list of 'url:...' ids)"""
import json
import sys
import time

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer

split, seed, length, known_path, offset = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
known = json.load(open(known_path))
model = "/zpool-00/home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
stream = load_dataset("allenai/c4", "en", split=split, streaming=True, revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2").shuffle(seed=seed, buffer_size=10_000)
seen, usable, started = set(), [], time.time()
for index, row in enumerate(stream):
    doc = f"url:{row['url']}"
    if doc in seen or len(tokenizer(row["text"], add_special_tokens=False)["input_ids"]) < length:
        continue
    seen.add(doc)
    usable.append(doc)
    if len(usable) == offset + 1 and usable[offset] != known[0]:
        print(f"{split} seed {seed} L{length} datasets {datasets.__version__}: usable[{offset}] differs ({usable[offset][:60]} vs {known[0][:60]})", flush=True)
        break
    if len(usable) == offset + len(known):
        matches = sum(a == b for a, b in zip(usable[offset:], known))
        print(f"{split} seed {seed} L{length} datasets {datasets.__version__}: {matches}/{len(known)} documents match in order ({time.time() - started:.0f}s)", flush=True)
        break
