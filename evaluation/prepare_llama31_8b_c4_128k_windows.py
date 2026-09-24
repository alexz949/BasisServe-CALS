"""Document-disjoint 128K C4 windows for Llama-3.1-8B-Instruct C1 calibration.

Each window packs 32 disjoint 4096-token excerpts, one per C4 document, with no
separators: the geometry of the project's existing 128K window banks. The
first ``--fit-windows`` windows are the fitting set and the rest are held out.
Every excerpt records its document hash so later datasets can stay disjoint.
"""
import argparse
import hashlib
from pathlib import Path
import random
import sys

from datasets import load_dataset
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.v96kl_common import read_json, write_json, save_tensors, sha256

PIECE = 4096
PIECES_PER_WINDOW = 32
DATASET_REVISION = '1588ec454efa1a09f29cd18ddd04fe05fc8653a2'


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fit-windows', type=int, default=32)
    p.add_argument('--heldout-windows', type=int, default=16)
    p.add_argument('--seed', type=int, default=20260921)
    p.add_argument('--shuffle-buffer', type=int, default=10000)
    p.add_argument('--pieces-per-window', type=int, default=PIECES_PER_WINDOW,
                   help='4096-token excerpts per window (32 = 128K, 16 = 64K)')
    args = p.parse_args()
    out = args.output
    pieces_per_window = args.pieces_per_window
    sequence_length = PIECE * pieces_per_window
    if (out / 'manifest.json').exists():
        record = read_json(out / 'manifest.json')
        assert record['status'] == 'complete' and record['sha256'] == sha256(out / 'windows.safetensors')
        print('C4 128K windows already complete', flush=True)
        return
    total = args.fit_windows + args.heldout_windows
    needed = total * pieces_per_window
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True,
                           revision=DATASET_REVISION).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    rng = random.Random(args.seed)
    pieces, records, seen = [], [], set()
    for index, row in enumerate(dataset):
        document = hashlib.sha256(row['text'].encode()).hexdigest()
        if document in seen:
            continue
        ids = tokenizer(row['text'], add_special_tokens=False)['input_ids']
        if len(ids) < PIECE:
            continue
        start = rng.randint(0, len(ids) - PIECE)
        piece = torch.tensor(ids[start:start + PIECE], dtype=torch.int32)
        pieces.append(piece)
        seen.add(document)
        records.append(dict(document_sha256=document, stream_index=index, start=start,
                            document_tokens=len(ids), token_sha256=tensor_hash(piece)))
        if len(pieces) % pieces_per_window == 0:
            print(f'packed C4 {sequence_length}-token windows {len(pieces) // pieces_per_window}/{total} '
                  f'(scanned {index + 1} documents)', flush=True)
        if len(pieces) == needed:
            break
    assert len(pieces) == len(seen) == needed
    packed = torch.stack(pieces).reshape(total, sequence_length)
    out.mkdir(parents=True, exist_ok=True)
    save_tensors(out / 'windows.safetensors', dict(input_ids=packed))
    write_json(out / 'manifest.json', dict(
        status='complete', sha256=sha256(out / 'windows.safetensors'),
        model=str(args.model.resolve()), model_config_sha256=sha256(args.model / 'config.json'),
        tokenizer_sha256=sha256(args.model / 'tokenizer.json'),
        dataset='allenai/c4', dataset_config='en', dataset_split='train', dataset_revision=DATASET_REVISION,
        method='stream C4 shuffled with the seed; skip repeated document hashes and documents shorter '
               f'than one piece; take one uniformly placed 4096-token excerpt per document; pack {pieces_per_window} '
               'excerpts per window in stream order; no separators; no special tokens',
        seed=args.seed, shuffle_buffer=args.shuffle_buffer,
        fit_ids=list(range(args.fit_windows)), validation_ids=list(range(args.fit_windows, total)),
        shape=[total, sequence_length], piece_length=PIECE, pieces_per_window=pieces_per_window,
        records=records, source_sha256=sha256(Path(__file__))))
    print('C4 windows complete', tuple(packed.shape), flush=True)


if __name__ == '__main__':
    main()
