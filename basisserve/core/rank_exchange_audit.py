"""Fixed-origin rank exchanges, durable batch records, and reversible weights."""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path

import torch


RANKS = (32, 48, 64, 80, 96, 112, 128)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def enumerate_exchanges(base, *, ranks=RANKS, step=16):
    base = list(map(int, base))
    assert base and all(r in ranks for r in base)
    candidates = []
    seen = set()
    for donor, before_d in enumerate(base):
        for receiver, before_r in enumerate(base):
            if donor == receiver or before_d - step not in ranks or before_r + step not in ranks:
                continue
            schedule = base.copy()
            schedule[donor] -= step
            schedule[receiver] += step
            assert sum(schedule) == sum(base)
            assert {i for i, pair in enumerate(zip(base, schedule)) if pair[0] != pair[1]} == {donor, receiver}
            assert tuple(schedule) not in seen
            seen.add(tuple(schedule))
            candidates.append({
                "id": f"donor_{donor:03d}_receiver_{receiver:03d}",
                "donor": donor, "receiver": receiver,
                "before": [before_d, before_r],
                "after": [schedule[donor], schedule[receiver]],
                "schedule": schedule, "schedule_hash": fingerprint(schedule),
                "replay_start": min(donor, receiver),
            })
    return candidates


def shard_candidates(candidates, index, count):
    assert count > 0 and 0 <= index < count
    return sorted(candidates, key=lambda row: row["id"])[index::count]


def work_counts(candidates, windows, batch_size, shards, layers=36):
    assert min(windows, batch_size, shards) > 0
    batches = math.ceil(windows / batch_size)
    return {
        "candidate_count": len(candidates), "windows": windows,
        "microbatch_size": batch_size, "batches_per_candidate": batches,
        "candidate_batch_calls": len(candidates) * batches,
        "candidate_full_forward_calls_if_full_forward": len(candidates) * batches,
        "candidate_replay_calls_if_suffix_replay": len(candidates) * batches,
        "decoder_layer_calls_if_suffix_replay": sum(layers - row["replay_start"] for row in candidates) * batches,
        "decoder_layer_calls_if_full_forward": layers * len(candidates) * batches,
        "teacher_preparation_full_forwards": batches,
        "baseline_full_forwards_all_workers": shards * batches,
        "shards": [{"index": i, "candidates": len(shard_candidates(candidates, i, shards)),
                    "candidate_batch_calls": len(shard_candidates(candidates, i, shards)) * batches}
                   for i in range(shards)],
        "excluded": ["resume recomputation", "smoke", "uniform V64", "winner/baseline full-forward recheck", "heldout KL", "test PPL"],
    }


def batch_starts(windows, batch_size):
    assert windows > 0 and batch_size > 0
    return list(range(0, windows, batch_size))


class BatchJournal:
    """One owner per journal. Atomic replacement makes a batch exactly-once."""

    def __init__(self, path, configuration, windows, batch_size, resume=False):
        self.path = Path(path)
        self.windows, self.batch_size = windows, batch_size
        self.starts = batch_starts(windows, batch_size)
        identity = {"configuration": configuration, "windows": windows, "batch_size": batch_size}
        if self.path.exists():
            assert resume, f"Existing results require --resume: {path}"
            self.data = json.loads(self.path.read_text())
            assert self.data["identity"] == identity, "Resume configuration mismatch"
        else:
            self.data = {"identity": identity, "records": {}, "events": [], "sessions": []}

    def save(self):
        atomic_json(self.path, self.data)

    def has(self, candidate, start):
        return str(start) in self.data["records"].get(candidate, {})

    def put(self, candidate, start, record):
        assert start in self.starts
        assert len(record["kl"]) == min(self.batch_size, self.windows - start)
        assert all(math.isfinite(x) for x in record["kl"])
        records = self.data["records"].setdefault(candidate, {})
        key = str(start)
        if key in records:
            assert records[key] == record, "A completed batch cannot be replaced"
            return
        records[key] = record
        self.save()

    def complete(self, candidate):
        return set(self.data["records"].get(candidate, {})) == set(map(str, self.starts))

    def values(self, candidate):
        assert self.complete(candidate), f"Incomplete candidate: {candidate}"
        return [v for start in self.starts for v in self.data["records"][candidate][str(start)]["kl"]]


@contextmanager
def temporary_projection_weights(layers, replacements):
    """Hold original tensors by reference; restore even after partial installation."""
    original = {}
    try:
        with torch.no_grad():
            for index, (value, output) in replacements.items():
                attention = layers[index].self_attn
                original[index] = (attention.v_proj.weight.data, attention.o_proj.weight.data)
                attention.v_proj.weight.data = value.to(device=attention.v_proj.weight.device, dtype=attention.v_proj.weight.dtype)
                attention.o_proj.weight.data = output.to(device=attention.o_proj.weight.device, dtype=attention.o_proj.weight.dtype)
        yield
    finally:
        with torch.no_grad():
            for index, (value, output) in original.items():
                layers[index].self_attn.v_proj.weight.data = value
                layers[index].self_attn.o_proj.weight.data = output


def summarize_candidate(candidate, values, baseline, records, positions):
    assert len(values) == len(baseline)
    differences = [x - y for x, y in zip(values, baseline)]
    return {
        **candidate, "complete": True, "per_window_kl": values,
        "mean_kl": math.fsum(values) / len(values),
        "per_window_delta": differences,
        "mean_delta": math.fsum(differences) / len(differences),
        "windows": len(values), "positions": len(values) * positions,
        "context_tokens": len(values) * 2048,
        "elapsed_seconds": math.fsum(row["elapsed_seconds"] for row in records),
        "gpu_seconds": math.fsum(row["gpu_seconds"] for row in records),
        "peak_gpu_bytes": max(row["peak_gpu_bytes"] for row in records),
        "full_forward_calls": sum(row["full_forward_calls"] for row in records),
        "suffix_replay_calls": sum(row["suffix_replay_calls"] for row in records),
        "decoder_layer_calls": sum(row["decoder_layer_calls"] for row in records),
    }
