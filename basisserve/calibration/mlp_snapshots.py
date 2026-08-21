"""Strict-split post-SwiGLU snapshots for Qwen3.5 MLP channel sketching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.core.qwen35_gdn_runtime import qwen35_decoder_layers


SNAPSHOT_FORMAT = "basisserve.qwen35.mlp_channel_snapshots.v1"
SPLITS = ("train", "dev", "validation")


@dataclass(frozen=True)
class DocumentWindow:
    input_ids: Tensor
    line_index: int
    available_tokens: int
    start_token: int

    def record(self) -> dict[str, Any]:
        token_bytes = self.input_ids.to(torch.int64).numpy().tobytes()
        return {
            "line_index": self.line_index,
            "available_tokens": self.available_tokens,
            "start_token": self.start_token,
            "used_tokens": int(self.input_ids.numel()),
            "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
        }


def records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(records),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_cached_windows_by_document(
    sample_records: Sequence[Mapping[str, Any]],
    *,
    split_window_targets: Mapping[str, int],
    seed: int,
    maximum_attempts: int = 256,
) -> tuple[dict[str, tuple[int, ...]], tuple[int, ...], dict[str, Any]]:
    """Assign cached windows to exact-size splits without document leakage.

    Every cached window from one ``line_index`` is either assigned to the same
    split or left in quarantine.  Repeated windows from a document therefore
    cannot leak across train, development, and validation.
    """

    if set(split_window_targets) != set(SPLITS):
        raise ValueError(f"split targets must be exactly {SPLITS}")
    targets = {split: int(split_window_targets[split]) for split in SPLITS}
    if any(value <= 0 for value in targets.values()):
        raise ValueError("all cached split targets must be positive")
    if sum(targets.values()) > len(sample_records):
        raise ValueError("cached split targets exceed the available windows")
    groups: dict[int, list[int]] = {}
    for record_index, record in enumerate(sample_records):
        if "line_index" not in record:
            raise ValueError(f"sample record {record_index} has no line_index")
        groups.setdefault(int(record["line_index"]), []).append(record_index)
    if not groups:
        raise ValueError("cached sample records are empty")

    def exact_subset(
        available: Mapping[int, list[int]],
        target: int,
        rng: random.Random,
    ) -> tuple[int, ...] | None:
        items = list(available.items())
        rng.shuffle(items)
        paths: dict[int, tuple[int, ...]] = {0: ()}
        for line_index, record_indices in items:
            weight = len(record_indices)
            for subtotal, path in tuple(paths.items()):
                candidate = subtotal + weight
                if candidate <= target and candidate not in paths:
                    paths[candidate] = path + (line_index,)
            if target in paths:
                return paths[target]
        return None

    assignments: dict[str, tuple[int, ...]] | None = None
    for attempt in range(maximum_attempts):
        rng = random.Random(seed + 104729 * attempt)
        remaining = {
            line_index: list(indices) for line_index, indices in groups.items()
        }
        candidate_assignments: dict[str, tuple[int, ...]] = {}
        for split in SPLITS:
            chosen_documents = exact_subset(remaining, targets[split], rng)
            if chosen_documents is None:
                break
            record_indices = sorted(
                index
                for line_index in chosen_documents
                for index in remaining.pop(line_index)
            )
            candidate_assignments[split] = tuple(record_indices)
        if len(candidate_assignments) == len(SPLITS):
            assignments = candidate_assignments
            break
    if assignments is None:
        group_sizes = sorted(len(indices) for indices in groups.values())
        raise ValueError(
            "could not form exact document-disjoint cached splits with targets "
            f"{targets}; document group sizes are {group_sizes}"
        )
    used = {index for indices in assignments.values() for index in indices}
    quarantine = tuple(
        index for index in range(len(sample_records)) if index not in used
    )
    split_records = {
        split: [
            {"source_record_index": index, **dict(sample_records[index])}
            for index in assignments[split]
        ]
        for split in SPLITS
    }
    split_lines = {
        split: {int(record["line_index"]) for record in split_records[split]}
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if split_lines[left] & split_lines[right]:
                raise RuntimeError(f"document leakage between {left} and {right}")
    quarantine_records = [
        {"source_record_index": index, **dict(sample_records[index])}
        for index in quarantine
    ]
    metadata = {
        "seed": int(seed),
        "split_window_targets": targets,
        "records": split_records,
        "record_sha256": {
            split: records_sha256(split_records[split]) for split in SPLITS
        },
        "combined_record_sha256": records_sha256(
            [
                {"split": split, **record}
                for split in SPLITS
                for record in split_records[split]
            ]
        ),
        "quarantine_records": quarantine_records,
        "quarantine_record_sha256": records_sha256(quarantine_records),
        "document_split_isolation": True,
        "unique_documents": {split: len(split_lines[split]) for split in SPLITS},
    }
    return assignments, quarantine, metadata


def sample_disjoint_document_windows(
    dataset_path: Path,
    tokenizer: Any,
    *,
    text_field: str,
    sequence_length: int,
    split_token_targets: Mapping[str, int],
    tokens_per_sequence: int,
    seed: int,
) -> tuple[dict[str, tuple[DocumentWindow, ...]], dict[str, Any]]:
    """Choose one deterministic window per document with disjoint splits."""

    if sequence_length <= 0 or not 0 < tokens_per_sequence <= sequence_length:
        raise ValueError("invalid sequence/token sampling lengths")
    if set(split_token_targets) != set(SPLITS):
        raise ValueError(f"split targets must be exactly {SPLITS}")
    if any(int(value) <= 0 for value in split_token_targets.values()):
        raise ValueError("all split token targets must be positive")
    documents: list[tuple[int, str]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            payload = json.loads(line)
            text = payload.get(text_field)
            if isinstance(text, str) and text:
                documents.append((line_index, text))
    rng = random.Random(seed)
    rng.shuffle(documents)
    required_windows = {
        split: math.ceil(int(split_token_targets[split]) / tokens_per_sequence)
        for split in SPLITS
    }
    result: dict[str, list[DocumentWindow]] = {split: [] for split in SPLITS}
    split_cursor = 0
    for line_index, text in documents:
        while (
            split_cursor < len(SPLITS)
            and len(result[SPLITS[split_cursor]])
            >= required_windows[SPLITS[split_cursor]]
        ):
            split_cursor += 1
        if split_cursor == len(SPLITS):
            break
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < sequence_length:
            continue
        maximum_start = len(token_ids) - sequence_length
        start = 0 if maximum_start == 0 else rng.randint(0, maximum_start)
        result[SPLITS[split_cursor]].append(
            DocumentWindow(
                input_ids=torch.tensor(
                    token_ids[start : start + sequence_length],
                    dtype=torch.long,
                ),
                line_index=line_index,
                available_tokens=len(token_ids),
                start_token=start,
            )
        )
    missing = {
        split: required_windows[split] - len(result[split])
        for split in SPLITS
        if len(result[split]) < required_windows[split]
    }
    if missing:
        raise ValueError(
            f"dataset could not supply disjoint document windows: {missing}"
        )
    frozen = {split: tuple(result[split]) for split in SPLITS}
    records = {split: [window.record() for window in frozen[split]] for split in SPLITS}
    line_sets = {
        split: {int(record["line_index"]) for record in records[split]}
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if line_sets[left] & line_sets[right]:
                raise RuntimeError(f"document leakage between {left} and {right}")
    metadata = {
        "seed": seed,
        "sequence_length": sequence_length,
        "tokens_per_sequence": tokens_per_sequence,
        "token_targets": {
            key: int(value) for key, value in split_token_targets.items()
        },
        "records": records,
        "record_sha256": {split: records_sha256(records[split]) for split in SPLITS},
    }
    metadata["combined_record_sha256"] = records_sha256(
        [{"split": split, **record} for split in SPLITS for record in records[split]]
    )
    return frozen, metadata


def sample_disjoint_document_windows_with_reuse(
    dataset_path: Path,
    tokenizer: Any,
    *,
    text_field: str,
    sequence_length: int,
    split_token_targets: Mapping[str, int],
    tokens_per_sequence: int,
    seed: int,
    maximum_windows_per_document: int = 4,
    maximum_assignment_attempts: int = 256,
) -> tuple[dict[str, tuple[DocumentWindow, ...]], dict[str, Any]]:
    """Choose non-overlapping windows while keeping documents split-exclusive.

    Diversity is maximized by assigning one document per window whenever the
    dataset permits it.  If the number of eligible documents is smaller than
    the requested number of windows, additional non-overlapping windows are
    drawn only from documents already assigned to the same split.
    """

    if sequence_length <= 0 or not 0 < tokens_per_sequence <= sequence_length:
        raise ValueError("invalid sequence/token sampling lengths")
    if set(split_token_targets) != set(SPLITS):
        raise ValueError(f"split targets must be exactly {SPLITS}")
    if any(int(value) <= 0 for value in split_token_targets.values()):
        raise ValueError("all split token targets must be positive")
    if maximum_windows_per_document <= 0 or maximum_assignment_attempts <= 0:
        raise ValueError("multi-window document controls must be positive")
    required_windows = {
        split: math.ceil(int(split_token_targets[split]) / tokens_per_sequence)
        for split in SPLITS
    }
    eligible: list[tuple[int, list[int], int]] = []
    candidates_scanned = 0
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            candidates_scanned += 1
            payload = json.loads(line)
            text = payload.get(text_field)
            if not isinstance(text, str) or not text:
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            capacity = min(
                len(token_ids) // sequence_length,
                maximum_windows_per_document,
            )
            if capacity:
                eligible.append((line_index, token_ids, capacity))
    total_required = sum(required_windows.values())
    total_capacity = sum(record[2] for record in eligible)
    if len(eligible) < len(SPLITS) or total_capacity < total_required:
        raise ValueError(
            "dataset cannot supply split-exclusive non-overlapping windows: "
            f"eligible_documents={len(eligible)} capacity={total_capacity} "
            f"required={total_required}"
        )

    used_document_count = min(len(eligible), total_required)
    raw_counts = {
        split: used_document_count * required_windows[split] / total_required
        for split in SPLITS
    }
    document_counts = {
        split: max(1, min(required_windows[split], math.floor(raw_counts[split])))
        for split in SPLITS
    }
    while sum(document_counts.values()) < used_document_count:
        candidates = [
            split
            for split in SPLITS
            if document_counts[split] < required_windows[split]
        ]
        if not candidates:
            break
        chosen = max(
            candidates,
            key=lambda split: (
                raw_counts[split] - document_counts[split],
                required_windows[split] - document_counts[split],
            ),
        )
        document_counts[chosen] += 1
    while sum(document_counts.values()) > used_document_count:
        candidates = [split for split in SPLITS if document_counts[split] > 1]
        if not candidates:
            raise RuntimeError("could not allocate at least one document per split")
        chosen = min(
            candidates,
            key=lambda split: (
                raw_counts[split] - document_counts[split],
                document_counts[split],
            ),
        )
        document_counts[chosen] -= 1
    if sum(document_counts.values()) != used_document_count:
        raise RuntimeError("document allocation did not reach the requested count")

    assigned: dict[str, list[tuple[int, list[int], int]]] | None = None
    assignment_attempt = -1
    for assignment_attempt in range(maximum_assignment_attempts):
        rng = random.Random(seed + 104729 * assignment_attempt)
        shuffled = list(eligible)
        rng.shuffle(shuffled)
        selected = shuffled[:used_document_count]
        cursor = 0
        candidate_assignment: dict[str, list[tuple[int, list[int], int]]] = {}
        for split in SPLITS:
            count = document_counts[split]
            group = selected[cursor : cursor + count]
            cursor += count
            candidate_assignment[split] = group
        if all(
            sum(record[2] for record in candidate_assignment[split])
            >= required_windows[split]
            for split in SPLITS
        ):
            assigned = candidate_assignment
            break
    if assigned is None:
        raise ValueError(
            "could not distribute multi-window document capacity across splits; "
            f"document_counts={document_counts} required_windows={required_windows}"
        )

    result: dict[str, list[DocumentWindow]] = {split: [] for split in SPLITS}
    per_document_window_counts: dict[str, dict[int, int]] = {}
    for split_index, split in enumerate(SPLITS):
        group = assigned[split]
        take_counts = [1] * len(group)
        remaining = required_windows[split] - len(group)
        split_rng = random.Random(seed + 1_000_003 * (split_index + 1))
        level = 2
        while remaining:
            candidates = [
                index for index, record in enumerate(group) if record[2] >= level
            ]
            split_rng.shuffle(candidates)
            if not candidates:
                raise RuntimeError(f"split {split} exhausted document window capacity")
            for index in candidates:
                take_counts[index] += 1
                remaining -= 1
                if not remaining:
                    break
            level += 1
        per_document_window_counts[split] = {}
        for (line_index, token_ids, capacity), take in zip(
            group, take_counts, strict=True
        ):
            slack = len(token_ids) - capacity * sequence_length
            base = 0 if slack == 0 else split_rng.randint(0, slack)
            slots = sorted(split_rng.sample(range(capacity), take))
            per_document_window_counts[split][line_index] = take
            for slot in slots:
                start = base + slot * sequence_length
                result[split].append(
                    DocumentWindow(
                        input_ids=torch.tensor(
                            token_ids[start : start + sequence_length],
                            dtype=torch.long,
                        ),
                        line_index=line_index,
                        available_tokens=len(token_ids),
                        start_token=start,
                    )
                )
        split_rng.shuffle(result[split])

    frozen = {split: tuple(result[split]) for split in SPLITS}
    records = {split: [window.record() for window in frozen[split]] for split in SPLITS}
    line_sets = {
        split: {int(record["line_index"]) for record in records[split]}
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if line_sets[left] & line_sets[right]:
                raise RuntimeError(f"document leakage between {left} and {right}")
    metadata = {
        "seed": int(seed),
        "sequence_length": int(sequence_length),
        "tokens_per_sequence": int(tokens_per_sequence),
        "token_targets": {
            key: int(value) for key, value in split_token_targets.items()
        },
        "required_windows": required_windows,
        "candidate_documents_scanned": candidates_scanned,
        "eligible_documents": len(eligible),
        "eligible_nonoverlapping_window_capacity": total_capacity,
        "used_documents": used_document_count,
        "assigned_documents": document_counts,
        "maximum_windows_per_document": maximum_windows_per_document,
        "assignment_attempt": assignment_attempt,
        "per_document_window_counts": {
            split: {str(key): value for key, value in counts.items()}
            for split, counts in per_document_window_counts.items()
        },
        "records": records,
        "record_sha256": {split: records_sha256(records[split]) for split in SPLITS},
        "document_split_isolation": True,
        "windows_nonoverlapping_within_document": True,
        "selection_policy": "one_per_document_then_nonoverlapping_reuse",
    }
    metadata["combined_record_sha256"] = records_sha256(
        [{"split": split, **record} for split in SPLITS for record in records[split]]
    )
    return frozen, metadata


def sample_disjoint_document_windows_from_indexed_texts(
    documents: Iterable[tuple[int, str]],
    tokenizer: Any,
    *,
    sequence_length: int,
    split_token_targets: Mapping[str, int],
    tokens_per_sequence: int,
    seed: int,
) -> tuple[dict[str, tuple[DocumentWindow, ...]], dict[str, Any]]:
    """Choose strict-split windows from an already ordered document stream.

    This variant is intended for memory-mapped datasets: callers determine the
    document order without materializing every text value.  Document identifiers
    must be unique across the stream so split isolation remains auditable.
    """

    if sequence_length <= 0 or not 0 < tokens_per_sequence <= sequence_length:
        raise ValueError("invalid sequence/token sampling lengths")
    if set(split_token_targets) != set(SPLITS):
        raise ValueError(f"split targets must be exactly {SPLITS}")
    if any(int(value) <= 0 for value in split_token_targets.values()):
        raise ValueError("all split token targets must be positive")
    required_windows = {
        split: math.ceil(int(split_token_targets[split]) / tokens_per_sequence)
        for split in SPLITS
    }
    result: dict[str, list[DocumentWindow]] = {split: [] for split in SPLITS}
    rng = random.Random(seed)
    seen_documents: set[int] = set()
    split_cursor = 0
    candidates_scanned = 0
    eligible_documents = 0
    for raw_document_id, text in documents:
        document_id = int(raw_document_id)
        if document_id in seen_documents:
            raise ValueError(f"duplicate document identifier: {document_id}")
        seen_documents.add(document_id)
        candidates_scanned += 1
        while (
            split_cursor < len(SPLITS)
            and len(result[SPLITS[split_cursor]])
            >= required_windows[SPLITS[split_cursor]]
        ):
            split_cursor += 1
        if split_cursor == len(SPLITS):
            break
        if not isinstance(text, str) or not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < sequence_length:
            continue
        eligible_documents += 1
        maximum_start = len(token_ids) - sequence_length
        start = 0 if maximum_start == 0 else rng.randint(0, maximum_start)
        result[SPLITS[split_cursor]].append(
            DocumentWindow(
                input_ids=torch.tensor(
                    token_ids[start : start + sequence_length],
                    dtype=torch.long,
                ),
                line_index=document_id,
                available_tokens=len(token_ids),
                start_token=start,
            )
        )
    missing = {
        split: required_windows[split] - len(result[split])
        for split in SPLITS
        if len(result[split]) < required_windows[split]
    }
    if missing:
        raise ValueError(
            "document stream could not supply disjoint windows: "
            f"{missing}; scanned={candidates_scanned} eligible={eligible_documents}"
        )
    frozen = {split: tuple(result[split]) for split in SPLITS}
    records = {split: [window.record() for window in frozen[split]] for split in SPLITS}
    line_sets = {
        split: {int(record["line_index"]) for record in records[split]}
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if line_sets[left] & line_sets[right]:
                raise RuntimeError(f"document leakage between {left} and {right}")
    metadata = {
        "seed": int(seed),
        "sequence_length": int(sequence_length),
        "tokens_per_sequence": int(tokens_per_sequence),
        "token_targets": {
            key: int(value) for key, value in split_token_targets.items()
        },
        "candidate_documents_scanned": candidates_scanned,
        "eligible_documents": eligible_documents,
        "records": records,
        "record_sha256": {split: records_sha256(records[split]) for split in SPLITS},
        "document_split_isolation": True,
    }
    metadata["combined_record_sha256"] = records_sha256(
        [{"split": split, **record} for split in SPLITS for record in records[split]]
    )
    return frozen, metadata


def sample_token_positions(
    available: int,
    take: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Stratify sampled positions over the unchanged sequence length."""

    if available <= 0 or not 0 < take <= available:
        raise ValueError("sample size must lie in [1, available]")
    if take == available:
        return torch.arange(available, dtype=torch.long)
    bins = torch.arange(take, dtype=torch.long)
    starts = torch.div(bins * available, take, rounding_mode="floor")
    stops = torch.div((bins + 1) * available, take, rounding_mode="floor")
    offsets = torch.floor(
        torch.rand(take, generator=generator) * (stops - starts).to(torch.float32)
    ).to(torch.long)
    return starts + offsets


def iter_qwen35_mlp_modules(
    model: nn.Module,
    layers: Iterable[int],
) -> tuple[tuple[int, nn.Module], ...]:
    decoder_layers = qwen35_decoder_layers(model)
    result: list[tuple[int, nn.Module]] = []
    for layer_index in sorted(set(map(int, layers))):
        if not 0 <= layer_index < len(decoder_layers):
            raise ValueError(f"layer {layer_index} is outside the decoder")
        mlp = getattr(decoder_layers[layer_index], "mlp", None)
        projections = [
            getattr(mlp, name, None) for name in ("gate_proj", "up_proj", "down_proj")
        ]
        weights = [getattr(module, "weight", None) for module in projections]
        if not all(isinstance(module, nn.Module) for module in projections) or not all(
            isinstance(weight, Tensor) and weight.ndim == 2 for weight in weights
        ):
            raise TypeError(f"layer {layer_index} is not a supported gated MLP")
        gate_weight, up_weight, down_weight = weights
        assert isinstance(gate_weight, Tensor)
        assert isinstance(up_weight, Tensor)
        assert isinstance(down_weight, Tensor)
        if gate_weight.shape != up_weight.shape:
            raise ValueError(f"layer {layer_index} gate/up shapes differ")
        if tuple(down_weight.shape) != (
            int(gate_weight.shape[1]),
            int(gate_weight.shape[0]),
        ):
            raise ValueError(
                f"layer {layer_index} down projection shape is incompatible"
            )
        assert isinstance(mlp, nn.Module)
        result.append((layer_index, mlp))
    return tuple(result)


class MLPSnapshotCollector:
    """Capture exact selected rows entering and leaving one ``down_proj``."""

    def __init__(
        self,
        *,
        layer_index: int,
        mlp: nn.Module,
        storage_dtype: torch.dtype,
    ) -> None:
        self.layer_index = int(layer_index)
        self.down_proj = mlp.down_proj
        self.storage_dtype = storage_dtype
        self.intermediate_size = int(self.down_proj.weight.shape[1])
        self.hidden_size = int(self.down_proj.weight.shape[0])
        self._split: str | None = None
        self._indices: Tensor | None = None
        self._pending_h_device: Tensor | None = None
        self._pending_h_cpu: Tensor | None = None
        self._pending_y_cpu: Tensor | None = None
        self._chunks: dict[str, dict[str, list[Tensor]]] = {
            split: {"H": [], "Y": []} for split in SPLITS
        }
        self._teacher_energy = 0.0
        self._consistency_error = 0.0
        self._consistency_max_abs = 0.0
        self.handles = (
            self.down_proj.register_forward_pre_hook(self._pre_hook),
            self.down_proj.register_forward_hook(self._output_hook),
        )

    def begin(self, split: str, flat_indices: Tensor) -> None:
        if split not in SPLITS or self._split is not None:
            raise RuntimeError("snapshot collector begin state is invalid")
        if flat_indices.ndim != 1 or flat_indices.dtype != torch.long:
            raise ValueError("flat indices must be a long vector")
        self._split = split
        self._indices = flat_indices.cpu()

    def _select(self, value: Tensor, width: int) -> Tensor:
        if self._indices is None or int(value.shape[-1]) != width:
            raise ValueError("snapshot hook tensor or collector state is invalid")
        rows = value.reshape(-1, width)
        return rows.index_select(0, self._indices.to(rows.device))

    @torch.no_grad()
    def _pre_hook(self, module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1 or not isinstance(inputs[0], Tensor):
            raise TypeError("down_proj must receive one tensor input")
        selected = self._select(inputs[0], self.intermediate_size).detach()
        self._pending_h_device = selected
        self._pending_h_cpu = selected.to(device="cpu", dtype=self.storage_dtype)

    @torch.no_grad()
    def _output_hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not isinstance(output, Tensor) or self._pending_h_device is None:
            raise TypeError("down_proj output hook received invalid state")
        selected_output = self._select(output, self.hidden_size).detach()
        direct = F.linear(
            self._pending_h_device,
            self.down_proj.weight,
            self.down_proj.bias,
        )
        difference = direct.float() - selected_output.float()
        self._consistency_error += float(
            torch.sum(difference.square(), dtype=torch.float64)
        )
        self._teacher_energy += float(
            torch.sum(selected_output.float().square(), dtype=torch.float64)
        )
        self._consistency_max_abs = max(
            self._consistency_max_abs,
            float(difference.abs().max()),
        )
        self._pending_y_cpu = selected_output.to(
            device="cpu",
            dtype=self.storage_dtype,
        )

    def finish(self) -> None:
        if (
            self._split is None
            or self._pending_h_cpu is None
            or self._pending_y_cpu is None
        ):
            raise RuntimeError(
                "snapshot collector did not observe a complete down projection"
            )
        self._chunks[self._split]["H"].append(self._pending_h_cpu.contiguous())
        self._chunks[self._split]["Y"].append(self._pending_y_cpu.contiguous())
        self._split = None
        self._indices = None
        self._pending_h_device = None
        self._pending_h_cpu = None
        self._pending_y_cpu = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def tensors(self, split: str, target_rows: int) -> dict[str, Tensor]:
        if split not in SPLITS or target_rows <= 0:
            raise ValueError("invalid snapshot split/row request")
        if not self._chunks[split]["H"]:
            raise RuntimeError(f"layer {self.layer_index} collected no {split} rows")
        return {
            name: torch.cat(chunks, dim=0)[:target_rows].contiguous()
            for name, chunks in self._chunks[split].items()
        }

    def diagnostics(self) -> dict[str, float]:
        return {
            "direct_down_relative_mse": self._consistency_error
            / max(self._teacher_energy, 1e-300),
            "direct_down_max_abs_error": self._consistency_max_abs,
        }
