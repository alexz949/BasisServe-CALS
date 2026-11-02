"""Convert the paper's published Qwen3-8B-Base factors and KL probes to the BasisKV release formats.

Banks: ICLR-results/qwen3-8b/c1/factor-banks/R{rank}-S6 (same tensor names and shapes as the release; the layer
files are linked, not copied). Rank 128 is the exact dense endpoint and is identical for any calibration, so the
release run's bank128 is linked. Probes: the per-layer terminal-KL deltas of profile-s6-a1 (rank 32 and 96 against
the uniform rank-64 anchor). The published allocated V64 bank is checked layer by layer against the converted banks.
"""
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/zpool-00/home/lz299/BasisServe-CALS/BasisKV")
from core.allocate import PROBES_FORMAT
from core.factor_io import LAYER_FORMAT, merge, write_json
from core.model_layout import ValueGeometry

ROOT = Path("/zpool-00/home/lz299/BasisServe-CALS")
HF = ROOT / "runs/q3-8b-base-v64/paper_hf/ICLR-results/qwen3-8b"
OUT = ROOT / "runs/q3-8b-base-v64/paper"
RUN = ROOT / "runs/q3-8b-base-v64"
geometry = ValueGeometry("qwen3", 36, 32, 8, 128, 4096, tuple(range(36)))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


for rank in (32, 48, 64, 80, 96, 112):
    source = HF / f"c1/factor-banks/R{rank}-S6"
    old = json.load(open(source / "results.json"))
    target = OUT / f"bank{rank}"
    target.mkdir(parents=True, exist_ok=True)
    config = {"cache_rank_per_head": rank, "encoder_sweeps": 6, "cg_iterations": 16, "covariance_damping": 1e-5,
              "geometry": geometry.to_dict(), "fit_windows": 256, "heldout_windows": 64, "positions_per_window": 2048,
              "source": f"alexz949/BasisServe-CALS ICLR-results/qwen3-8b/c1/factor-banks/R{rank}-S6"}
    for record in old["records"]:
        layer = int(record["layer"])
        artifact = target / f"layer_{layer:03d}.safetensors"
        if not artifact.exists():
            os.symlink(source / record["artifact"]["file"], artifact)
        assert file_sha256(artifact) == record["artifact"]["sha256"], f"rank {rank} layer {layer}: download checksum differs"
        write_json(target / f"layer_{layer:03d}.json", {
            "format": LAYER_FORMAT, "layer": layer, "fit_config": config,
            "fit": {k: record["fit"][k] for k in ("relative_mse", "factor_dtype_relative_mse")},
            "heldout": {k: record["heldout"][k] for k in ("relative_mse", "factor_dtype_relative_mse")}, "solver": {"method": "paper als (published bank)"},
            "artifact": {"file": artifact.name, "sha256": record["artifact"]["sha256"], "tensors": record["artifact"]["tensors"]}})
    if not (target / "results.json").exists():
        merge(target, geometry.attention_layers, geometry)
    print(f"bank{rank}: converted, mean held-out rel. MSE {json.load(open(target / 'results.json'))['aggregate']['mean_heldout_factor_dtype_relative_mse']:.5f}")

if not (OUT / "bank128").exists():
    os.symlink(RUN / "bank128", OUT / "bank128")

records = []
for path in sorted(glob.glob(str(ROOT / "ICLR-results/qwen3-8b/c1/two-sided-kl/profile-s6-a1/shard_*.json"))):
    shard = json.load(open(path))
    anchor = shard["uniform_anchor"]["terminal_kl"]["mean"]
    records += shard["records"]
by_layer = {}
for row in records:
    by_layer.setdefault(int(row["layer"]), {})[int(row["candidate_rank"])] = row["terminal_kl_delta"]["mean"]
(OUT / "probes").mkdir(parents=True, exist_ok=True)
write_json(OUT / "probes/probes.json", {
    "format": PROBES_FORMAT, "anchor_rank": 64, "low_rank": 32, "high_rank": 96, "anchor_kl": anchor,
    "source": "alexz949/BasisServe-CALS ICLR-results/qwen3-8b/c1/two-sided-kl/profile-s6-a1 (terminal_kl_delta.mean)",
    "layers": [{"layer": layer, "delta_low": by_layer[layer][32], "delta_high": by_layer[layer][96], "anchor_kl": anchor}
               for layer in sorted(by_layer)]})
print(f"probes: {len(by_layer)} layers, anchor KL {anchor:.5f}")

paper = json.load(open(HF / "checkpoints/Q3-8B-C1-R64/manifest.json"))
schedule = [row[0] if isinstance(row, list) else row for row in paper["compression"]["layer_ranks"]]
identical = 0
for layer, rank in enumerate(schedule):
    selected = HF / f"checkpoints/Q3-8B-C1-R64/selected_factors/layer_{layer:03d}.safetensors"
    identical += file_sha256(selected) == file_sha256(OUT / f"bank{rank}/layer_{layer:03d}.safetensors")
print(f"published V64 selected factors identical to the converted bank layers: {identical}/36")
write_json(OUT / "paper_schedule.json", {"layer_ranks": schedule})

# The published allocated V64 bank itself, as a release-format directory (the selected layers carry the paper's
# post-selection decoder refit, so they differ slightly from the uniform bank layers of changed rank).
from core.factor_io import _write_results, read_layer, shared_config
target = OUT / "Q3-8B-C1-R64-published"
target.mkdir(parents=True, exist_ok=True)
records = []
for layer, rank in enumerate(schedule):
    source = HF / f"checkpoints/Q3-8B-C1-R64/selected_factors/layer_{layer:03d}.safetensors"
    artifact = target / f"layer_{layer:03d}.safetensors"
    if not artifact.exists():
        os.symlink(source, artifact)
    base = json.load(open(OUT / f"bank{rank}/layer_{layer:03d}.json"))
    record = {**base, "artifact": {**base["artifact"], "sha256": file_sha256(artifact)},
              "solver": {"method": "paper two-sided KL selection with decoder refit (published selected_factors)"}}
    write_json(target / f"layer_{layer:03d}.json", record)
    records.append(read_layer(target, layer, None))
if not (target / "results.json").exists():
    config = shared_config(records[0]["fit_config"])
    _write_results(target, geometry.attention_layers, config, schedule, records, geometry,
                   allocation={"method": "published", "source": "alexz949/BasisServe-CALS ICLR-results/qwen3-8b/checkpoints/Q3-8B-C1-R64"})
print("published V64 bank written:", json.load(open(target / "results.json"))["layer_ranks"] == schedule)
