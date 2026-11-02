"""Convert the published Qwen3-8B (post) uniform V96 128K bank to the release value-factor layout, and write the YaRN-4 config.

Source: alexz949/BasisServe-CALS checkpoints/attention_c1/qwen3_8b_post_uniform_v96_128k_als6_retrievalmix/attention_v96
(same tensor names and shapes as the release; layer files are linked). The published bank carries no per-layer fit errors.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/zpool-00/home/lz299/BasisServe-CALS/BasisKV")
from core.factor_io import FORMAT, LAYER_FORMAT, write_json
from core.model_layout import ValueGeometry

ROOT = Path("/zpool-00/home/lz299/BasisServe-CALS/runs/q3-8b-t2")
SOURCE = ROOT / "hf/checkpoints/attention_c1/qwen3_8b_post_uniform_v96_128k_als6_retrievalmix/attention_v96"
TARGET = ROOT / "paper_v96"
MODEL = Path("/zpool-00/home/lz299/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
geometry = ValueGeometry("qwen3", 36, 32, 8, 128, 4096, tuple(range(36)))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


manifest = json.load(open(SOURCE / "manifest.json"))
config = {"cache_rank_per_head": 96, "geometry": geometry.to_dict(), "encoder_sweeps": 6, "fit_windows": 32, "heldout_windows": 16,
          "positions_per_window": 131072, "source": "alexz949/BasisServe-CALS " + str(SOURCE.relative_to(ROOT / "hf"))}
TARGET.mkdir(parents=True, exist_ok=True)
records = []
for entry in manifest["layers"]:
    layer = int(entry["layer"])
    artifact = TARGET / entry["file"]
    if not artifact.exists():
        os.symlink(SOURCE / entry["file"], artifact)
    digest = file_sha256(artifact)
    assert digest == entry["sha256"], f"layer {layer}: download checksum differs"
    record = {"format": LAYER_FORMAT, "layer": layer, "fit_config": config,
              "artifact": {"file": artifact.name, "sha256": digest,
                           "tensors": {"value_coordinate_encoders": {"shape": [8, 128, 96], "dtype": "torch.bfloat16"},
                                       "head_output_decoders": {"shape": [32, 96, 4096], "dtype": "torch.bfloat16"}}}}
    write_json(TARGET / f"layer_{layer:03d}.json", record)
    records.append(record)
write_json(TARGET / "results.json", {"format": FORMAT, "status": "complete", "layers": list(range(36)), "fit_config": config,
                                     "layer_ranks": [96] * 36, "artifacts": {str(r["layer"]): r["artifact"] for r in records},
                                     "records": records})
print("paper V96 bank:", len(records), "layers")

model_config = json.load(open(MODEL / "config.json"))
model_config["rope_scaling"] = None
model_config["rope_parameters"] = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768,
                                   "rope_theta": model_config.get("rope_theta", 1000000)}
model_config["max_position_embeddings"] = 131072
(ROOT / "yarn4").mkdir(exist_ok=True)
json.dump(model_config, open(ROOT / "yarn4/config.json", "w"), indent=1)
print("yarn4 config:", model_config["rope_parameters"], model_config["max_position_embeddings"])
