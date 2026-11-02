"""Pin the Llama V96 and B16R16 assets used by the Ampere TP8 runner."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO = "alexz949/BasisServe-CALS"
ROOT = "checkpoints/attention_c1"
ORIGINAL = "llama31_8b_instruct_uniform_v96_128k_als6"
RETRIEVAL = "llama31_8b_instruct_uniform_v96_128k_als6_retrievalmix"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
WINDOWS = "calibration/llama31-8b-instruct-128k/c4-48x128k"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(root, name):
    folder = root / ROOT / name
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["layers"]) == 32
    layers = []
    for index, entry in enumerate(manifest["layers"]):
        path = folder / entry["file"]
        digest = sha256(path)
        assert entry["layer"] == index and digest == entry["sha256"]
        assert entry["ranks"] == [96] * 8
        layers.append({"layer": index, "path": str(path), "sha256": digest})
    return {
        "path": str(folder),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "layers": layers,
    }


def router(root):
    folder = root / ROOT / RETRIEVAL / "router_b16r16"
    layers = []
    training_manifests = set()
    identities = set()
    for index in range(32):
        path = folder / f"layer_{index:03d}.safetensors"
        record_path = folder / f"layer_{index:03d}.json"
        record = json.loads(record_path.read_text())
        protocol = record["protocol"]
        digest = sha256(path)
        assert record["status"] == "complete" and record["layer"] == index
        assert digest == record["sha256"] and record["v_rank"] == 96
        assert (record["sweeps"], record["pcg_iterations"]) == (40, 100)
        assert (protocol["base_rank"], protocol["residual_rank"]) == (16, 16)
        assert protocol["page_size"] == 32
        training_manifests.add(protocol["deployed_checkpoint_manifest_sha256"])
        identities.add(record["identity_sha256"])
        layers.append({
            "layer": index,
            "path": str(path),
            "sha256": digest,
            "record_path": str(record_path),
            "record_sha256": sha256(record_path),
        })
    assert len(training_manifests) == len(identities) == 1
    return {
        "path": str(folder),
        "training_manifest_sha256": training_manifests.pop(),
        "training_identity_sha256": identities.pop(),
        "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    args = parser.parse_args()

    revision = HfApi().model_info(REPO).sha
    patterns = [
        f"{ROOT}/{name}/manifest.json"
        for name in (ORIGINAL, RETRIEVAL)
    ] + [
        f"{ROOT}/{name}/layer_*.safetensors"
        for name in (ORIGINAL, RETRIEVAL)
    ] + [
        f"{ROOT}/{RETRIEVAL}/router_b16r16/layer_*.{suffix}"
        for suffix in ("json", "safetensors")
    ] + [f"{WINDOWS}/manifest.json", f"{WINDOWS}/windows.safetensors"]
    root = Path(snapshot_download(REPO, revision=revision, allow_patterns=patterns))
    model = args.model_cache.resolve()
    assert model.is_dir() and model.name == MODEL_REVISION
    config = model / "config.json"
    weight_index = model / "model.safetensors.index.json"
    original = checkpoint(root, ORIGINAL)
    retrieval = checkpoint(root, RETRIEVAL)
    bank = router(root)
    windows_manifest_path = root / WINDOWS / "manifest.json"
    windows_path = root / WINDOWS / "windows.safetensors"
    windows_manifest = json.loads(windows_manifest_path.read_text())
    assert windows_manifest["status"] == "complete"
    assert windows_manifest["shape"] == [48, 131072]
    assert sha256(windows_path) == windows_manifest["sha256"]
    assert sha256(config) == "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e"
    assert sha256(weight_index) == "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b"
    manifest = {
        "schema": "basisserve.ampere_tp8_assets.v1",
        "source": {"repo": REPO, "revision": revision},
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "path": str(model),
            "config_sha256": sha256(config),
            "safetensors_index_sha256": sha256(weight_index),
        },
        "original_v96": original,
        "retrievalmix_v96": retrieval,
        "router_als40_pcg100_b16r16": bank,
        "input_windows": {
            "path": str(windows_path),
            "sha256": sha256(windows_path),
            "manifest_path": str(windows_manifest_path),
            "manifest_sha256": sha256(windows_manifest_path),
        },
        "joint_checkpoint": "retrievalmix_v96",
        "router_training_manifest_matches_current": bank["training_manifest_sha256"] == retrieval["manifest_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "revision": revision,
        "original_layers": len(original["layers"]),
        "retrievalmix_layers": len(retrieval["layers"]),
        "router_layers": len(bank["layers"]),
        "router_training_manifest_matches_current": manifest["router_training_manifest_matches_current"],
    }), flush=True)


if __name__ == "__main__":
    main()
