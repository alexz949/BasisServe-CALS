"""Verify local Llama TP8 V96 and B16R16 benchmark inputs."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shapes(path):
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        return {key: stream.get_slice(key).get_shape() for key in stream.keys()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.factors / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    window_manifest = json.loads(args.windows.with_name("manifest.json").read_text())
    identity = manifest["model"]
    assert identity["revision"] == args.model.name
    for name, expected in (("config.json", "config_sha256"),
                           ("model.safetensors.index.json", "safetensors_index_sha256"),
                           ("tokenizer.json", "tokenizer_sha256")):
        assert sha256(args.model / name) == identity[expected]
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    assert len(shards) == 4 and all((args.model / shard).is_file() for shard in shards)
    assert sha256(args.windows) == window_manifest["sha256"]
    assert window_manifest["model_config_sha256"] == identity["config_sha256"]
    assert shapes(args.windows)["input_ids"] == [48, 131072]
    assert len(manifest["layers"]) == 32
    router_identity = None
    deployed_manifest = None
    for layer in range(32):
        factor_record = manifest["layers"][layer]
        factor = args.factors / factor_record["file"]
        router = args.factors / "router_b16r16" / f"layer_{layer:03d}.safetensors"
        router_record = json.loads(router.with_suffix(".json").read_text())
        protocol = router_record["protocol"]
        assert protocol["model_config_sha256"] == identity["config_sha256"]
        assert protocol["windows_sha256"] == manifest["calibration"]["windows_sha256"]
        assert protocol["base_rank"] == protocol["residual_rank"] == 16
        assert protocol["page_size"] == 32
        assert protocol["deployment_page_budget_tokens"] == 2048
        assert protocol["maximum_deployment_support_tokens"] == 2048
        assert protocol["sink_tokens_within_page_budget"] == 32
        assert protocol["recent_tokens_within_page_budget"] == 64
        assert router_record["v_rank"] == manifest["compression"]["rank"] == 96
        router_identity = router_identity or router_record["identity_sha256"]
        deployed_manifest = deployed_manifest or protocol["deployed_checkpoint_manifest_sha256"]
        assert router_record["identity_sha256"] == router_identity
        assert protocol["deployed_checkpoint_manifest_sha256"] == deployed_manifest
        assert factor_record["layer"] == router_record["layer"] == layer
        assert sha256(factor) == factor_record["sha256"]
        assert sha256(router) == router_record["sha256"]
        v = shapes(factor)
        r = shapes(router)
        assert v["value_coordinate_encoders"] == [8, 128, 96]
        assert v["head_output_decoders"] == [32, 96, 4096]
        assert r["base_left_b16"] == [8, 96, 16]
        assert r["base_right_b16"] == [8, 16, 128]
        assert r["base_bias_b16"] == [8, 128]
        assert r["residual_encoder_b16_r16"] == [8, 128, 16]
        assert r["residual_query_b16_r16"] == [32, 128, 16]
    result = {"status": "verified", "model_revision": args.model.name,
              "factor_manifest_sha256": sha256(manifest_path), "factor_layers": 32,
              "router_layers": 32, "prompt_windows": 48,
              "prompt_tokens_per_window": 131072,
              "prompt_windows_sha256": window_manifest["sha256"],
              "router_identity_sha256": router_identity,
              "router_deployed_manifest_sha256": deployed_manifest,
              "router_calibration_windows_sha256": manifest["calibration"]["windows_sha256"],
              "router_page_size": 32, "router_base_rank": 16,
              "router_residual_rank": 16, "support_tokens": 2048}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
