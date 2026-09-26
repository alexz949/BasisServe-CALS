"""Validate the matched 128K Qwen banks and materialize routing coordinates."""

import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basisserve.core.routing_basis import make_routing_basis


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen32", "qwen8"), default="qwen32")
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = args.snapshot / "checkpoints/qwen3-32b-128k/uniform-v96-als6-cg16"
    router = args.snapshot / "checkpoints/qwen3-32b-128k/uniform96-b16r16-als40-pcg100"
    layers, heads, hidden = 64, 64, 5120
    if args.model == "qwen8":
        bank = args.snapshot / "checkpoints/attention_c1/qwen3_8b_post_uniform_v96_128k_als6_retrievalmix"
        source, router = bank / "attention_v96", bank / "router_b16r16"
        layers, heads, hidden = 36, 32, 4096
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for layer in range(layers):
        name = f"layer_{layer:03d}.safetensors"
        values = load_file(str(source / name))
        factors = load_file(str(router / name))
        encoder = values["value_coordinate_encoders"].double()
        decoder = values["head_output_decoders"].double()
        assert encoder.shape == (8, 128, 96)
        assert decoder.shape == (heads, 96, hidden)
        shapes = {"base_left_b16": (8, 96, 16), "base_right_b16": (8, 16, 128),
                  "base_bias_b16": (8, 128), "residual_encoder_b16_r16": (8, 128, 16),
                  "residual_query_b16_r16": (heads, 128, 16)}
        for key, shape in shapes.items():
            assert factors[key].shape == shape and torch.isfinite(factors[key]).all()
        assert torch.isfinite(encoder).all() and torch.isfinite(decoder).all()
        basis = make_routing_basis(factors["base_left_b16"])
        expected = torch.zeros(8, 96, 16, dtype=torch.float64)
        expected[:, :16] = torch.eye(16, dtype=torch.float64)
        torch.testing.assert_close(basis.inverse @ factors["base_left_b16"].double(),
                                   expected, atol=1e-8, rtol=1e-8)
        transformed_encoder = basis.encoder(encoder)
        transformed_decoder = basis.decoder(decoder)
        # Probe each head's composed map before the explicit BF16 export.
        generator = torch.Generator().manual_seed(17 + layer)
        probe = torch.randn(heads, 2, 128, dtype=torch.float64, generator=generator)
        original = (probe @ encoder.repeat_interleave(heads // 8, 0)) @ decoder
        changed = (probe @ transformed_encoder.repeat_interleave(heads // 8, 0)) @ transformed_decoder
        torch.testing.assert_close(changed, original, atol=1e-8, rtol=1e-8)
        output = args.output / name
        assert not output.exists(), f"Preserve existing factors: {output}"
        save_file({"value_coordinate_encoders": transformed_encoder.bfloat16().contiguous(),
                   "head_output_decoders": transformed_decoder.bfloat16().contiguous()}, str(output))
        record = {"layer": layer, "condition_max": basis.condition.max().item(),
                  "coordinate_map_max_abs_error": (changed - original).abs().max().item()}
        records.append(record)
        print(json.dumps(record), flush=True)
    manifest = {"status": "complete", "validation": "structure_and_numeric_no_sha256",
                "source": str(source), "router": str(router), "revision": args.snapshot.name,
                "value_rank": 96, "base_rank": 16, "residual_rank": 16,
                "layers": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
