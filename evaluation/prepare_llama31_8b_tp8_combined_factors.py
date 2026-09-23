"""Materialize the V96 coordinate change shared by TP8 ALS and Basis arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from basisserve.core.routing_basis import make_routing_basis


DEFAULT_SOURCE = Path(
    "/workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/"
    "snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/checkpoints/attention_c1/"
    "llama31_8b_instruct_uniform_v96_128k_als6"
)
DEFAULT_ROUTER = Path("/workspace/runs/l31-cal128/router/ours_b16r16")
DEFAULT_OUTPUT = Path("/workspace/runs/l31-cal128/tp8-combined-v96")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--router", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    assert args.source.is_dir() and args.router.is_dir()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    for layer in range(rank, 32, world):
        source_path = args.source / f"layer_{layer:03d}.safetensors"
        router_path = args.router / f"layer_{layer:03d}.safetensors"
        output_path = args.output / f"layer_{layer:03d}.safetensors"
        assert source_path.is_file() and router_path.is_file() and not output_path.exists()
        values = load_file(str(source_path), device=str(device))
        router = load_file(str(router_path), device=str(device))
        encoder = values["value_coordinate_encoders"].double()
        decoder = values["head_output_decoders"].double()
        assert tuple(encoder.shape) == (8, 128, 96)
        assert tuple(decoder.shape) == (32, 96, 4096)
        basis = make_routing_basis(router["base_left_b16"])
        transformed_encoder = basis.encoder(encoder)
        transformed_decoder = basis.decoder(decoder)
        base_error = (
            transformed_encoder.new_zeros(8, 96, 16)
        )
        base_error[:, :16] = torch.eye(16, device=device, dtype=torch.float64)
        transformed_left = basis.inverse @ router["base_left_b16"].double()
        torch.testing.assert_close(transformed_left, base_error, rtol=1e-8, atol=1e-8)
        save_file(
            {
                "value_coordinate_encoders": transformed_encoder.bfloat16().cpu().contiguous(),
                "head_output_decoders": transformed_decoder.bfloat16().cpu().contiguous(),
            },
            str(output_path),
        )
        print(
            json.dumps(
                {
                    "rank": rank,
                    "layer": layer,
                    "condition_max": float(basis.condition.max()),
                    "output": str(output_path),
                }
            ),
            flush=True,
        )

    if world > 1:
        dist.barrier()
    if rank == 0:
        files = []
        for layer in range(32):
            source_path = args.source / f"layer_{layer:03d}.safetensors"
            router_path = args.router / f"layer_{layer:03d}.safetensors"
            output_path = args.output / f"layer_{layer:03d}.safetensors"
            assert output_path.is_file()
            files.append(
                {
                    "layer": layer,
                    "file": output_path.name,
                    "sha256": _sha256(output_path),
                    "source_sha256": _sha256(source_path),
                    "router_sha256": _sha256(router_path),
                }
            )
        manifest_path = args.output / "manifest.json"
        assert not manifest_path.exists()
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "format": "basisserve.llama31_8b.tp8_combined_v96.v1",
                    "value_rank": 96,
                    "base_rank": 16,
                    "residual_rank": 16,
                    "source": str(args.source.resolve()),
                    "router": str(args.router.resolve()),
                    "files": files,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "complete", "manifest": str(manifest_path)}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
