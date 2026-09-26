"""Check completed train-only Llama calibration artifacts without hashes."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from evaluation.eval_qwen3_kv4_fp8_ppl import write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/l31-nuq4")
    args = parser.parse_args()
    torch.set_num_threads(2)
    reports = []
    for rank in (64, 96):
        directory = args.root / "formal" / f"r{rank}"
        manifest = json.loads((directory / "manifest.json").read_text())
        complete = json.loads((directory / "complete.json").read_text())
        assert complete["status"] == "complete" and complete["rank"] == rank
        assert manifest["phase"] == "formal" and manifest["rank"] == rank
        assert manifest["length"] == 2048 and len(manifest["starts"]) == 16
        assert manifest["calibration_split"] == "WT2 train" and manifest["environment"] == "basis"
        codes = torch.load(directory / "quantizers.pt", map_location="cpu", weights_only=False)
        assert set(codes) == {f"{i}.{kind}" for i in range(32) for kind in ("k", "v")}
        for name, (hi, lo, lut) in codes.items():
            assert hi.shape == lo.shape and torch.isfinite(hi).all() and torch.isfinite(lo).all()
            assert (hi >= lo).all()
            poles = torch.as_tensor(lut[0]).flatten()
            assert poles.shape == (16,) and torch.isfinite(poles).all()
            if name.endswith(".k"):
                assert hi.numel() == 1024
        scales = json.loads((directory / "a8_scales.json").read_text())
        assert set(scales) == {f"{i}.decoder" for i in range(32)}
        for row in scales.values():
            assert all(math.isfinite(row[k]) and row[k] > 0 for k in ("input_scale", "weight_scale", "observed_amax"))
            assert math.isclose(row["input_scale"], row["observed_amax"] / 448, rel_tol=1e-6)
        for source in (ROOT / "evaluation/calibrate_llama_nuq4.py",
                       ROOT / "basisserve/core/llama_nuq4_quality.py",
                       ROOT / "basisserve/core/qwen3_kv4_fp8_quality.py",
                       ROOT / "evaluation/eval_qwen3_kv4_fp8_ppl.py",
                       ROOT / "external/KVQuant/quant/kvquant/simquant_module_quantizer.py"):
            assert source.read_bytes() == (directory / "source" / source.name).read_bytes()
        report = dict(rank=rank, status="passed", calibration_tokens=32768,
            codebooks=len(codes), decoder_scales=len(scales), source_bytes_match=True,
            quantizer_bytes=(directory / "quantizers.pt").stat().st_size, hashes=False)
        reports.append(report)
        print(report, flush=True)
    write(args.root / "formal/audit.json", dict(status="passed", environment="basis", reports=reports))


if __name__ == "__main__":
    main()
