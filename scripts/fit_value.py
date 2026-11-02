#!/usr/bin/env python3
"""Fit Value factors for MHA/GQA models from calibration snapshot metadata.

Model dimensions and attention-layer indices are read from the snapshot;
merge reads them from the fitted layer records. No model-specific launcher
or model-weight loading is needed. Install the repository with pip install -e .
"""

import argparse
import json
from pathlib import Path


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument(
        "--validation-snapshot-dir",
        help="Optional separate snapshot used only for held-out evaluation",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--fit-windows",
        type=int,
        help="Defaults to the fit partition recorded in a covariance snapshot",
    )
    parser.add_argument(
        "--validation-windows",
        type=int,
        help="Defaults to the held-out partition recorded in a covariance snapshot",
    )
    parser.add_argument(
        "--validation-window-start",
        type=int,
        help=(
            "Explicit zero-based validation-window offset inside a larger "
            "snapshot; omitted means fit and validation exactly partition it"
        ),
    )
    parser.add_argument(
        "--rank",
        dest="cache_rank",
        type=int,
        required=True,
        help="Retained Value dimensions per physical KV head",
    )
    parser.add_argument(
        "--work-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument(
        "--factor-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--encoder-initialization",
        choices=(
            "activation-weighted-svd",
            "weight-only-svd",
            "random-orthogonal",
        ),
        default="activation-weighted-svd",
        help=(
            "Initial encoder subspace; every choice is followed by the same "
            "closed-form decoder refit"
        ),
    )
    parser.add_argument(
        "--encoder-initialization-seed",
        type=int,
        default=0,
        help=(
            "Base seed for random-orthogonal initialization; the effective "
            "seed is deterministically offset by layer"
        ),
    )
    parser.add_argument(
        "--encoder-sweeps",
        type=int,
        choices=range(0, 129),
        default=6,
        metavar="N",
        help="Exact number of full encoder/decoder ALS sweeps",
    )
    parser.add_argument(
        "--export-checkpoint-sweeps",
        help=(
            "Comma-separated decoder-closed sweep endpoints to export as "
            "complete factor banks; sweep 0 is the initialization decoder closure"
        ),
    )
    parser.add_argument("--covariance-damping", type=float, default=1.0e-7)
    parser.add_argument(
        "--covariance-row-chunk-size",
        type=int,
        help="CPU-offloaded activation rows transferred per covariance update",
    )
    parser.add_argument(
        "--decoder-objective",
        choices=("full_layer", "per_head"),
        default="full_layer",
        help=(
            "full_layer jointly fits all head decoders to aggregate attention output; "
            "per_head reproduces the A3-style sum of independent head-output losses"
        ),
    )
    parser.add_argument(
        "--encoder-cg-mode",
        choices=("fixed", "tolerance"),
        default="fixed",
    )
    parser.add_argument("--encoder-cg-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--encoder-cg-max-iterations",
        type=int,
        help="Maximum CG iterations; defaults to --encoder-cg-fixed-iterations",
    )
    parser.add_argument(
        "--encoder-cg-fixed-iterations",
        type=int,
        default=16,
        help="Fixed CG iteration budget; also the default maximum in tolerance mode",
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    fit = subparsers.add_parser("fit-shard")
    _add_common(fit)
    fit.add_argument("--layer-shard-index", type=int, default=0)
    fit.add_argument("--layer-shard-count", type=int, default=1)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--torch-num-threads", type=int, default=4)
    fit.add_argument("--resume", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--layers", default="all")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    # Imports are delayed so --help does not initialize the numerical runtime.
    from basisserve.compression import ValueGeometry
    from basisserve.compression.value import fit_shard, merge

    if args.command_name == "fit-shard":
        geometry = ValueGeometry.from_snapshot(args.snapshot_dir)
        calibration = json.loads(
            (Path(args.snapshot_dir) / "manifest.json").read_text()
        )["calibration"]
        for option, key in (
            ("fit_windows", "fit_windows"),
            ("validation_windows", "heldout_windows"),
        ):
            if getattr(args, option) is None:
                assert key in calibration, (
                    f"Specify --{option.replace('_', '-')} for an activation snapshot"
                )
                setattr(args, option, int(calibration[key]))
        fit_shard(args, geometry=geometry)
    else:
        geometry = ValueGeometry.from_factors(args.output_dir)
        merge(args, geometry=geometry)


if __name__ == "__main__":
    main()
