"""Kernel-only timing of bounded prefill staging; includes dequantization and copies."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import triton

from basisserve.kernels.nuq4_attention import nuq4_attention
from basisserve.kernels.nuq4_decode import decode_workspace, nuq4_decode
from basisserve.kernels.nuq4_prefill import PrefillWorkspace, plan_prefill, staged_prefill
from evaluation.tune_nuq4_prefill import fixture


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 96])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for width in args.widths:
        for shape in ("prefill", "mixed"):
            q, kc, vc, rope, cu, lens, table = fixture(width, shape)
            reference = nuq4_attention(q, kc, vc, rope, cu, lens, table)
            output = torch.empty_like(reference)
            decode_scratch = decode_workspace(lens.numel(), width, q.device)
            workspace = PrefillWorkspace(q.shape[0], rope.shape[0], q.device)
            # Fixtures supply GPU metadata. This setup is outside kernel timing;
            # serving obtains these CPU values directly from the scheduler.
            chunks = plan_prefill(cu.cpu(), lens.cpu(), lens, workspace.capacity)

            def original():
                nuq4_attention(q, kc, vc, rope, cu, lens, table, out=output, prefill_only=True)
                nuq4_decode(q, kc, vc, rope, cu, lens, table, out=output, workspace=decode_scratch)

            def staged():
                staged_prefill(q, kc, vc, rope, cu, lens, table, chunks, workspace, out=output)
                nuq4_decode(q, kc, vc, rope, cu, lens, table, out=output, workspace=decode_scratch)

            for name, run in (("packed", original), ("staged", staged)):
                run()
                torch.testing.assert_close(output, reference, atol=0.008, rtol=0.015)
                row = dict(width=width, shape=shape, path=name,
                    ms=triton.testing.do_bench_cudagraph(run, rep=50),
                    workspace_bytes=workspace.nbytes, chunks=len(chunks),
                    max_abs=float((output.float()-reference.float()).abs().max()))
                rows.append(row)
                with args.output.with_suffix(".jsonl").open("a") as handle:
                    handle.write(json.dumps(row)+"\n")
                print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(dict(environment="basis", rows=rows), indent=2)+"\n")


if __name__ == "__main__":
    main()
