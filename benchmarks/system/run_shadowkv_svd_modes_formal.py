"""Formal comparison of the ShadowKV TP SVD modes (basisserve/core/shadowkv_upstream_tp.py) at TP4 on this node:
65536-token prefill + 128 decode, B16, upstream prefill groups of 4. Per arm: request and steady on cohorts 0/1/2,
breakdown on cohort 0, each a fresh torchrun replica, serial (run_v2q8_exp_formal.py's Exp 2 loop). Outputs go to
svd_modes/formal/; formal_v2/ is not touched. A failed run (e.g. OOM) is recorded, never retried with other settings.

Gates, from run_shadowkv_svd_modes_smoke.py: the arm's 8192-token B4 request/steady/breakdown smoke completed, its
teacher-forced check at batch 8 passed (pooled top-1 agreement >= 0.90), and its 65536-token B16 probe completed.

Not started automatically: run only after the requester confirms.

Usage: python benchmarks/system/run_shadowkv_svd_modes_formal.py [--arms a,b,c]
"""

import argparse
import json
import sys

import run_shadowkv_svd_modes_smoke as modes
import run_v2q8_exp_formal as formal


def consistent(arm):
    path = modes.OUT / "checks/tp_consistency/result_teacher_forced_b8.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    probe = formal.result(modes.OUT / "exp2" / f"{arm}_request_b16_t65536") or {}
    return bool(data.get("arms", {}).get(arm, {}).get("passed")) and probe.get("status") == "complete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default=",".join(modes.ARMS))
    args = parser.parse_args()
    assert set(args.arms.split(",")) <= set(modes.ARMS)
    formal.OUT, formal.SMOKE, formal.consistent = modes.OUT / "formal", modes.OUT, consistent
    sys.argv = [sys.argv[0], "--experiments", "2", "--arms", args.arms]
    formal.main()


if __name__ == "__main__":
    main()
