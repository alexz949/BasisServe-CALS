"""Formal V2Q8 experiments (V128 first), run after the smoke (run_v2q8_exp_smoke.py) on this node.

  Exp 1  TP1 (W=1, GPU 0 only, serial): 130048-token prefill + 128 decode, B1
  Exp 2  TP4 (W=4): 65536-token prefill + 128 decode, B16
Round 2 (formal_v2/, smoke_v2/): arms basis_koff_v128, basis_koff_v128_reuse, shadowkv_upstream_v128 and
lrqk_upstream_koff_v128 (the upstream baseline runtimes, unmodified; TP4 through the minimal ports in
basisserve/core; PyTorch's default CUDA allocator for them, as upstream). Per arm: request and steady on cohorts
0/1/2, breakdown on cohort 0. Each run is a fresh torchrun replica. Round 1 results stay in formal/.

Gates (from the smoke results): an arm runs only if its 8192-token request/steady/breakdown smoke completed for
that experiment and, in Exp 2, its TP4 path passed the teacher-forced TP1-vs-TP4 check (check_v2q8_tp_consistency:
pooled top-1 agreement >= 0.90 over three prompts; every arm, BasisKV included). A run that fails (e.g. OOM) is
recorded, never retried with other settings. Timing runs are serial.

Usage: python benchmarks/system/run_v2q8_exp_formal.py [--experiments 1,2]
"""

import argparse
import json
from pathlib import Path
import time

import run_v2q8_exp_smoke as smoke

ROOT = smoke.ROOT
OUT = ROOT / "results/rebuttal/v2q8_l40s_sparse/formal_v2"
SMOKE = smoke.OUT
SETUP = {1: dict(gpus=[0], length=130048, batch=1), 2: dict(gpus=[0, 1, 2, 3], length=65536, batch=16)}


def result(directory):
    path = directory / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def consistent(arm):
    """Teacher-forced TP1-vs-TP4 check (check_v2q8_tp_consistency): pooled top-1 agreement >= 0.90."""
    path = SMOKE / "checks/tp_consistency/result_teacher_forced.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    return bool(data.get("arms", {}).get(arm, {}).get("passed"))


def adapter_checks(arm, experiment):
    parts = dict(tp_consistency_teacher_forced=consistent(arm)) if experiment == 2 else {}
    return all(parts.values()), parts


def gates(experiment):
    stage, batch = ("exp1", 1) if experiment == 1 else ("exp2", 4)
    decisions = {}
    for arm in smoke.ARMS:
        smoke_ok = all((result(SMOKE / stage / f"{arm}_{mode}_b{batch}_t8192") or {}).get("status") == "complete"
                       for mode in ("request", "steady", "breakdown"))
        checks_ok, parts = adapter_checks(arm, experiment)
        decisions[arm] = dict(run=smoke_ok and checks_ok, smoke_complete=smoke_ok, adapter_checks_passed=checks_ok,
                              adapter_checks=parts, environment=smoke.arm_env(arm))
    return decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="1,2")
    args = parser.parse_args()
    smoke.OUT = OUT  # logs and outputs of formal runs go under formal/
    records, plan = [], {}
    for experiment in (int(e) for e in args.experiments.split(",")):
        setup, decision = SETUP[experiment], gates(experiment)
        plan[f"exp{experiment}"] = decision
        for arm in smoke.ARMS:
            if not decision[arm]["run"]:
                records.append(dict(name=f"exp{experiment}_{arm}", status="skipped_by_smoke_gate", gate=decision[arm]))
                print(json.dumps(records[-1]), flush=True)
                continue
            runs = [(mode, cohort) for cohort in (0, 1, 2) for mode in ("request", "steady")] + [("breakdown", 0)]
            for mode, cohort in runs:
                name = f"exp{experiment}_{arm}_{mode}_b{setup['batch']}_t{setup['length']}_c{cohort}"
                output = OUT / f"exp{experiment}" / f"{arm}_{mode}_b{setup['batch']}_t{setup['length']}_c{cohort}"
                if (result(output) or {}).get("status") == "complete":
                    continue  # resume: completed runs are kept
                script = ["benchmarks/system/bench_v2q8_tp.py", "--arm", arm, "--mode", mode,
                          "--length", str(setup["length"]), "--batch", str(setup["batch"]), "--cohort", str(cohort),
                          "--inputs", str(smoke.INPUTS[len(setup["gpus"])]), "--output", str(output), *smoke.COMMON]
                extra = smoke.arm_env(arm)
                records.append(smoke.run(name, smoke.torchrun(script, setup["gpus"]), setup["gpus"], output,
                                         extra=extra))
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (OUT / f"plan_{stamp}.json").write_text(json.dumps(plan, indent=2) + "\n")
    (OUT / f"runner_{stamp}.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"formal done {stamp}", flush=True)


if __name__ == "__main__":
    main()
