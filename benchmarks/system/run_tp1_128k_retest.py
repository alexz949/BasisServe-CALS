"""TP1 / 128K retest on one GPU: ShadowKV with and without its per-layer pre-SVD cleanup, and the fixed LRQK.

Llama-3.1-8B-Instruct, BF16, original V128/Wo, B1, 130048 prompt tokens (the TP1 Exp 1 inputs: tp1_offline
cohorts 0/1/2). Arms:
  basis_koff_v128_reuse_opt    BasisKV as in round 2: exact K offloaded, Base read from V, lean forward, fast router
  shadowkv_upstream_v128       upstream ShadowKV, the control
  shadowkv_upstream_nohk_v128  ShadowKV with per-layer pre-SVD GC and empty-cache calls disabled (not upstream)
  lrqk_upstream_koff_v128      LRQK with the prefill stream-order fix, exact K offloaded, V on GPU
  lrqk_upstream_kvoff_v128     LRQK with the fix, K and V offloaded
Every run is one fresh torchrun replica (W=1) on the one visible GPU; runs are strictly serial. CPU pinning, the
child environment, load sampling and the per-run records are those of run_lrqk_tp1_retest.py.

Phases:
  build    upstream ShadowKV CUDA kernels and LRQK cpp_kernel, compiled on this node
  checks   check_shadowkv_cleanup_scope.py for both ShadowKV arms (call counts, not timed) and
           check_lrqk_offload_order.py; both judged from their result.json
  smoke8k  8192 tokens (--groups): ShadowKV, the control decodes 8 tokens and the modified arm is fed exactly those
           tokens; BasisKV, one short run (which also compiles its kernels)
  probe    130048 tokens, full prefill and 8 tokens (--groups); BasisKV alone, and per pair (ShadowKV control ->
           modified; LRQK K-only -> K+V):
           the first arm decodes greedily and the second is fed its tokens; if the first fails, the second runs
           on its own. A failure is recorded with its kind (out of memory or other) and the phase it happened in.
  formal   130048 + 128: --modes (request, steady) on --cohorts for the arms that passed the probe (or --arms); the two
           arms of a pair run back to back and alternate which goes first; an arm that fails is not run again

  python benchmarks/system/run_tp1_128k_retest.py --model SNAPSHOT --cuda-home CUDA --phases build,checks,smoke8k,probe
  python benchmarks/system/run_tp1_128k_retest.py --model SNAPSHOT --cuda-home CUDA --phases formal --cohorts 0
"""

import argparse
import ast
import json
import math
from pathlib import Path
import re
import sys

import run_lrqk_tp1_retest as base

ROOT = base.ROOT
BASIS = "basis_koff_v128_reuse_opt"
SHADOWKV = ("shadowkv_upstream_v128", "shadowkv_upstream_nohk_v128")
LRQK = ("lrqk_upstream_koff_v128", "lrqk_upstream_kvoff_v128")
LABELS = {BASIS: "BasisKV, exact K offloaded (Base read from V, lean forward, fast router)",
          SHADOWKV[0]: "ShadowKV, upstream (control)",
          SHADOWKV[1]: "ShadowKV with per-layer pre-SVD GC and empty-cache calls disabled",
          LRQK[0]: "LRQK with the prefill stream-order fix, K offload (V on GPU)",
          LRQK[1]: "LRQK with the prefill stream-order fix, K and V offload"}
SHADOWKV_FILES = ("external/ShadowKV/models/kv_cache.py", "external/ShadowKV/models/base.py",
                  "external/ShadowKV/models/llama.py", "basisserve/core/shadowkv_upstream_tp.py",
                  "benchmarks/system/bench_v2q8_tp.py")
BASIS_FILES = ("benchmarks/system/v2q8_basis.py", "basisserve/core/lean_llama.py",
               "benchmarks/system/table3_page_router.py", "benchmarks/system/bench_v2q8_tp.py")
SOURCES = base.SOURCES + SHADOWKV_FILES[:4] + BASIS_FILES[:3] + (
    "external/ShadowKV/models/tensor_op.py", "benchmarks/system/run_tp1_128k_retest.py",
    "benchmarks/system/check_shadowkv_cleanup_scope.py", "benchmarks/system/table3_router_fast.cu",
    "benchmarks/system/table3_page_select.cu")


def launch(args, arm):
    """Per-method arguments and environment, as the archived runs: BasisKV reads its V128 router bank, compiles its
    kernels under the build root and runs with expandable_segments; the upstream baselines use PyTorch's default
    CUDA allocator."""
    if arm == BASIS:
        return dict(extra=("--bank-v128", str(args.bank_v128), "--build-root", str(args.build_root)),
                    extra_env=dict(PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
    return dict(extra=(), extra_env=None)


def one(args, state, folder, arm, mode, length, cohort, smoke=False, forced=None):
    options = launch(args, arm)
    return base.run(args, state, folder.name, base.bench(args, arm, mode, length, cohort, folder, smoke=smoke,
                                                         forced=forced, extra=options["extra"]),
                    folder, extra_env=options["extra_env"])


def shadowkv_review():
    """Static review of what the modified arm changes: upstream get_svd makes one gc.collect and one empty_cache,
    both before its torch.svd and between two synchronize calls; the benchmark replaces only those two, only while
    get_svd runs, and restores them on any exit; H2D keeps its own cleanup."""
    source = (ROOT / "external/ShadowKV/models/kv_cache.py").read_text()
    cache = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.ClassDef) and node.name == "ShadowKVCache_CPU")
    methods = {node.name: ast.get_source_segment(source, node) for node in cache.body
               if isinstance(node, ast.FunctionDef)}
    get_svd, h2d = methods["get_svd"], methods["H2D"]
    builder = (ROOT / "benchmarks/system/bench_v2q8_tp.py").read_text()
    return dict(
        get_svd_has_one_gc_collect_and_one_empty_cache=get_svd.count("gc.collect()") == 1
                                                         and get_svd.count("torch.cuda.empty_cache()") == 1,
        get_svd_cleanup_is_before_the_svd=max(get_svd.index("gc.collect()"), get_svd.index("torch.cuda.empty_cache()"))
                                           < get_svd.index("torch.svd("),
        get_svd_has_two_synchronize=get_svd.count("torch.cuda.synchronize()") == 2,
        h2d_has_its_own_cleanup=h2d.count("gc.collect()") == 2 and h2d.count("torch.cuda.empty_cache()") == 2,
        wrapper_applies_to_get_svd_only=builder.count("without_housekeeping(") == 2
                                         and "cache.get_svd = without_housekeeping(cache.get_svd)" in builder,
        wrapper_restores_on_any_exit="with ExitStack() as restore:" in builder
                                      and 'restore.enter_context(mock.patch.object(gc, "collect"' in builder
                                      and 'restore.enter_context(mock.patch.object(torch.cuda, "empty_cache"' in builder,
        modified_arm_skips_exactly_both='skip_in_get_svd=("gc_collect", "empty_cache"))' in builder)


def failure_of(record, folder):
    """Kind and place of a failed run, from its log: out of memory is named only when the log says so."""
    if record["status"] == "complete":
        return None
    lines = (folder / "run.log").read_text(errors="replace").splitlines()
    phases = [json.loads(line)["phase"] for line in lines if line.startswith('{"arm"') and '"phase"' in line]
    memory = [line for line in lines if "OutOfMemoryError" in line or "CUDA out of memory" in line]
    frames = [line.strip() for line in lines if 'File "' in line and "site-packages" not in line]
    errors = [line.strip() for line in lines if re.search(r"\b\w*(Error|Exception)\b:", line)]
    return dict(kind="out_of_memory" if memory else "other", exit_code=record["exit_code"],
                last_phase_reached=phases[-1] if phases else None,
                message=(memory[0] if memory else errors[0] if errors else "")[:900],
                innermost_repository_frames=frames[-4:], log=str((folder / "run.log").relative_to(ROOT)))


def arm_row(record, data, hashes, folder, expected_tokens):
    measured = base.measurement(data)
    tokens = measured.get("generated_tokens", [[]])[0]
    logprobs = measured.get("fed_token_logprobs", [[]])[0]
    lrqk = record["name"].startswith("lrqk")
    loaded = base.loaded_as_reviewed(data, hashes, base.LRQK_FILES if lrqk else BASIS_FILES
                                     if record["name"].startswith("basis") else SHADOWKV_FILES)
    row = dict(name=record["name"], status=record["status"], exit_code=record["exit_code"], seconds=record["seconds"],
               output_tokens=len(tokens), expected_output_tokens=expected_tokens,
               logprobs_finite=bool(logprobs) and all(math.isfinite(value) for value in logprobs),
               imported_files_match_review=all(loaded["files_match"].values()) if data else None,
               effective=data.get("environment", {}).get("effective"),
               implementation=data.get("configuration", {}).get("implementation"),
               failure=failure_of(record, folder))
    if lrqk:
        row.update(ordered_prefill_in_mro=loaded["ordered_prefill_in_mro"], prefill_class_mro=loaded["mro"])
    if measured:
        row.update(ttft_seconds=measured["ttft_seconds"][0], placement_seconds=measured["placement_seconds"],
                   decode_ready_seconds=measured["batch_ready_seconds"],
                   memory_after_prefill=measured["memory_after_prefill"],
                   memory_after_decode=measured["memory_after_decode"])
    row["passed"] = bool(row["status"] == "complete" and len(tokens) == expected_tokens and row["logprobs_finite"]
                         and row["imported_files_match_review"] and (not lrqk or row["ordered_prefill_in_mro"]))
    return row


def pair(args, state, hashes, stage, arms, length, cohort):
    """First arm greedy, second arm fed its tokens (teacher forced); the comparison is of what the second arm
    predicts and of the log-probability each gives the same token, never of the fed tokens themselves."""
    tag = f"request_b1_t{length}_smoke"
    first = args.output / stage / f"{arms[0]}_{tag}"
    record, data = one(args, state, first, arms[0], "request", length, cohort, smoke=True)
    rows = {arms[0]: arm_row(record, data, hashes, first, 8)}
    forced = None
    if rows[arms[0]]["passed"]:
        forced = first / "continuation.json"
        forced.write_text(json.dumps(base.measurement(data)["generated_tokens"]))
    second = args.output / stage / f"{arms[1]}_{tag}"
    record_second, data_second = one(args, state, second, arms[1], "request", length, cohort, smoke=True,
                                     forced=forced)
    rows[arms[1]] = arm_row(record_second, data_second, hashes, second, 8)
    comparison = dict(done=False, reason="needs both arms to complete" if forced is None or not rows[arms[1]]["passed"]
                      else None)
    if forced is not None and rows[arms[1]]["passed"]:
        source, target = base.measurement(data), base.measurement(data_second)
        tokens, argmax = source["generated_tokens"][0], target["generated_tokens"][0]
        difference = [abs(left - right) for left, right in zip(source["fed_token_logprobs"][0],
                                                               target["fed_token_logprobs"][0])]
        comparison = dict(
            done=True, reference=arms[0], teacher_forced=arms[1], context_tokens=length, cohort=cohort,
            positions=len(argmax), argmax_matches=sum(left == right for left, right in zip(tokens, argmax)),
            first_argmax_mismatch=next((index for index, (left, right) in enumerate(zip(tokens, argmax))
                                        if left != right), None),
            logprob_steps=len(difference), logprob_max_abs_difference=max(difference),
            logprob_mean_abs_difference=sum(difference) / len(difference),
            identical=tokens == argmax and max(difference) == 0 and target.get("teacher_forced") is True,
            method="The second arm is fed the first arm's greedy tokens. Compared on the same prefix: the second "
                   "arm's argmax against the first arm's token at every position (position 0 is the prefill "
                   "prediction), and the log-probability each arm gives the same fed token.")
    return dict(arms=rows, comparison=comparison)


def single(args, state, hashes, stage, arm, length, cohort):
    """An arm with nothing to be compared against: completes, right output length, finite log-probabilities."""
    folder = args.output / stage / f"{arm}_request_b1_t{length}_smoke"
    record, data = one(args, state, folder, arm, "request", length, cohort, smoke=True)
    return dict(arms={arm: arm_row(record, data, hashes, folder, 8)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", required=True)
    parser.add_argument("--cohorts", default="0,1,2")
    parser.add_argument("--arms", default="passed", help="formal: comma list, or 'passed' = arms whose probe passed")
    parser.add_argument("--modes", default="request,steady", help="formal: request, steady, or both")
    parser.add_argument("--groups", default="shadowkv,lrqk", help="smoke8k / probe: basis, shadowkv, lrqk")
    parser.add_argument("--cpu-slot", type=int, default=0, choices=(0, 1),
                        help="1 for a second launcher running at the same time: disjoint cores if on the same NUMA node")
    parser.add_argument("--length", type=int, default=130048,
                        help="prompt tokens of probe and formal; give the matching --inputs and --output")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=ROOT / "results/system_benchmarks/tp1_offline/inputs/prompts.safetensors")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rebuttal/v2q8_l40s_sparse/tp1_128k_retest")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--bank-v128", type=Path, default=ROOT / "results/rebuttal/table3_k_offload_128k/factors/ours_b16r16",
                        help="BasisKV: the Table 3 runtime V128 router bank")
    parser.add_argument("--build-root", type=Path, default=None, help="BasisKV: where its CUDA kernels are compiled")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    phases, cohorts = args.phases.split(","), [int(item) for item in args.cohorts.split(",")]
    correctness_path = args.output / "correctness-summary.json"
    correctness = json.loads(correctness_path.read_text()) if correctness_path.exists() else {}
    probed = correctness.get(f"probe_{args.length}", {})
    passed = [arm for name in ("basis", "shadowkv", "lrqk") for arm, row in probed.get(name, {}).get("arms", {}).items()
              if row["passed"]]
    arms = passed if args.arms == "passed" else args.arms.split(",")
    review = dict(shadowkv=shadowkv_review(), lrqk=base.source_review())
    git = lambda folder: base.text(["git", "-C", str(ROOT / folder), "rev-parse", "HEAD"])
    state, hashes, fit = base.open_session(
        args, phases, cohorts, arms if "formal" in phases else (BASIS,) + SHADOWKV + LRQK, SOURCES,
        {f"{group}:{key}": value for group, rows in review.items() for key, value in rows.items()}, dict(
            arm_labels=LABELS, context_tokens=args.length,
            shadowkv=dict(upstream_head=git("external/ShadowKV"), cutlass_head=git("external/ShadowKV/3rdparty/cutlass"),
                          rank=160, chunk=8, sparse_budget=2048,
                          modified_arm="gc.collect and torch.cuda.empty_cache are no-ops while get_svd runs; "
                                       "upstream's source is not edited"),
            lrqk=dict(rank=32, topk=2048, lite=64, iterations=[2, 2], tolerance=[0.01, 0.01])))
    if not fit:
        return
    correctness.update(source_review=review, source_sha256=hashes)

    if "build" in phases:
        base.run(args, state, "build_shadowkv_kernels", ["setup.py", "build_ext", "--inplace"],
                 args.output / "checks/build_shadowkv_kernels", torchrun=False, cwd=ROOT / "external/ShadowKV")
        base.run(args, state, "build_lrqk_cpp_kernel", ["setup.py", "build_ext", "--inplace"],
                 args.output / "checks/build_lrqk_cpp_kernel", torchrun=False, cwd=ROOT / "external/LRQK/cpp_kernel")
    if "checks" in phases:
        scope = {}
        for arm in SHADOWKV:
            folder = args.output / f"checks/cleanup_scope_{arm}"
            record, data = base.run(args, state, folder.name, [
                "benchmarks/system/check_shadowkv_cleanup_scope.py", "--arm", arm, "--model", str(args.model),
                "--inputs", str(args.inputs), "--output", str(folder)], folder, torchrun=False)
            scope[arm] = dict(exit_code=record["exit_code"], **data)
        counts = [scope[arm].get("calls_inside_get_svd", {}) for arm in SHADOWKV]
        correctness["shadowkv_cleanup_scope"] = dict(
            arms=scope, same_svd_calls=bool(counts[0]) and counts[0].get("svd") == counts[1].get("svd"),
            passed=all(scope[arm].get("passed") for arm in SHADOWKV) and counts[0].get("svd") == counts[1].get("svd"),
            scope="Call counts at 8192 tokens, one arm per process, not timed.")
        folder = args.output / "checks/lrqk_offload_order"
        record, data = base.run(args, state, "check_lrqk_offload_order",
                                ["benchmarks/system/check_lrqk_offload_order.py", "--output", str(folder)], folder,
                                torchrun=False)
        ordered = data.get("ordered") or {}
        correctness["lrqk_offload_order"] = dict(
            exit_code=record["exit_code"], ordered=ordered, upstream_unordered=data.get("upstream"),
            passed=bool(ordered) and all(ordered[name]["wrong"] == 0 and ordered[name]["total"] > 0
                                         for name in ("K", "V", "A_K")),
            scope="Source tensors are kept alive by the script: it isolates the producer-to-copy stream ordering "
                  "and is not a full allocator-lifetime regression test.")
    # Each group is probed once; a later invocation for another group keeps the earlier groups' records.
    groups = dict(basis=lambda stage, length: single(args, state, hashes, stage, BASIS, length, 0),
                  shadowkv=lambda stage, length: pair(args, state, hashes, stage, SHADOWKV, length, 0),
                  lrqk=lambda stage, length: pair(args, state, hashes, stage, LRQK, length, 0))
    selected = args.groups.split(",")
    if "smoke8k" in phases:
        correctness.setdefault("smoke_8k", {}).update(
            {name: groups[name]("smoke", 8192) for name in selected if name != "lrqk"})
    if "probe" in phases:
        correctness.setdefault(f"probe_{args.length}", {}).update(
            {name: groups[name]("probe", args.length) for name in selected})
    correctness["written_utc"] = base.now()
    correctness_path.write_text(json.dumps(correctness, indent=2) + "\n")

    if "formal" in phases:
        stopped_path = args.output / "stopped_arms.json"
        stopped = json.loads(stopped_path.read_text()) if stopped_path.exists() else {}
        for mode in args.modes.split(","):
            for cohort in cohorts:
                # The compared arms run back to back; the order is reversed on odd cohorts.
                order = [arm for arm in (BASIS,) + SHADOWKV + LRQK if arm in arms][::1 if cohort % 2 == 0 else -1]
                for arm in order:
                    folder = args.output / f"formal/{arm}_{mode}_b1_t{args.length}_c{cohort}"
                    if arm in stopped:
                        print(json.dumps(dict(skipped=folder.name, stopped_after=stopped[arm]["run"])), flush=True)
                        continue
                    record, data = one(args, state, folder, arm, mode, args.length, cohort)
                    failure = failure_of(record, folder)
                    if failure:  # recorded as it is; the arm is not timed again
                        stopped[arm] = dict(run=folder.name, **failure)
                        stopped_path.write_text(json.dumps(stopped, indent=2) + "\n")
                        (folder / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
    print(f"done {base.now()}", flush=True)


if __name__ == "__main__":
    main()
