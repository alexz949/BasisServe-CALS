"""LRQK TP1 retest after the prefill offload stream-order fix (be7c8a2): one GPU, B1, original V128.

Arms: lrqk_upstream_koff_v128 (exact K offloaded, V on GPU) and lrqk_upstream_kvoff_v128 (K and V offloaded).
Every run is one fresh torchrun replica (W=1) on the one visible GPU; runs are strictly serial.

Phases:
  build        upstream cpp_kernel (LRQK's CPU miss gather), compiled on this node (upstream uses -march=native)
  check        check_lrqk_offload_order.py, read from its result.json
  smoke8k      each arm: 8192 tokens, request, --smoke
  consistency  8192 tokens, row 0 of cohorts 0/1/2: K-only decodes 128 greedy tokens, K+V is fed exactly those
               tokens; compares the K+V argmax with the K-only token and the log-probability of the same token
  probe64k     each arm: 65536 tokens, request, --smoke (the target length reaches decode)
  formal       65536 + 128: request and steady on --cohorts, the two arms alternating in order
  breakdown    diagnostic, not a latency result: 65536 tokens with synchronized component timers, on --cohorts

CPU: 24 threads on 12 physical cores (both hyperthreads) of the GPU's NUMA node, the shape the archived TP1 runs
used. The children are confined to 24 cores x 2 threads of that node; v2q8_affinity then keeps every other one.
Each run directory holds result.json, rank_0.json, run.log, run.json (command, environment, times, exit status,
warnings) and load.jsonl (host / pinned-set / own CPU use and GPU use every 5 s).

  python benchmarks/system/run_lrqk_tp1_retest.py --model SNAPSHOT --phases build,check,smoke8k,consistency,probe64k
  python benchmarks/system/run_lrqk_tp1_retest.py --model SNAPSHOT --phases formal --cohorts 0
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("lrqk_upstream_koff_v128", "lrqk_upstream_kvoff_v128")
COHORTS = (0, 1, 2)
THREADS = 24
WARNING = re.compile(r"warn|recompil|out of memory|OutOfMemory|Error|Traceback", re.IGNORECASE)
SOURCES = ("benchmarks/system/bench_v2q8_tp.py", "benchmarks/system/run_lrqk_tp1_retest.py",
           "benchmarks/system/check_lrqk_offload_order.py", "benchmarks/system/v2q8_affinity.py",
           "benchmarks/system/paper_faithful_compat.py", "basisserve/core/lrqk_upstream.py",
           "external/LRQK/lrqk_attention.py", "external/LRQK/cpp_kernel/take_along_dim_grouped.cpp")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(command, **keywords):
    return subprocess.run(command, capture_output=True, text=True, **keywords).stdout.strip()


def gpu():
    fields = "index,name,uuid,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu," \
             "ecc.errors.uncorrected.aggregate.total"
    row = text(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]).splitlines()[0]
    return dict(zip(fields.split(","), (item.strip() for item in row.split(","))))


def pinned_cpus(device, slot=0):
    """24 cores x 2 hyperthreads of the GPU's NUMA node: (the set given to the children, the 24 they keep).
    slot 1 shifts the set by one core, so two launchers on one NUMA node keep disjoint cores; `spare` are the
    node's cores neither slot uses."""
    bus = device["pci.bus_id"].lower()[4:]
    node = max(int(Path(f"/sys/bus/pci/devices/{bus}/numa_node").read_text()), 0)
    cpus = []
    for item in Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(","):
        left, _, right = item.partition("-")
        cpus.extend(range(int(left), int(right or left) + 1))
    cores, siblings = cpus[:len(cpus) // 2], cpus[len(cpus) // 2:]
    confined = sorted(cores[slot:slot + THREADS] + siblings[slot:slot + THREADS])
    spare = cores[THREADS + 2:] + siblings[THREADS + 2:]
    kept = confined[0::2]  # v2q8_affinity: slot 0 of the CPUs the process may use
    pairs = sorted({Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
                    for cpu in kept})
    return node, confined, kept, pairs, spare


def child_environment(args):
    python = Path(args.python)
    wheels = python.parent.parent / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia"
    headers = ":".join(str(wheels / name / "include") for name in ("cusparse", "cublas", "cusolver", "cufft", "curand"))
    # As the archived upstream-baseline runs (run_v2q8_exp_smoke.environment + UPSTREAM_ENV): PyTorch's default
    # CUDA allocator, torch.compile left on (TORCH_COMPILE_DISABLE unset).
    changed = dict(CUDA_HOME=str(args.cuda_home), TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="16", CPATH=headers,
                   PYTHONPATH=str(ROOT), OMP_NUM_THREADS=str(THREADS), PYTORCH_CUDA_ALLOC_CONF="",
                   PATH=f"{args.cuda_home}/bin:{python.parent}:{os.environ['PATH']}",
                   LD_LIBRARY_PATH=f"{python.parent.parent}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}")
    env = dict(os.environ, **changed)
    env.pop("TORCH_COMPILE_DISABLE", None)
    recorded = dict(changed, CUDA_VISIBLE_DEVICES=env.get("CUDA_VISIBLE_DEVICES"),
                    TORCH_COMPILE_DISABLE=env.get("TORCH_COMPILE_DISABLE"),
                    NVIDIA_TF32_OVERRIDE=env.get("NVIDIA_TF32_OVERRIDE"))
    return env, recorded


def sample_load(root, kept, path, stop):
    """Every 5 s: busy logical CPUs on the host and on the pinned set, CPUs used by this run, GPU use."""
    psutil.cpu_percent(percpu=True)
    used, clock = 0.0, time.time()
    with path.open("w") as handle:
        while not stop.wait(5):
            busy = psutil.cpu_percent(percpu=True)
            table = {process.pid: process.info for process in psutil.process_iter(["ppid", "cpu_times"])}
            members, grown = {root}, True
            while grown:  # the run's process tree
                children = {pid for pid, info in table.items() if info["ppid"] in members} - members
                members, grown = members | children, bool(children)
            total = sum(table[pid]["cpu_times"].user + table[pid]["cpu_times"].system for pid in members
                        if pid in table and table[pid]["cpu_times"])
            device, moment = gpu(), time.time()
            others = text(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
            own = {str(pid) for pid in members}
            handle.write(json.dumps(dict(
                time_utc=now(), host_busy_cpus=round(sum(busy) / 100, 1),
                pinned_busy_cpus=round(sum(busy[cpu] for cpu in kept) / 100, 1),
                own_busy_cpus=round(max(total - used, 0) / (moment - clock), 1) if used else None,
                load_average_1m=os.getloadavg()[0], gpu_utilization=device["utilization.gpu"],
                gpu_memory_used_mib=device["memory.used"],
                other_gpu_processes=[line for line in others.splitlines() if line.split(",")[0].strip() not in own]))
                + "\n")
            handle.flush()
            used, clock = total, moment


def run(args, state, name, script, output, torchrun=True, cwd=ROOT, extra_env=None):
    """One fresh process; a relaunch never inherits an earlier attempt's result files."""
    output.mkdir(parents=True, exist_ok=True)
    for stale in [output / "result.json", *output.glob("rank_*.json")]:
        stale.unlink(missing_ok=True)
    state["port"] += 1
    command = [str(args.python)] + (["-m", "torch.distributed.run", "--nproc-per-node=1",
                                     f"--master-port={state['port']}"] if torchrun else []) + script
    env, recorded = child_environment(args)
    env.update(extra_env or {})
    recorded.update(extra_env or {})
    confined = state["confined"]
    started, begin = now(), time.time()
    with (output / "run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   preexec_fn=lambda: os.sched_setaffinity(0, confined))
        stop = threading.Event()
        sampler = threading.Thread(target=sample_load, args=(process.pid, state["kept"],
                                                             output / "load.jsonl", stop))
        sampler.start()
        code = process.wait()
        stop.set()
        sampler.join()
    lines = (output / "run.log").read_text(errors="replace").splitlines()
    warnings = [line[:300] for line in lines if WARNING.search(line)]
    result = output / "result.json"
    data = json.loads(result.read_text()) if result.exists() else {}
    status = data.get("status", "exit0" if code == 0 else "failed") if code == 0 else "failed"
    samples = [json.loads(line) for line in (output / "load.jsonl").read_text().splitlines()]
    column = lambda key: [row[key] for row in samples if row[key] is not None]
    spare = [row["pinned_busy_cpus"] - row["own_busy_cpus"] for row in samples if row["own_busy_cpus"] is not None]
    record = dict(name=name, output=str(output.relative_to(ROOT)), command=command, environment=recorded,
        cpu_confined_to=confined, started_utc=started, finished_utc=now(), seconds=round(time.time() - begin, 1),
        exit_code=code, status=status, warning_lines=len(warnings), warnings=warnings[:40],
        failure=next((line[:400] for line in reversed(lines) if "Error" in line), "") if status != "complete" else "",
        load=dict(samples=len(samples),
                  host_busy_cpus_max=max(column("host_busy_cpus"), default=None),
                  own_busy_cpus_mean=round(sum(column("own_busy_cpus")) / max(len(column("own_busy_cpus")), 1), 1),
                  pinned_busy_by_others_max=round(max(spare, default=0.0), 1),
                  other_gpu_processes_seen=sorted({item for row in samples for item in row["other_gpu_processes"]})))
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    with (args.output / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps({key: record[key] for key in ("name", "status", "exit_code", "seconds", "warning_lines")}
                     | dict(load=record["load"], failure=record["failure"][:200])), flush=True)
    return record, data


def bench(args, arm, mode, length, cohort, output, smoke=False, forced=None, extra=()):
    return ["benchmarks/system/bench_v2q8_tp.py", "--arm", arm, "--mode", mode, "--length", str(length),
            "--batch", "1", "--cohort", str(cohort), "--inputs", str(args.inputs), "--model", str(args.model),
            "--output", str(output), *extra] + (["--smoke"] if smoke else []) + (
                ["--force-tokens", str(forced)] if forced else [])


def measurement(data):
    return data["ranks"][0]["measurement"] if data.get("ranks") else {}


def source_review():
    """Static review of the file the runs import: the ordered prefill waits for the producer stream, records the
    copy stream on K, V and A_K, and both placements build their prefill class from it."""
    wrapper = (ROOT / "basisserve/core/lrqk_upstream.py").read_text()
    functions = {node.name: ast.get_source_segment(wrapper, node) for node in ast.parse(wrapper).body
                 if isinstance(node, ast.FunctionDef)}
    ordered = functions["ordered_offload_prefill"]
    builder = (ROOT / "benchmarks/system/bench_v2q8_tp.py").read_text()
    return dict(
        copy_stream_waits_for_producer="self.stream_temp.wait_stream(torch.cuda.current_stream(" in ordered,
        sources_record_copy_stream="for tensor in (self.Kgpu, self.Vgpu, self._A_K.data):" in ordered
                                   and "tensor.record_stream(self.stream_temp)" in ordered,
        k_only_uses_ordered="class KOnlyOffloadPrefill(ordered_offload_prefill(), KOnlyFactory)" in functions["make_classes"],
        kv_uses_ordered="factory_class = lrqk_upstream.ordered_offload_prefill()" in builder)


LRQK_FILES = ("basisserve/core/lrqk_upstream.py", "external/LRQK/lrqk_attention.py", "benchmarks/system/bench_v2q8_tp.py")


def loaded_as_reviewed(data, hashes, names=LRQK_FILES):
    """The run imported exactly the reviewed files; for LRQK, whether its prefill class derives from the ordered one."""
    imported = data.get("environment", {}).get("imported_sources", {})
    mro = data.get("configuration", {}).get("prefill_class_mro", [])
    return dict(files_match={name: imported.get(name) == hashes[name] for name in names},
                ordered_prefill_in_mro=any(name.endswith("OrderedOffloadPrefill") for name in mro), mro=mro)


def smoke_row(record, data, hashes, expected_tokens):
    measured = measurement(data)
    tokens = measured.get("generated_tokens", [[]])[0]
    logprobs = measured.get("fed_token_logprobs", [[]])[0]
    loaded = loaded_as_reviewed(data, hashes)
    row = dict(name=record["name"], status=record["status"], exit_code=record["exit_code"],
               output_tokens=len(tokens), expected_output_tokens=expected_tokens,
               logprobs=len(logprobs), logprobs_finite=bool(logprobs) and all(math.isfinite(value) for value in logprobs),
               loaded=loaded, effective=data.get("environment", {}).get("effective"), seconds=record["seconds"])
    row["passed"] = bool(row["status"] == "complete" and len(tokens) == expected_tokens and row["logprobs_finite"]
                         and all(loaded["files_match"].values()) and loaded["ordered_prefill_in_mro"])
    return row


def open_session(args, phases, cohorts, arms, sources, review, extra):
    """Pin the runner off the benchmark's cores and write this invocation's manifest. Returns the run state, the
    reviewed source hashes, and whether the GPU and the source review allow running."""
    device = gpu()
    node, confined, kept, pairs, spare = pinned_cpus(device, getattr(args, "cpu_slot", 0))
    state = dict(port=29900 + os.getpid() % 500, confined=confined, kept=kept)
    # The runner and its load sampler stay on physical cores no benchmark slot uses.
    os.sched_setaffinity(0, spare)
    hashes = {name: sha256(ROOT / name) for name in sources}

    import torch
    import transformers
    manifest = dict(
        written_utc=now(), phases=phases, cohorts=cohorts, arms=arms, host=platform.node(),
        slurm={key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_JOB_GPUS",
                                                    "SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE")},
        gpu=device, cpu_model=text(["bash", "-c", "lscpu | sed -n 's/^Model name: *//p'"]),
        cpu=dict(gpu_numa_node=node, children_confined_to=confined, benchmark_keeps=kept, threads=THREADS,
                 physical_cores=len(pairs), hyperthread_pairs=pairs,
                 archived_tp1="24 threads on 12 physical cores x 2 hyperthreads of the GPU's NUMA node "
                              "(Xeon Platinum 8558), OMP_NUM_THREADS=24",
                 host_logical_cpus=psutil.cpu_count(), load_average=os.getloadavg(),
                 scheduler_affinity_of_runner=len(os.sched_getaffinity(0))),
        software=dict(python=platform.python_version(), conda_environment=Path(sys.prefix).name, prefix=sys.prefix,
                      torch=torch.__version__, cuda_runtime=torch.version.cuda, transformers=transformers.__version__,
                      nccl=".".join(map(str, torch.cuda.nccl.version())), cuda_home=str(args.cuda_home),
                      allocator="PyTorch default (PYTORCH_CUDA_ALLOC_CONF empty), as the archived upstream runs"),
        model=dict(path=str(args.model), revision=args.model.name,
                   files={path.name: os.readlink(path).split("/")[-1] for path in sorted(args.model.iterdir())
                          if path.is_symlink()}),
        inputs=dict(path=str(args.inputs), sha256=sha256(args.inputs),
                    manifest={key: value for key, value in json.loads(
                        (args.inputs.parent / "manifest.json").read_text()).items() if key != "cohorts"},
                    cohorts="cohort_0 / cohort_1 / cohort_2, first row of each (three different prompts)"),
        repository=dict(head=text(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                        contains_fix_be7c8a2=subprocess.run(
                            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                             "be7c8a2349b53d260a098a16e3656cc0b7e441c7", "HEAD"]).returncode == 0,
                        uncommitted_files=text(["git", "-C", str(ROOT), "status", "--short", "-uno"]).splitlines(),
                        uncommitted_diff_sha256=hashlib.sha256(
                            text(["git", "-C", str(ROOT), "diff", "HEAD"]).encode()).hexdigest(),
                        lrqk_head=text(["git", "-C", str(ROOT / "external/LRQK"), "rev-parse", "HEAD"]),
                        lrqk_clean=text(["git", "-C", str(ROOT / "external/LRQK"), "status", "--short", "-uno"]) == ""),
        source_sha256=hashes, source_review=review,
        **extra,
        protocol="TP1, B1, original V128/Wo, BF16; same-shape warmup request then fresh state; request = 128 "
                 "tokens; steady = 16 conditioning + 128 measured steps; no component timers")
    (args.output / f"manifest_{'_'.join(phases)}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(gpu=device, numa=node, kept=kept, pairs=pairs, review=review)), flush=True)
    if int(device["ecc.errors.uncorrected.aggregate.total"] or 0) > 0 or not all(review.values()):
        print("STOP: GPU has uncorrected ECC errors or the source review failed", flush=True)
        return state, hashes, False
    return state, hashes, True


def main():
    global ARMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", required=True)
    parser.add_argument("--cohorts", default="0,1,2")
    parser.add_argument("--arms", default=",".join(ARMS), help="K-only arm, then K+V arm (trial arms: give --output too)")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path,
                        default=ROOT / "results/rebuttal/v2q8_l40s_sparse/tp4_64k/inputs/prompts.safetensors")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rebuttal/v2q8_l40s_sparse/tp1_lrqk_retest")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cuda-home", type=Path, required=True)
    args = parser.parse_args()
    ARMS = tuple(args.arms.split(","))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    phases, cohorts = args.phases.split(","), [int(item) for item in args.cohorts.split(",")]
    review = source_review()
    state, hashes, fit = open_session(args, phases, cohorts, ARMS, SOURCES, review, dict(
        lrqk=dict(rank=32, topk=2048, lite=64, iterations=[2, 2], tolerance=[0.01, 0.01])))
    if not fit:
        return

    correctness_path = args.output / "correctness-summary.json"
    correctness = json.loads(correctness_path.read_text()) if correctness_path.exists() else {}
    correctness.update(source_review=review, source_sha256=hashes)

    if "build" in phases:
        run(args, state, "build_cpp_kernel", ["setup.py", "build_ext", "--inplace"],
            args.output / "checks/build_cpp_kernel", torchrun=False, cwd=ROOT / "external/LRQK/cpp_kernel")
    if "check" in phases:
        folder = args.output / "checks/offload_order"
        record, data = run(args, state, "check_offload_order",
                           ["benchmarks/system/check_lrqk_offload_order.py", "--output", str(folder)], folder,
                           torchrun=False)
        ordered = data.get("ordered") or {}
        correctness["offload_order"] = dict(
            exit_code=record["exit_code"], ordered=ordered, upstream_unordered=data.get("upstream"),
            passed=bool(ordered) and all(ordered[name]["wrong"] == 0 and ordered[name]["total"] > 0
                                         for name in ("K", "V", "A_K")),
            scope="Source tensors are kept alive by the script: it isolates the producer-to-copy stream ordering "
                  "and is not a full allocator-lifetime regression test.")
    if "smoke8k" in phases:
        correctness["smoke_8k"] = []
        for arm in ARMS:
            folder = args.output / f"smoke/{arm}_request_b1_t8192_smoke"
            record, data = run(args, state, folder.name, bench(args, arm, "request", 8192, 0, folder, smoke=True), folder)
            correctness["smoke_8k"].append(smoke_row(record, data, hashes, 8))
    if "consistency" in phases:
        rows = []
        for cohort in COHORTS:
            base = args.output / "consistency"
            reference = base / f"{ARMS[0]}_reference_t8192_c{cohort}"
            record, data = run(args, state, reference.name, bench(args, ARMS[0], "request", 8192, cohort, reference),
                               reference)
            source = measurement(data)
            tokens, expected = source.get("generated_tokens", [[]])[0], source.get("fed_token_logprobs", [[]])[0]
            forced = reference / "continuation.json"
            forced.write_text(json.dumps([tokens]))
            followed = base / f"{ARMS[1]}_forced_t8192_c{cohort}"
            record_forced, data_forced = run(args, state, followed.name,
                                             bench(args, ARMS[1], "request", 8192, cohort, followed, forced=forced), followed)
            target = measurement(data_forced)
            argmax, actual = target.get("generated_tokens", [[]])[0], target.get("fed_token_logprobs", [[]])[0]
            complete = bool(record["status"] == record_forced["status"] == "complete" and len(tokens) == 128
                            and len(argmax) == 128 and len(expected) == len(actual) == 127
                            and target.get("teacher_forced") is True)
            difference = [abs(left - right) for left, right in zip(expected, actual)]
            rows.append(dict(
                cohort=cohort, complete=complete, reference=record["output"], forced=record_forced["output"],
                steps=len(argmax), argmax_matches=sum(left == right for left, right in zip(tokens, argmax)),
                top1_agreement=sum(left == right for left, right in zip(tokens, argmax)) / max(len(tokens), 1),
                first_argmax_mismatch=next((index for index, (left, right) in enumerate(zip(tokens, argmax))
                                            if left != right), None),
                logprob_steps=len(difference), logprob_max_abs_difference=max(difference, default=None),
                logprob_mean_abs_difference=sum(difference) / len(difference) if difference else None,
                logprobs_finite=all(math.isfinite(value) for value in expected + actual),
                loaded=[loaded_as_reviewed(data, hashes), loaded_as_reviewed(data_forced, hashes)]))
        steps = sum(row["steps"] for row in rows)
        correctness["placement_consistency"] = dict(
            method="Fixed K-only (lrqk_upstream_koff_v128) decodes 128 greedy tokens from an 8192-token prompt; fixed "
                   "K+V (lrqk_upstream_kvoff_v128) is fed exactly those tokens. Compared at every position on the "
                   "same prefix: the K+V argmax against the K-only token (position 0 is the prefill prediction), "
                   "and the log-probability each placement gives the same fed token (127 decode steps).",
            reference_for_the_gate="TP8 check reached top-1 100% and max log-probability difference 0",
            cohorts=rows, pooled_top1_agreement=sum(row["argmax_matches"] for row in rows) / max(steps, 1),
            pooled_logprob_max_abs_difference=max((row["logprob_max_abs_difference"] for row in rows
                                                   if row["logprob_max_abs_difference"] is not None), default=None),
            exact=all(row["complete"] and row["argmax_matches"] == row["steps"] == 128
                      and row["logprob_max_abs_difference"] == 0 for row in rows),
            scope="Implementation and placement consistency only; not a quality comparison with Dense.")
    if "probe64k" in phases:
        correctness["probe_64k"] = []
        for arm in ARMS:
            folder = args.output / f"smoke/{arm}_request_b1_t65536_smoke"
            record, data = run(args, state, folder.name, bench(args, arm, "request", 65536, 0, folder, smoke=True), folder)
            row = smoke_row(record, data, hashes, 8)
            measured = measurement(data)
            row.update(decode_ready_seconds=measured.get("batch_ready_seconds"),
                       request_seconds=measured.get("request_seconds"),
                       memory_after_prefill=measured.get("memory_after_prefill"),
                       memory_after_decode=measured.get("memory_after_decode"))
            correctness["probe_64k"].append(row)
    correctness["written_utc"] = now()
    correctness_path.write_text(json.dumps(correctness, indent=2) + "\n")

    if "formal" in phases:
        for mode in ("request", "steady"):
            for cohort in cohorts:
                for arm in (ARMS if cohort % 2 == 0 else ARMS[::-1]):  # alternate which arm goes first
                    folder = args.output / f"formal/{arm}_{mode}_b1_t65536_c{cohort}"
                    record, data = run(args, state, folder.name, bench(args, arm, mode, 65536, cohort, folder), folder)
                    loaded = loaded_as_reviewed(data, hashes)
                    print(json.dumps(dict(name=folder.name, loaded_ok=all(loaded["files_match"].values())
                                          and loaded["ordered_prefill_in_mro"])), flush=True)
    if "breakdown" in phases:
        # Diagnostic only: synchronized component timers serialize the work, so these are not latency results.
        for arm in ARMS:
            for cohort in cohorts:
                folder = args.output / f"breakdown/{arm}_breakdown_b1_t65536_c{cohort}"
                run(args, state, folder.name, bench(args, arm, "breakdown", 65536, cohort, folder), folder)
    print(f"done {now()}", flush=True)


if __name__ == "__main__":
    main()
