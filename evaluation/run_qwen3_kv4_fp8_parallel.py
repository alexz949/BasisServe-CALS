"""Run independent single-GPU PPL arms without changing the frozen evaluator."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from basisserve.core.qwen3_kv4_fp8_quality import install_factors, install_nuq4_hooks
from evaluation.eval_qwen3_kv4_fp8_ppl import (
    ARMS, calibrate_fp8, evaluate, fit_quantizers, write,
)
from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids


def worker(output, rank, task, arm):
    protocol = json.loads((output / "manifest.json").read_text())["protocol"]
    assert protocol["environment"] == "basis" and protocol["tp"] == 1
    assert protocol["seqlen"] == protocol["calibration_length"] == 2048
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_name() == protocol["gpu"]
    assert torch.__version__ == protocol["torch"]
    directory = output / f"r{rank}"
    directory.mkdir(exist_ok=True)
    source = ROOT / "external/KVQuant/quant/kvquant/simquant_module_quantizer.py"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source.parents[2], text=True).strip() == protocol["upstream_commit"]
    spec = importlib.util.spec_from_file_location("kvquant_parallel", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"], local_files_only=True)
    train = _token_ids(tokenizer, "wikitext2", "train", None).reshape(-1)
    test = _token_ids(tokenizer, "wikitext2", "test", None).reshape(-1)
    assert test.numel() // 2048 == protocol["windows"]
    model = AutoModelForCausalLM.from_pretrained(protocol["model"], dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").eval().cuda()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    projections, modules, indices, _ = install_factors(
        model, Path(protocol["checkpoint_root"]) / f"Q3-8B-C1-R{rank}", rank,
    )
    starts = protocol["calibration_starts"]
    if task == "calibrate":
        assert not (directory / "quantizers.pt").exists()
        fit_quantizers(model, upstream, modules, indices, dict.fromkeys(modules, 4), train, starts, 2048, directory)
        write(directory / "calibration_complete.json", dict(status="complete", rank=rank,
            calibration_starts=starts, calibration_length=2048, quantizers="quantizers.pt"))
        print("CALIBRATION_COMPLETE", rank, flush=True)
        return
    assert arm in ARMS and not (directory / f"{arm}.json").exists()
    kv4, fp8 = arm in ("kv4", "kv4_fp8"), arm in ("fp8", "kv4_fp8")
    handles = []
    if kv4:
        quantizers = torch.load(directory / "quantizers.pt", map_location="cpu", weights_only=False)
        handles = install_nuq4_hooks(upstream, quantizers, modules, indices)
    if fp8:
        scales = calibrate_fp8(model, projections, train, starts, 2048)
        write(directory / f"{arm}_scales.json", scales)
    for m in projections.values():
        m.fp8 = fp8
        m.reset_stats()
    started = time.perf_counter()
    metrics = evaluate(model, test, 2048, protocol["windows"], directory / f"{arm}_progress.json")
    metrics.update(wall_seconds=time.perf_counter() - started, quant_kv=kv4, fp8_gemm=fp8,
        fp8_projection_stats={name: dict(calls=m.calls, input_elements=m.elements,
            clipped_elements=int(m.clipped)) for name, m in projections.items()} if fp8 else {})
    if fp8:
        assert all(m.calls == protocol["windows"] for m in projections.values())
    for h in handles:
        h.remove()
    write(directory / f"{arm}.json", metrics)
    print("ARM_COMPLETE", rank, arm, metrics["ppl"], flush=True)


def summarize(output):
    protocol = json.loads((output / "manifest.json").read_text())["protocol"]
    summary = {}
    lines = ["# Qwen3-8B-Base KV4 + FP8 Full PPL", "",
        "Environment: `basis`. Eight independent single-GPU arms; no TP8 collectives.",
        "Full WT2 test: 146 non-overlapping 2048-token windows, B1, 298862 scored tokens per arm.",
        "", "| Nominal rank | Arm | PPL | Delta PPL | Change |", "|---:|---|---:|---:|---:|"]
    for rank in (64, 96):
        directory = output / f"r{rank}"
        results = {arm: json.loads((directory / f"{arm}.json").read_text()) for arm in ARMS}
        baseline = results["bf16"]["ppl"]
        for arm, m in results.items():
            assert m["windows"] == m["requested_windows"] == protocol["windows"]
            assert m["tokens"] == 2047 * protocol["windows"] and m["seqlen"] == 2048
            m.update(delta_ppl=m["ppl"] - baseline, relative_ppl_percent=100 * (m["ppl"] / baseline - 1))
            lines.append(f"| {rank} | {arm} | {m['ppl']:.6f} | {m['delta_ppl']:+.6f} | {m['relative_ppl_percent']:+.3f}% |")
        checkpoint = Path(protocol["checkpoint_root"]) / f"Q3-8B-C1-R{rank}/manifest.json"
        schedule = json.loads(checkpoint.read_text())["compression"]["layer_ranks"]
        assert not (directory / "results.json").exists()
        write(directory / "results.json", dict(status="complete", nominal_rank=rank,
            layer_ranks=schedule, protocol=protocol, results=results))
        summary[str(rank)] = results
    write(output / "summary.json", summary)
    lines += ["", "R64/R96 are adaptive equivalent average ranks. Actual E4M3 W8A8 encoder/decoder GEMMs return BF16; other modules remain BF16.",
        "KV4 is official NUQ4 with outliers, quantize/dequantize quality simulation, not a packed-cache speed measurement.",
        "NUQ4 and frozen FP8 scales are calibrated only on WT2 train (16 x 2048 tokens). No SHA256 checks or TP8 validation.",
        "R64 BF16 and its NUQ4 calibration are reused from the original serial process; other arms use the same evaluator in separate processes.",
        "The anomalous short R64 smoke is not used in this table; see the README and order audit.",
        "Parallel workers may contend for CPU/PCIe; wall_seconds is not a serving latency result.",
        "", "Commands and protocols: `../README.md`, `manifest.json`, `parallel_manifest.json` and `parallel_outcomes.json`.", ""]
    (output / "SUMMARY.md").write_text("\n".join(lines))


def coordinate(output):
    assert output.is_dir() and not (output / "parallel_manifest.json").exists()
    handoff = json.loads((output.parent / "handoff.log").read_text())
    assert handoff["status"] == "handoff" and not Path(f"/proc/{handoff['pid']}").exists()
    protocol = json.loads((output / "manifest.json").read_text())["protocol"]
    assert protocol["ranks"] == [64, 96]
    for original in (ROOT / "evaluation/eval_qwen3_kv4_fp8_ppl.py",
                     ROOT / "basisserve/core/qwen3_kv4_fp8_quality.py",
                     ROOT / "basisserve/kernels/fp8_wire.py",
                     ROOT / "external/KVQuant/quant/kvquant/simquant_module_quantizer.py"):
        assert original.read_bytes() == (output / "source" / original.name).read_bytes()
    shutil.copy2(__file__, output / "source" / Path(__file__).name)
    calibrated = torch.load(output / "r64/quantizers.pt", map_location="cpu", weights_only=False)
    assert set(calibrated) == {f"{layer}.{side}" for layer in range(36) for side in ("k", "v")}
    del calibrated
    # Pin independent tasks; leave GPUs 6/7 for the two arms that need R96 codes.
    tasks = [dict(rank=96, task="calibrate", arm="bf16", gpu=1),
             dict(rank=64, task="evaluate", arm="kv4", gpu=0),
             dict(rank=64, task="evaluate", arm="fp8", gpu=2),
             dict(rank=64, task="evaluate", arm="kv4_fp8", gpu=3),
             dict(rank=96, task="evaluate", arm="bf16", gpu=4),
             dict(rank=96, task="evaluate", arm="fp8", gpu=5),
             dict(rank=96, task="evaluate", arm="kv4", gpu=6),
             dict(rank=96, task="evaluate", arm="kv4_fp8", gpu=7)]
    write(output / "parallel_manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        command=shlex.join(sys.argv), handoff=handoff, tasks=tasks,
        reused=["r64/bf16.json", "r64/quantizers.pt"], protocol="unchanged manifest.json",
        scheduling_only=True, environment="basis", threads_per_worker=2))
    running, finished = [], []
    while tasks or running:
        for task in list(tasks):
            if task["rank"] == 96 and task["arm"] in ("kv4", "kv4_fp8"):
                if not (output / "r96/calibration_complete.json").exists():
                    continue
            directory = output / f"r{task['rank']}"
            directory.mkdir(exist_ok=True)
            label = "calibration" if task["task"] == "calibrate" else task["arm"]
            log = (directory / f"{label}.log").open("x")
            command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                       "--task", task["task"], "--rank", str(task["rank"]), "--arm", task["arm"]]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(task["gpu"]), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record = dict(**task, command=shlex.join(command), pid=process.pid,
                          started_utc=datetime.now(timezone.utc).isoformat(), log=str(directory / f"{label}.log"))
            write(directory / f"{label}_command.json", record)
            running.append((process, log, record))
            tasks.remove(task)
            print("START", record, flush=True)
        for process, log, record in list(running):
            code = process.poll()
            if code is None:
                continue
            log.close()
            record.update(returncode=code, finished_utc=datetime.now(timezone.utc).isoformat())
            finished.append(record)
            running.remove((process, log, record))
            write(output / "parallel_outcomes.json", finished)
            print("FINISH", record, flush=True)
        if any(r["returncode"] != 0 for r in finished):
            tasks.clear()
        if running or tasks:
            time.sleep(2)
    assert len(finished) == 8 and all(r["returncode"] == 0 for r in finished), finished
    summarize(output)
    print("PARALLEL_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/q3-kv4-fp8/formal")
    parser.add_argument("--task", choices=("coordinate", "calibrate", "evaluate"), default="coordinate")
    parser.add_argument("--rank", type=int, choices=(64, 96))
    parser.add_argument("--arm", choices=ARMS, default="bf16")
    args = parser.parse_args()
    if args.task == "coordinate":
        coordinate(args.output.resolve())
    else:
        assert args.rank is not None
        worker(args.output.resolve(), args.rank, args.task, args.arm)


if __name__ == "__main__":
    main()
