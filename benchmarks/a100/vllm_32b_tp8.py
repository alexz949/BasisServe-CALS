"""Paper Figure 3, Qwen3-32B (TP8 vLLM serving, dense vs C1 V64) on 8 x A100-SXM4-80GB (NVSwitch).

Runs the archived L40S runtime of `system_benchmarks/vllm_32b_tp8_sm89_4k_budget8k` (the paper's 1.31x V64 point):
a git worktree of its base commit with the archived `source.tar` overlaid, exactly as that run executed. Same model,
factor bank, prompts, scheduler budget, CUDA Graph mode, warmups and repeats. Variants (each its own worktree and
output root; every change recorded with before/after sha256 in `runtime_manifest.json`):

  v64            one asserted exact replacement in `prepared_c1_output.py`: the C1 decoder GEMM takes the gathered
                 feature-major arena through a transposed view; on A100 cuBLAS picks align-2 kernels for that operand
                 at small token counts (2.3-3x slower at 2/4/16 tokens; cuBLASLt likewise), so at <= 16 tokens the
                 arena is copied token-major first. Same product; above 16 tokens unchanged.
  v64_nccl_symm  `prepared_c1_output.py` replaced by the repository version (archived and repository files pinned by
                 sha256): the same GEMM change on SM80, and the decode AllGather runs in place on one NCCL
                 symmetric-window arena, so NCCL uses its one-shot NVLink kernels instead of the ring (prefill keeps
                 the prepared NCCL plan). An exploratory smoke with vLLM's custom all-gather instead is kept in
                 `results/system_benchmarks/a100/vllm_32b_tp8_v64_custom_ag` (its runtime_manifest pins that file).

Not changed: vLLM picks its own all-reduce (custom all-reduce on NVLink; the L40S PCIe node could only use PyNccl).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
REPO, REVISION = "alexz949/BasisServe-CALS", "63a35906f354cc239339465db97fab136ccdb1de"
SOURCE, SOURCE_SHA256 = ("system_benchmarks/vllm_32b_tp8_sm89_4k_budget8k/source.tar",
                         "9a92fca236a1ee152d9cdf8a729e6faf3a51a403d57e54396c780d3dc2e789f3")
BASE_COMMIT = "bab891d3e108d4280eeee65e7bb6c07f86dea83a"
MODEL_REPO, MODEL_REVISION = "Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"
FACTOR_REVISION, FACTOR_DIR = "0872566b1da66eb4c813d7a1cb3313325f22b287", "ICLR-results/qwen3-32b/c1/factor-banks/R64-S6"
FACTOR_RESULTS_SHA256 = "26bafc0674362208b05d56b16f8b5a74a22756738cf355af39bf80b859195b3f"
CUDA_HOME = Path.home() / "cuda-13.0"  # benchmarks/a100/setup_cuda_home.sh (FlashInfer JIT at engine start)
PHASES = {
    # 128 keeps vLLM's 3D-attention threshold at 128 as in the full sweep (it snaps to the nearest capture size).
    "smoke": dict(batch_sizes=[1, 2, 4, 16, 128], warmups=1, repeats=1, profile_batches=[]),
    "formal": dict(batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256], warmups=1, repeats=3, profile_batches=[1, 32, 256]),
}
ARCHIVED_C1_OUTPUT_SHA256 = "4fa68335291ef41b4fd72d201f2c5fc8a922bdbe5a2a8248dd1fbffcf50b2404"
VARIANTS = {
    "v64": dict(patches=[dict(
        file="basisserve/vllm/prepared_c1_output.py",
        old=("        arena = prepared.gather_inplace_fast()\n"
             "        return torch.mm(arena.T, decoder)\n"),
        new=("        arena = prepared.gather_inplace_fast()\n"
             "        # A100 (SM80) cuBLAS picks align-2 kernels for the transposed feature-major operand at small token\n"
             "        # counts (2.3-3x slower at 2/4/16 tokens); a token-major copy restores the fast path there.\n"
             "        tokens = arena.T if arena.shape[1] > 16 else arena.T.contiguous()\n"
             "        return torch.mm(tokens, decoder)\n"),
        count=1, reason="A100 cuBLAS kernel selection for the small-token decoder GEMM")], replace={}),
    "v64_nccl_symm": dict(patches=[], replace={
        "basisserve/vllm/prepared_c1_output.py": (
            ARCHIVED_C1_OUTPUT_SHA256, "1723d536ed8735c4343718332abea64f64b0ee2f64deee94c061b8e51fc4a672",
            "decode AllGather in place on an NCCL symmetric-window arena (one-shot NVLink kernels) "
            "+ SM80 small-token decoder GEMM layout")}),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def prepare_runtime(variant, output_root):
    runtime = Path.home() / "basiskv-runtimes" / f"vllm_32b_tp8_{variant}"
    archive = Path(hf_hub_download(REPO, SOURCE, revision=REVISION))
    assert sha256(archive.read_bytes()) == SOURCE_SHA256
    if not runtime.exists():
        runtime.parent.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(runtime), BASE_COMMIT)
    assert git("rev-parse", "HEAD", cwd=runtime) == BASE_COMMIT
    with tarfile.open(archive) as tar:
        sources = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    records = []
    for patch in VARIANTS[variant]["patches"]:
        text = sources[patch["file"]].decode()
        assert text.count(patch["old"]) == patch["count"], patch["file"]
        before = sha256(sources[patch["file"]])
        sources[patch["file"]] = text.replace(patch["old"], patch["new"]).encode()
        records.append(dict(patch, sha256_before=before, sha256_after=sha256(sources[patch["file"]])))
    for name, (before, after, reason) in VARIANTS[variant]["replace"].items():
        assert sha256(sources[name]) == before, name
        sources[name] = (ROOT / name).read_bytes()
        assert sha256(sources[name]) == after, f"repository {name} changed; update the pinned sha256"
        records.append(dict(file=name, replaced_with=f"repository {name}", reason=reason,
                            sha256_before=before, sha256_after=after))
    for name, data in sources.items():
        path = runtime / name
        assert ".." not in name and path.resolve().is_relative_to(runtime.resolve())
        if path.exists() and path.read_bytes() != data:
            # Only a pristine base-commit file may be replaced; anything else is an unexpected local edit.
            pristine = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:{name}"], cwd=runtime)
            assert path.read_bytes() == pristine, name
        path.write_bytes(data)
    model = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    factors = Path(snapshot_download(REPO, revision=FACTOR_REVISION, allow_patterns=[f"{FACTOR_DIR}/*"])) / FACTOR_DIR
    assert sha256((factors / "results.json").read_bytes()) == FACTOR_RESULTS_SHA256
    manifest = dict(hf_repo=REPO, source=SOURCE, source_sha256=SOURCE_SHA256, source_hf_revision=REVISION,
                    base_commit=BASE_COMMIT, runtime=str(runtime), runtime_files=sorted(sources), patches=records,
                    runtime_git_status=git("status", "--short", cwd=runtime),
                    model=dict(repo=MODEL_REPO, revision=MODEL_REVISION, path=str(model)),
                    factors=dict(revision=FACTOR_REVISION, path=str(factors), results_sha256=FACTOR_RESULTS_SHA256),
                    cuda_home=str(CUDA_HOME), host_compiler="/usr/bin/g++-11")
    path = output_root / "runtime_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(manifest))
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return model, factors, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="v64")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    assert (CUDA_HOME / "include/curand.h").is_file() and Path("/usr/bin/g++-11").is_file()
    output_root = (args.output_root or ROOT / f"results/system_benchmarks/a100/vllm_32b_tp8_{args.variant}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model, factors, runtime = prepare_runtime(args.variant, output_root)
    output = output_root / args.phase
    output.mkdir(exist_ok=True)
    if args.phase == "formal":
        assert all(json.loads((output_root / f"smoke/{arm}.json").read_text())["status"] == "complete"
                   for arm in ("dense", "c1"))
    settings = PHASES[args.phase]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(runtime), PATH=f"{Path(sys.executable).parent}:{env['PATH']}", CUDA_HOME=str(CUDA_HOME),
               CC="/usr/bin/gcc-11", CXX="/usr/bin/g++-11", OMP_NUM_THREADS="1",
               VLLM_WORKER_MULTIPROC_METHOD="spawn", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
    for arm in ("dense", "c1"):
        result = output / f"{arm}.json"
        if result.exists():
            assert json.loads(result.read_text())["status"] == "complete", f"Inspect {result} before resuming"
            continue
        command = [sys.executable, "evaluation/benchmark_vllm_qwen3_8b_c1.py", "--arm", arm, "--model", str(model),
                   "--factor-dir", str(factors), "--factor-validation", "structure", "--output-dir", str(output),
                   "--batch-sizes", *map(str, settings["batch_sizes"]), "--prefill-tokens", "4096",
                   "--decode-tokens", "128", "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.8",
                   "--warmups", str(settings["warmups"]), "--repeats", str(settings["repeats"]),
                   "--profile-batches", *map(str, settings["profile_batches"])]
        print(shlex.join(command), flush=True)
        with (output / f"{arm}.log").open("a") as log:
            subprocess.run(command, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert json.loads(result.read_text())["status"] == "complete"
    with (output / "summary.log").open("a") as log:
        subprocess.run([sys.executable, "evaluation/summarize_vllm_qwen3_8b_c1.py", "--input-dir", str(output)],
                       cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print((output / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
