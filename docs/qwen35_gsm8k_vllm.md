# Qwen3.5 gated V + Wo GSM8K via vLLM

Implementation, real GPU smoke, HF reference checks and all four full GSM8K
evaluations are complete. Selected arms: Dense, Uniform V64, Two-sided V64,
Two-sided V64+Wo. Thinking is disabled. Results and actual final commands are
in `docs/qwen35_gsm8k_results.md`; the commands below preserve the initial plan.

## Runtime and scope

The TP1 plugin loads original text weights without changing checkpoint files.
Each KV group's folded V writer emits latent coordinates followed by zeros in
standard 256-wide cache slots. Native vLLM attention aggregates these values;
only current outputs are reconstructed, before the native sigmoid gate.
No decoder is folded across the gate and no historical V is reconstructed.

Wo reuses four source-local encoders, concatenation and the joint decoder after
the native full-attention/GDN gate. This is a TP1 computation of the trained
TP4 layout. No distributed collective executes. This version measures quality
and generation throughput, not compact-cache memory or communication savings.

## Protocol and environment

- All 1,319 GSM8K test questions; 5-shot; seed 20260909; native chat template;
  thinking off; greedy generation; at most 1,024 output tokens.
- Report strict/flexible exact match, per-question responses, length-capped
  counts and prompt identities. Assert no prompt truncation.
- vLLM: at most 32 concurrent sequences, 4,096 batched tokens, context 8,192;
  GPU memory budget starts at 40% of physical device capacity.
- Native GDN recurrence is unchanged; prefix caching is disabled initially.
- Unit tests: four passed in lowrank. vLLM 0.18.1 and lm-eval 0.4.11 are
  installed in lowrankarena/Python 3.13; lowrank/Python 3.10 has no vLLM.
  No environment was modified. The user explicitly approved lowrankarena for
  these vLLM jobs. HF reference and final audits ran in lowrank.
- Two CPU threads per process; local A100 execution, no Slurm. Candidate GPUs
  0, 2, 5 and 6 will be checked again immediately before launch.
- First run four real-engine smoke arms (four different-length prompts,
  32 generated tokens each). Then check token probabilities against HF in
  lowrank. Assess compressed differences relative to dense backend differences.
- After review, run a 16-question graph-enabled pilot per arm before full GSM8K.

## Exact commands

The following function expands to the actual Python command. Independent arms
may run concurrently on separate GPUs. All output files are exclusive-created.

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrankarena
export PYTHONPATH=.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p results/q35_hybrid/gsm8k_vllm
run_arm() {
    local gpu=$1 arm=$2 phase=$3
    shift 3
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m evaluation.eval_qwen35_hybrid_gsm8k_vllm \
        --max-num-seqs 32 --max-num-batched-tokens 4096 \
        --max-new-tokens 1024 --max-model-len 8192 \
        --gpu-memory-utilization 0.40 \
        --output "results/q35_hybrid/gsm8k_vllm/${phase}_${arm}.json" "$@" \
        > "results/q35_hybrid/logs/gsm8k_vllm_${phase}_${arm}.log" 2>&1
}
run_arm 6 dense smoke --smoke --enforce-eager
run_arm 2 uniform64 smoke --smoke --enforce-eager \
    --bank results/q35_hybrid/banks/c1_uniform_v64.pt
run_arm 0 twosided64 smoke --smoke --enforce-eager \
    --bank results/q35_hybrid/banks/c1_twosided_v64.pt
run_arm 5 twosided64_wo smoke --smoke --enforce-eager \
    --bank results/q35_hybrid/banks/c1_twosided_v64.pt \
    --wo-bank results/q35_hybrid/wo_twosided_v64/wo_bank.pt
```

HF reference checks:

```bash
conda activate lowrank
export PYTHONPATH=results/q35_hybrid/deps:.
for arm in dense uniform64 twosided64 twosided64_wo; do
    CUDA_VISIBLE_DEVICES=6 python -u -m evaluation.check_qwen35_vllm_reference \
        --smoke "results/q35_hybrid/gsm8k_vllm/smoke_${arm}.json" \
        --output "results/q35_hybrid/gsm8k_vllm/reference_${arm}.json" \
        > "results/q35_hybrid/logs/gsm8k_vllm_reference_${arm}.log" 2>&1
done
```

After smoke/reference review, run each arm with phase `pilot`, omit `--smoke`
and `--enforce-eager`, and add `--limit 16`. Then run the full commands:

```bash
conda activate lowrankarena
export PYTHONPATH=.
run_arm 6 dense result
run_arm 2 uniform64 result --bank results/q35_hybrid/banks/c1_uniform_v64.pt
run_arm 0 twosided64 result --bank results/q35_hybrid/banks/c1_twosided_v64.pt
run_arm 5 twosided64_wo result \
    --bank results/q35_hybrid/banks/c1_twosided_v64.pt \
    --wo-bank results/q35_hybrid/wo_twosided_v64/wo_bank.pt
```

Completed execution required native hybrid-cache and M-RoPE interfaces,
spawned workers, and correcting lm-eval's two filter rows per question.
Shared-GPU memory fluctuations required moving Uniform V64 and using a fixed
6 GiB cache for compressed arms. Final physical GPUs were Dense=6,
Uniform64=5, Two-sided64=0, Two-sided64+Wo=2. Failures are preserved in appended
logs. No Git commit, upload or push has been performed for this extension.
