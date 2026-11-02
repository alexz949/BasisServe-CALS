# Qwen3-8B-Base V64 rank allocation (Table 1, 50% budget) with the BasisKV release code

Environment: conda `lowrank` (torch 2.6 cu124) with transformers 5.17.0 from `/zpool-00/home/lz299/.cache/basiskv_env/pylib517`;
RTX A5000 24 GB, Slurm partitions athena / athena-mini (node2 GPUs 4-7 are unhealthy and were excluded).

Pipeline (run from `BasisKV/`, model = Qwen/Qwen3-8B-Base snapshot 49e3418):

    python -m calibration.windows --model M --output windows --sequence-length 2048 --fit-windows 256 --heldout-windows 64
    python -m calibration.windows --model M --output profile --sequence-length 2048 --fit-windows 128 --heldout-windows 0 --seed 20260922
    python -m calibration.value --model M --windows windows/windows.safetensors --output snap                      # 35 min, 1 GPU
    python -m core.fit_value fit-shard --snapshot-dir snap --output-dir bank{R} --rank {R} --covariance-damping 1e-5  # R in 32 48 64 80 96 112 128, ~3 min each
    python -m core.fit_value merge --output-dir bank{R}
    python -m calibration.probe_kl --model M --windows profile/windows.safetensors --banks 32=bank32 64=bank64 96=bank96 \
        --anchor-rank 64 --low-rank 32 --high-rank 96 --output probes --layer-shard-index i --layer-shard-count 6      # ~15 min per shard
    python -m calibration.probe_kl ... --layer-shard-count 6 --merge
    python -m core.allocate --banks 32=bank32 48=bank48 64=bank64 80=bank80 96=bank96 112=bank112 128=bank128 \
        --probes probes/probes.json --target-average-rank 64 --output Q3-8B-Base-V64

Result (`Q3-8B-Base-V64/results.json`, mean rank 64.0, predicted KL change -0.0150):

    ours  [96, 64, 32, 48, 48, 48, 80, 64, 80, 64, 80, 64, 48, 48, 64, 64, 64, 64, 64, 80, 64, 64, 96, 128, 96, 48, 64, 48, 32, 64, 64, 48, 48, 80, 64, 32]
    paper [80, 64, 32, 48, 48, 32, 64, 64, 80, 80, 80, 64, 48, 48, 64, 64, 64, 64, 64, 80, 64, 64, 96, 112, 96, 48, 64, 64, 32, 64, 64, 48, 48, 96, 80, 32]

28 of 36 layers agree; the eight others (0, 5, 6, 9, 23, 27, 33, 34) differ by one 16-rank step. Uniform-bank held-out
relative MSE matches the published banks (rank 32: 0.2657 vs 0.2668; 64: 0.1366 vs 0.1368; 96: 0.0555 vs 0.0552).
Anchor KL of uniform V64 against the dense teacher: 0.1195 nats (mean full-vocabulary KL over 128 x 2047 positions).

Protocol differences from the paper run: freshly sampled 2048-token windows (the paper cut 4096-token windows), profile
windows from a second stream seed instead of the continuation of the same stream, and the KL probe over all positions in
FP32 instead of sampled positions. Evaluation (eval.ppl, eval.mcq) was not run.

## Perplexity (eval.ppl, batch 1, FP32 cross entropy in 512-position chunks)

    python -m calibration.windows --model M --output c4_validation_s20260824 --split validation --fit-windows 128 --heldout-windows 0 --seed 20260824
    python -m eval.ppl --model M [--value-factors BANK] --c4-windows c4_validation_s20260824/windows.safetensors --output OUT --batch-size 1

| bank                                   | WikiText-2 | C4 (128 validation windows, seed 20260824) |
|----------------------------------------|-----------:|-------------------------------------------:|
| dense                                  |      7.003 |                                     10.371 |
| V64, this run's allocation             |      8.413 |                                     11.339 |
| V64, paper schedule on this run's banks|      8.369 |                                     11.352 |
| paper (Table 1)                        |  7.00 / 8.26 |                               9.17 / 10.06 |

WikiText-2 dense matches the paper exactly. The V64 gap of 0.15 splits into about 0.04 from the eight differently
allocated layers (paper schedule on the same banks: 8.369) and about 0.11 from the factors themselves, which were
fitted on freshly sampled calibration windows. The C4 column uses different documents from the paper's: the paper's
sampler was the same algorithm with seed 20260824, but under datasets 5.0.0; with datasets 4.4.2 the streaming
shuffle yields other documents (seed 20260921 gave dense 11.048 / V64 12.118). The relative V64 degradation on C4 is
9.3% here against 9.7% in the paper. The paper's validation windows are not in the HF release.

## Zero-shot MCQ (eval.mcq: lm-eval 0.4.11, batch 8, max_length 4096, acc_norm where present)

    python -m eval.mcq --model M [--value-factors Q3-8B-Base-V64] --output OUT

| arm         | BoolQ | ARC-E | ARC-C | WinoG | PIQA  | HellaS | OBQA  | Avg   |
|-------------|------:|------:|------:|------:|------:|-------:|------:|------:|
| dense       | 0.830 | 0.799 | 0.566 | 0.720 | 0.793 | 0.786  | 0.420 | 0.702 |
| dense paper | 0.831 | 0.802 | 0.569 | 0.728 | 0.794 | 0.786  | 0.420 | 0.704 |
| V64         | 0.784 | 0.761 | 0.515 | 0.715 | 0.786 | 0.735  | 0.396 | 0.670 |
| V64 paper   | 0.804 | 0.771 | 0.519 | 0.719 | 0.788 | 0.735  | 0.398 | 0.676 |

The dense model is within 0.008 of the paper on every task (transformers 5.17 on RTX A5000 here, 5.15.1 on L40S in the
paper). V64 averages 0.006 below the paper; BoolQ accounts for most of it (-0.020), the other six tasks are within 0.010.
Each arm took about 21 minutes on one A5000.

## The paper's published factors under the release code

The published uniform banks (ICLR-results/qwen3-8b/c1/factor-banks/R*-S6) and KL probes (two-sided-kl/profile-s6-a1)
were converted to the release formats (`paper/`, script `paper_hf/convert.py`). `core.allocate` on them reproduces
the published V64 schedule in all 36 layers. The published V64 selected factors (`paper/Q3-8B-C1-R64-published`) under
`eval.ppl`:

| bank                                           | WikiText-2 | C4 (seed 20260824 windows) |
|------------------------------------------------|-----------:|---------------------------:|
| dense                                          |      7.003 |                     10.371 |
| V64 published paper factors                    |      8.237 |                     11.372 |
| V64 paper schedule on this run's banks         |      8.369 |                     11.352 |
| V64 this run (own banks, own allocation)       |      8.413 |                     11.339 |
| paper Table 1 (dense / V64)                    | 7.00 / 8.26 |               9.17 / 10.06 |

With the paper's own factors the release evaluator gives 8.237 against the published 8.260, so the evaluation path
reproduces the paper to within 0.3%. The remaining WikiText-2 gap of this run is the calibration data: the paper's
Qwen3-8B calibration windows and covariance snapshots are not published, and the window bank could not be regenerated
(the fingerprint documents of profile windows 320-463 did not match for seeds 20260821/20260824/20260921 under
datasets 4.4.2, nor 20260821/20260824 under datasets 5.0.0; the two datasets versions also order the same seed
differently).

## The paper's exact windows, regenerated from the records in git

The window builder reproduces the paper's document streams under datasets 4.4.2 (5.0.0 orders the same seed
differently). Seeds recovered from the committed records:

| bank                               | use                         | arguments                               | check                                         |
|------------------------------------|-----------------------------|-----------------------------------------|-----------------------------------------------|
| `c4_validation_s20260828`          | C4 PPL                      | `--split validation --seed 20260828`    | windows.safetensors sha256 equals the paper's |
| `windows_s4096_s20260828` -> `profile_paper` | KL probes (rows 320..447, first 2048 tokens) | `--sequence-length 4096 --fit-windows 512 --heldout-windows 128 --seed 20260828` | all 144 profile document urls match in order |
| `windows_s20260821`                | ALS calibration (256 + 64)  | `--sequence-length 2048 --seed 20260821` | verified by refitting (below)                 |

On the paper's C4 windows the release evaluator gives dense 9.1686 (paper 9.1686; per-document NLL within 3e-7) and
the published V64 factors 10.0592 (paper 10.0595).

## Full reproduction on the regenerated windows (release code end to end)

    python -m calibration.value --windows windows_s20260821/windows.safetensors --output snap_s20260821
    python -m core.fit_value fit-shard --snapshot-dir snap_s20260821 --output-dir repro_bank{R} --rank {R} --covariance-damping 1e-5   # R = 32 ... 128
    python -m calibration.probe_kl --windows profile_paper/windows.safetensors --banks 32=repro_bank32 64=repro_bank64 96=repro_bank96 ...
    python -m core.allocate --banks 32=repro_bank32 ... 128=repro_bank128 --probes repro_probes/probes.json --target-average-rank 64 --output repro_V64

- Uniform banks: mean held-out relative MSE equals the paper's to six digits at every rank (e.g. R64 0.136810 vs 0.136810);
  per-head V/O products E D are within 0.2% (mean) of the published R64 bank.
- Probes: anchor KL 0.12315 vs the paper's 0.12148. The paper sampled 1024 of the 2047 prediction positions per window
  (seed 20260903); `probe_kl` uses all positions. With the paper's probes the reproduced banks give the published V64
  schedule in all 36 layers; with the release probes two layers swap (layer 9: 80 -> 64, layer 32: 48 -> 64).
- Evaluation of `repro_V64`:

| metric      | repro V64 | paper V64 |
|-------------|----------:|----------:|
| WikiText-2  |    8.3144 |    8.2600 |
| C4          |   10.0643 |   10.0595 |
| BoolQ       |     0.805 |     0.804 |
| ARC-E       |     0.773 |     0.771 |
| ARC-C       |     0.506 |     0.519 |
| WinoGrande  |     0.718 |     0.719 |
| PIQA        |     0.782 |     0.788 |
| HellaSwag   |     0.733 |     0.735 |
| OBQA        |     0.398 |     0.398 |
| MCQ average |     0.674 |     0.676 |
