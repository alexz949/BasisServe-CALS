# Three hard-prompt subspace diagnostic

Samples160(qmsum),128(gov_report),161(qmsum) are selected by the lowest identity-router mean non-sink mass recall over layers15/19/24/33 in the previous12-prompt core diagnostic. Selection values are85.7162%,86.4183%,88.6413%. They are explicitly selected hard cases, not an unbiased benchmark sample. No reference answers enter selection or fitting.

The model uses frozen full-K C1-V96 FP16 V100 memory-efficient SDPA prefill. Base16 remains fixed. Post-RoPE residual and routing diagnostics are FP32. Each prompt supplies256 fit Q and32 disjoint diagnostic Q, uniformly interleaved over the latter region of the prompt; every teacher distribution uses its own causal prefix. This is a within-completed-prompt diagnostic, not a generated-query evaluation.

All selected layers and all8 GQA groups compare:

1. Offline E8/U8.
2. Prompt-U128x8 with offline E8 fixed.
3. Prompt-specific E8/U8 reference.
4. Exact-QK selection with the same Page32/shared B2048/pinned-page0 rule.

The fit objective is the existing exact-teacher non-sink Page-Fisher quadratic. Query and encoder subproblems reuse the existing PCG solver with relative damping1e-5, relative residual tolerance1e-5 and maximum200 iterations. This is the existing damped solver, not a proximal penalty toward offline U. Joint fitting uses10 BCD sweeps from offline factors and residual-PCA initialization. Selection uses fit loss only and includes offline/prompt-U incumbents to prevent reporting a worse fitted candidate as the best reference. All sweep losses and solver diagnostics are saved. Neither BCD nor the Fisher objective establishes a global rank8 or mass-recall ceiling.

Outputs include per-arm fit loss, diagnostic Fisher loss, teacher mass recall, non-sink mass recall and page overlap. Every diagnostic Q/group also saves all page priorities, tied rank intervals and selected masks, plus teacher per-head page mass. The JSON highlights up to5 highest-average-teacher-mass pages in the exact-selected minus offline-selected set for each condition. These are high-mass missed pages, not annotated answer-evidence pages.

Code: `evaluation/diagnose_c1_prompt_subspace.py`. Outputs: `results/evaluation/c1_prompt_subspace`. Query split and tied-page-rank CPU checks passed. Smoke8301234 (sample160/layer15/group0) passed. Formal array8301235 uses three independent V100 workers, one complete prompt each, followed by summary8301236. At this snapshot formal jobs were running; no complete three-prompt result was available.

Smoke mass recall: offline85.8016%, prompt-U85.6426%, prompt-E/U85.9564%, exact-QK86.6218%. These single-group values are not the final result or LongBench generation scores. Original checkpoints and completed benchmark artifacts are unchanged.
