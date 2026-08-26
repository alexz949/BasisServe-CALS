# Qwen3.5-9B C1 decoder-only initialization vs ALS PPL

## Setup

- Model: Qwen3.5-9B
- TP simulation: 8 source-private encoders per output projection
- Local rank: 192 for every layer
- Sequence length: 512
- Confirmation split: 16 windows starting at offset 336
- Factor calibration: 256 fit windows and 64 disjoint held-out windows
- Factor dtype and model dtype: FP16
- Environment: `basis`
- Factor build Slurm job: `8283039`
- PPL Slurm job: `8283040`

The C1 decoder-only baseline uses source-local activation-weighted SVD
encoders followed by an optimal joint decoder. It is a Private-AllGather C1
initialization, not standard global SVD-LLM AllReduce. ALS10 starts from the
same C1 factors and permits up to ten decoder-closed encoder/redecoder sweeps,
selecting on the same factor-level held-out moments. Both PPL variants use
identical ranks, communication volume, confirmation windows, and dense weights
outside the output-projection intervention.

## Results

| Variant | Activation-aware PPL | ALS10 PPL | ALS PPL change | Paired delta-NLL +/- SE | ALS improved windows |
|:--|--:|--:|--:|--:|--:|
| GDN only | 11.66501 | 11.60379 | -0.525% | -0.005262 +/- 0.001328 | 14/16 |
| Full attention only | 11.06590 | 11.01591 | -0.452% | -0.004528 +/- 0.000806 | 15/16 |
| All attention layers | 12.15660 | 12.01602 | -1.156% | -0.011632 +/- 0.001979 | 14/16 |

Dense PPL is `10.75096`. The combined activation-aware checkpoint increases
PPL by `13.075%` relative to dense, while ALS10 increases it by `11.767%`.
Thus ALS10 recovers `10.00%` of the activation-aware compression PPL gap at
unchanged communication cost.

For context, adding the independently selected Global-KL ragged schedule at
the same average local rank 192 reaches PPL `11.85263`. Relative to uniform
activation-aware factors, ALS plus layerwise allocation improves PPL by
`2.501%` and recovers `21.63%` of the compression gap.

## Artifacts

- Activation-aware PPL: `results/evaluation/qwen35_9b_c1_r192_activation_aware_confirmation_ppl.json`
- ALS10 PPL: `results/evaluation/qwen35_9b_c1_r192_als10_confirmation_ppl.json`
- Activation-aware GDN factors: `results/qwen35_9b_c1/activation_aware_r192/gdn_private_ag_decoder_only_all.pt`
- Activation-aware full-attention factors: `results/qwen35_9b_c1/activation_aware_r192/full_private_ag_decoder_only_all.pt`

The 16-window comparison is paired and disjoint from factor fitting, but it is
still a small confirmation set rather than a full validation-corpus PPL run.
