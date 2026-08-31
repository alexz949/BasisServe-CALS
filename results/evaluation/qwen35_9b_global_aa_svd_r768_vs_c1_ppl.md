# Qwen3.5-9B communication-matched Global AA-SVD vs C1 PPL

## Setup

- Dense output width: 4096
- TP size: 8
- Global AA-SVD AllReduce rank: 768
- C1 Private AllGather local rank: 192
- Ring traffic per token per rank: 1344 FP16 elements for both methods
- Communication fraction relative to dense AllReduce: 18.75%
- Communication reduction: 81.25%
- Confirmation split: 16 sequence-length-512 windows starting at offset 336
- Environment: `basis`
- Checkpoint-build Slurm job: `8283512`
- PPL Slurm job: `8283515`

The Global AA-SVD runtime explicitly evaluates eight source-local partial
common-code projections, sums them to simulate TP8 AllReduce, and applies one
replicated decoder. C1 instead gathers eight private code packets and applies
a joint decoder. Neither method compresses the attention cache or GDN
recurrent state.

## All-layer PPL

| Method | PPL | Change vs Global AA-SVD | Relative to dense |
|:--|--:|--:|--:|
| Dense | 10.75096 | - | - |
| Global AA-SVD AllReduce R768 | 12.73274 | - | +18.434% |
| C1 decoder-only Private AllGather local-R192 | 12.15660 | -4.525% | +13.075% |
| C1 ALS10 Private AllGather local-R192 | 12.01602 | -5.629% | +11.767% |
| C1 Global-KL ragged, average local-R192 | 11.85263 | -6.912% | +10.247% |

Relative to Global AA-SVD, uniform C1 decoder-only has paired delta-NLL
`-0.046304 +/- 0.004651` and improves all `16/16` windows. Uniform C1 ALS10
has paired delta-NLL `-0.057936 +/- 0.004950` and also improves `16/16`
windows. The Global-KL ragged checkpoint improves `16/16` windows with paired
delta-NLL `-0.071627 +/- 0.008047`.

## Block isolation

| Installed blocks | Global AA-SVD | C1 decoder-only | C1 ALS10 |
|:--|--:|--:|--:|
| 24 GDN layers | 12.14434 | 11.66501 | 11.60379 |
| 8 full-attention layers | 11.03700 | 11.06590 | 11.01591 |

The communication-matched C1 advantage is concentrated in GDN. On full
attention alone, Global AA-SVD and private C1 are nearly tied; after ALS10,
C1 is lower by only 0.19%. On GDN alone, decoder-only C1 is lower by 3.95%
and ALS10 is lower by 4.45%.

## Artifacts

- Global AA-SVD checkpoint: `results/qwen35_9b_c1/global_aa_svd_allreduce_r768.pt`
- Global AA-SVD PPL: `results/evaluation/qwen35_9b_global_aa_svd_allreduce_r768_confirmation_ppl.json`
- C1 decoder-only PPL: `results/evaluation/qwen35_9b_c1_r192_activation_aware_confirmation_ppl.json`
- C1 ALS10 PPL: `results/evaluation/qwen35_9b_c1_r192_als10_confirmation_ppl.json`
- C1 Global-KL result: `results/qwen35_9b_c1/global_kl_r192_grid64/mean_dp/result.json`

This is a paired, disjoint confirmation experiment, but the 16-window set is
still smaller than a full validation-corpus PPL evaluation.
