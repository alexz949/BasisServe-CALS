# Exact-proposal C1 block-commit numerical oracle

This isolates block-versus-sequential BF16 execution. Candidate tokens come from the transactional one-token exact-Key C1 path used by strict replay; Shadow-Key is not used.

| Block | Tokens | Top-1 agreement | Prompts diverged | Mean NLL delta | PPL ratio | Mean KL |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 256 | 0.98437500 | 3 | 3.208441803e-03 | 1.003213594 | 1.139728206e-03 |
| 4 | 256 | 0.97656250 | 3 | 1.935802190e-05 | 1.000019358 | 1.115906248e-03 |
| 8 | 256 | 0.98046875 | 3 | 4.599747066e-03 | 1.004610342 | 9.570515707e-04 |
| 16 | 256 | 0.97265625 | 3 | 3.191831879e-03 | 1.003196931 | 1.148548826e-03 |

A top-1 disagreement is an implementation-schedule difference, not by itself a task failure. This report contains no Shadow-Key proposal error and no production throughput measurement.
