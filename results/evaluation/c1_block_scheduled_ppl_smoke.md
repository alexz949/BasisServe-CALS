# C1 block-scheduled teacher-forced NLL/PPL

Fixed corpus tokens are scored under a one-token sequential exact-C1 cache schedule and under direct exact block-KV commits. Shadow-Key drafting is not used.

Dataset: `wikitext2`; split: `test`; sequence length: `128`; samples: `4`.

| Block | Tokens | Sequential PPL | Block PPL | PPL ratio | Mean NLL delta | Top-1 agreement | Samples diverged |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 508 | 15.58325202 | 15.60956982 | 1.001688852 | 1.687427312e-03 | 0.97047244 | 4 |
| 4 | 508 | 15.58325202 | 15.58247115 | 0.999949891 | -5.011063204e-05 | 0.97637795 | 3 |
| 8 | 508 | 15.58325202 | 15.63191403 | 1.003122712 | 3.117846373e-03 | 0.98228346 | 3 |
| 16 | 508 | 15.58325202 | 15.62021775 | 1.002372145 | 2.369335507e-03 | 0.97637795 | 3 |

This is schedule-conditioned teacher-forced PPL. It isolates numerical execution-order effects and is not a free-running generation or MCQ result.
