# Qwen3-8B C1-V96 32K RULER comparison

## Protocol

- Model: Qwen3-8B-Base.
- Evaluation: 11 RULER-v1 tasks, 8 fixed samples per task, 32K context, greedy decoding.
- V96 checkpoint: 256 x 2048-token C4 fit windows and 64 x 2048-token validation windows.
- Routing: independent per-layer activation-aware KQ-SVD R32, page size 64, exact-token budget 1024 per query head.
- `dense K + C1-V96` isolates the value payload. `R32/B1024 + C1-V96` adds the routing error on top.

## Main result

| Arm | Task-balanced accuracy | Delta vs previous arm |
|:---|---:|---:|
| BF16 dense K/V | 86.48% | -- |
| Dense exact K + C1-V96 | 83.60% | -2.88 pp |
| R32/B1024 exact-K routing + C1-V96 | 81.04% | -2.56 pp |

The complete routed system is 5.44 pp below BF16. The loss is approximately split between the V96 payload (2.88 pp) and the B1024 selector (another 2.56 pp).

| Task | BF16 | Exact K + V96 | R32/B1024 + V96 |
|:---|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 87.50% | 87.50% | 87.50% |
| niah_multikey_2 | 100.00% | 75.00% | 62.50% |
| niah_multiquery | 93.75% | 87.50% | 90.62% |
| niah_multivalue | 100.00% | 93.75% | 100.00% |
| vt | 95.00% | 92.50% | 92.50% |
| fwe | 87.50% | 83.33% | 70.83% |
| qa_1 | 50.00% | 62.50% | 50.00% |
| qa_2 | 37.50% | 37.50% | 37.50% |

The clearest routing-specific regressions are `fwe` and `niah_multikey_2`, each losing 12.50 pp relative to exact-K C1. Improvements on `niah_multiquery` and `niah_multivalue`, and the V96 improvement on `qa_1`, are individual decoded-output flips on only eight samples and should not be interpreted as a monotonic quality gain from approximation.

## V-rank and calibration comparison

| C1 payload | C1 fit | K-routing fit | Exact-K C1 | Routed C1 | Persistent GPU KV scalar ratio |
|:---|:---|:---|---:|---:|---:|
| V64 | 256 x 2K | 8 x 32K | 66.12% | 66.17% | 37.5% |
| V64 | 128 x 4K | 8 x 32K | 76.23% | 73.71% | 37.5% |
| V64 | 16 x 32K | 8 x 32K | 81.84% | 77.69% | 37.5% |
| V64 | 32 x 32K | 32 x 32K | 79.00% | 79.11% | 37.5% |
| **V96** | **256 x 2K** | **32 x 32K** | **83.60%** | **81.04%** | **50.0%** |

The clean same-calibration payload comparison is V64 versus V96 at 256 x 2K: both use the same calibration snapshot and objective. Raising the per-head value rank from 64 to 96 improves exact-K RULER by 17.48 pp (`66.12% -> 83.60%`) and reduces the checkpoint's 2K held-out relative output MSE from 13.70% to 5.54%. This is strong evidence that V capacity was at least as important as calibration length in the earlier V64 failures.

The routed V64-2K versus V96-2K comparison is not perfectly controlled because the available V64 run used the older 8-window K-routing checkpoint, while V96 uses the 32-window checkpoint. The exact-K rows do not have this confound.

## Storage and logical traffic

- Persistent resident representation: V96 payload plus R32 routing metadata = 50.0% of dense BF16 KV scalar count.
- Mean physically selected exact-K token fraction: 5.7104%.
- Logical exact-K traffic in the oracle: 131.571 MiB per decode token.
- Peak allocated memory: about 32.27 GiB per A100 process.
- Exact K remained GPU-resident in this correctness oracle; the traffic figure is logical and does not measure PCIe latency or page-cache behavior.

## Run notes

- Sample evaluation job: `8288151`, 2 x NVIDIA A100-PCIE-40GB, 44 samples per rank, about 46 minutes of useful evaluation time.
- Both complete rank shards were written before the first-use NCCL final barrier hung on this A100 node.
- The evaluator's control-plane process group was changed to Gloo because ranks perform independent GPU work and only require a final CPU synchronization.
- Resume/merge job: `8288166`, completed in 14 seconds with `pending=0` on both ranks.
- Final status: `complete`, 88 records, no NaN or OOM.

## Conclusion

V96 substantially repairs the C1 payload ceiling even though it was calibrated only at 2K: exact-K V96 reaches 83.60%, only 2.88 pp below BF16 and above every tested V64 checkpoint. The remaining end-to-end gap is no longer predominantly a generic V-capacity failure. At B1024, routing contributes a comparable additional 2.56 pp loss, concentrated in `fwe` and `niah_multikey_2`.

The next controlled experiment should therefore target the selector (for example a B1024/B2048 budget comparison in one shared prefill run) before paying the cost of fitting a 32K V96 checkpoint.
