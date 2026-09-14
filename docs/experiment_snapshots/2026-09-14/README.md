# Experiment snapshot, 2026-09-14

Environment: `basis` (PyTorch 2.6.0+cu124). This snapshot accompanies the current calibration, routing, and evaluation source changes.

## Llama-3.1-8B-Instruct

Dense original V (head dimension 128), original Wo. B16R16 is fitted on 64 x 65536-token windows, with 16 independent diagnostic windows. Fit Q64/window, diagnostic Q32/window, Page32, 40 BCD sweeps and PCG100. Loki uses rank32 PCA from the same 64 fit windows. The 128K experiment reuses these 64K factors.

Completed 64K results are in `llama_instruct_ruler64k.json` and `llama_instruct_loki64k.json`. The 128K snapshot is partial; its means must not be treated as final audited results or compared across different sample subsets.

Current 128K prefill uses FlashAttention-2; decode remains method-specific, including our split indexed Triton kernel. All methods use the same Dense V. Budget semantics: ours hard2048 including sink32 and recent64; LRQK per-query-head2048 plus lite64; Loki per-query-head2048 with uncapped GQA union; ShadowKV routed2048 plus its local/outlier support.

The evaluation command is `python -u -m evaluation.eval_k_routing_ruler evaluate --dense-v --official-lrqk --chat-template --identity "$R/manifests/v128.json" --data "$R/ruler128k" --bank "$R/ours_b16r16" --output "$R/eval128k_fa2" --sequence-length 131072 --rope native --arm ARM --shard-index INDEX --num-shards 4`, where R is `/home/zhangal/BasisServe-CALS-runs/llama31_8b_instruct_64k`.

## Qwen3.5-9B LongBench v2

70 eligible inputs in the 32K-64K protocol, allocated V192, original Wo; two-stage CoT with reasoning and answer caps1024, greedy decoding. Full attention uses the same compressed V192. The LRQK adapter completed69/70; its one failed sample prevents reporting completed-only accuracy as full-set accuracy. This is the older adapted LRQK path, distinct from the official Instruct experiment.

## Device transfer diagnostic

Direct GPU-to-GPU copies on the tested gpu-a100-04 allocation corrupted values, including finite-valued tensors. Host-mediated copies passed bidirectional exact checks; model smoke passed for ours, full and ShadowKV. Official LRQK host-mediated dual40GB smoke hit OOM. This is not proof of a particular hardware or driver cause. Single80GB formal evaluation is unaffected. Host-mediated execution is separately recorded and should not represent native multi-GPU throughput.

## External dependencies

Third-party checkouts and their datasets are not vendored in this upload. Clone the following sources at the recorded revisions:

- LRQK: `https://github.com/tenghuilee/LRQK.git`, revision `caf16293db2e4423a84ab2e895bacf64479f1eb7`.
- LongBench: `https://github.com/THUDM/LongBench.git`, revision `2e00731f8d0bff23dc4325161044d0ed8af94c1e`.
- ShadowKV: `https://github.com/ByteDance-Seed/ShadowKV.git`, revision `e51904cdeab7d4d34013370f09f2cf5fcd655e15`.

Model weights, raw activations, fitted binary factors, benchmark inputs, and live logs remain in their existing storage locations. Existing result JSONs are copied verbatim; absolute artifact paths inside them describe the originating machine. See provenance.json for source hashes.
