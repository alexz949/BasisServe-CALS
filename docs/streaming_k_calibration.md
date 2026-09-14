# Llama C1 / Base16 / Residual16 streaming calibration

`evaluation/fit_k_routing_streaming.py` replaces raw-activation capture for this workflow with two dense teacher replays. It processes one token window at a time, using `model.model` without a KV cache or vocabulary logits. It never writes raw V/K, full Q, or an `[all_windows, sequence_length, hidden_size]` tensor.

The existing completed raw captures and their fitting entry point remain historical experiment artifacts. They are not inputs to the new pipeline; only frozen token IDs and the model / C1 identity are required.

## Stages

1. `moments`: dense replay accumulates normalized attention `o_proj` input covariance, raw V / pre-RoPE K first and second moments, and fit-only candidate queries. Query-Gram selection uses only fit windows; candidate queries are discarded after selection.
2. `base`: transform raw moments through the chosen fixed value encoder and fit affine rank-16 RRR. The encoder can be fitted after the moments replay. `Gkk` also allows held-out Base MSE to be evaluated without raw activations.
3. `fisher`: second dense replay with the encoder and Base fixed. Build each window's Page-Fisher statistics, then save FP32 symmetric upper triangles and selected queries. No token-level K/V is persisted.
4. `fit`: load statistics for one layer, restore the original query-major / window-minor ordering, and run existing B16R16 BCD. Export factor tensors with the existing tensor names. The new JSON metadata records the streaming provenance.

`all` runs these four stages using the encoder already identified by `--identity`. It does **not** refit C1 automatically. To refit C1, first collect `moments` for all 32 layers and run `assemble-covariance`. The resulting `covariance/manifest.json` uses `basisserve.attention_o_proj_covariances.v1`, accepted by the existing Llama C1 fitter. Fit / allocate C1 and create its authenticated identity, then run `base`, `fisher`, and `fit` with that identity. The moment protocol is independent of the encoder identity; Base and Fisher artifacts bind the chosen encoder identity.

## Frozen numerical protocol

- 64 fit windows, IDs 0–63; 16 diagnostic windows, IDs 64–79.
- Fit Q64, diagnostic Q32, both chosen using fit-only stratified Query-Gram, four bins.
- Base16 affine pre-RoPE K regression; Residual16 Page32 Fisher with one excluded prefix page.
- 40 BCD sweeps, PCG max 100, damping / tolerance 1e-5; TF32 disabled.
- Native dense BF16 Llama teacher; original Wo.
- Base raw moments and their encoder transform use FP64. This is algebraically equivalent to projected moments, but floating-point accumulation differs from the historical FP32-products implementation. Bitwise factor or prediction identity is not claimed.
- Fisher Grams are symmetrized before upper-triangle packing. This removes floating-point antisymmetry, not information from the symmetric objective.
- Covariance uses FP32 chunked products, normalized by the actual number of tokens in each split.

Changing sequence length does not change query count, residual rank, or BCD settings. Non-smoke runs require full-length windows and the frozen counts / solver settings. Files tagged `test_only` cannot be used for a formal run.

## Invocation

Use the `basis` environment and Slurm. Set `--layers` to the comma-separated layers assigned to that worker. There is no hidden activation cache shared between workers; workers replay the dense model independently.

```bash
python -u -m evaluation.fit_k_routing_streaming all \
  --identity PATH_TO_AUTHENTICATED_C1_IDENTITY \
  --windows PATH_TO_80_BY_65536_WINDOWS \
  --output PATH_TO_STREAMING_OUTPUT \
  --layers 0,1,2,3 --sequence-length 65536
```

The adjacent token manifest must authenticate the tokenizer/model configuration, token-file hash and fit/validation IDs. Existing 32K windows are not silently padded or extended. Preparing genuine 64K calibration / reasoning rollout windows is a separate operation.

## Storage and memory

The output contains `moments/`, `covariance/`, `base/`, per-window packed `fisher/`, and final `ours_b16r16/` factors. Existing files are hash-verified; no artifacts are automatically deleted.

Fisher storage depends on window/query/head counts and head dimension, not token length. At Llama geometry, Q64 fit / Q32 diagnostic requires about 145.1 GiB of packed Grams across 32 layers, plus about 2.25 GiB of selected FP32 queries. Processing one layer through all stages limits retained temporary Fisher storage to that layer if the user later removes completed layer statistics. No deletion policy is implicit in the command.

Transient model activations still scale with one context length. Fit-only candidate Q buffers also scale with the 64-token candidate stride and number of layers processed together; they are held on CPU, not written to disk. `--layers` controls simultaneous statistics / candidate memory. BCD unpacks one layer's Grams into CPU and GPU memory. This implementation avoids the approximately 980 GiB raw cache; it does not claim zero temporary memory or zero Fisher storage.

## Validation

CPU tests compare transformed raw moments with directly projected affine regression and held-out MSE, and verify Fisher packing and original sample ordering. A real Llama layer-0 smoke with two fit / one held-out 4K windows completes both replays and BCD. A separate synthetic 64K smoke repeats each source window solely to test sequence-length handling; it is not a calibration corpus or a quality result.

Validation runs in `basis` on one L40S:

- Slurm 8315456: layer 0, two fit / one held-out 4096-token windows, two BCD sweeps / PCG 4; completed in 33 seconds, about 330.8 MiB total artifacts.
- Slurm 8315470: layer 0, one fit / one held-out synthetic 65536-token windows, two BCD sweeps / PCG 4; completed in 66 seconds, about 265.3 MiB total artifacts. Every artifact hash was checked and no raw activation tensor was present. The existing C1 covariance loader accepted the output.
- `python -m pytest -q tests/test_streaming_k_statistics.py`: 2 passed.

The short PCG cap intentionally does not establish convergence or calibration quality. A full genuine 64K corpus / all-layer fit has not been launched. No existing calibration artifacts were deleted.
