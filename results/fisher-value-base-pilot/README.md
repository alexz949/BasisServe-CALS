# fisher-value-base-pilot

Fisher-trained Value Base for the B16R16 K router (evaluation/fit_k_routing_fisher_base.py), pilot on Qwen3-8B post-trained,
page 8, no sink, 32 fit + 16 held-out 128K windows (C4 / synthetic-retrieval 50-50 calibration, identity v96), layers 0, 13, 33.
Arms: R16-Fisher (B0R16), MSE-B16 + R16-Fisher (current production recipe), Fisher-B16 + R16-Fisher (sequential: Base fitted alone by
exact proximal block-coordinate page-Fisher steps, then residual refit), Joint-Fisher B16R16 (alternating Base / residual steps on the
combined score, 4 outer sweeps, final canonical residual polish), R32-Fisher (B0R32), Exact-K. `pilot_summary.txt` has the
fit / held-out Fisher NMSE (loss / exact-score page-Fisher energy) and the held-out routing diagnostics with the deployed
selector (2048 + recent 64): routed-page recall, page KL, retained exact attention mass. `diagnostics/` are the per-layer JSON,
`records/` the per-layer fit records (losses, protocol, hashes; tensors not included).

Mean over the three layers (held-out): MSE-B16+R16 0.156 / recall 0.711 / KL 0.404 / mass 0.869; Fisher-B16+R16 0.142 / 0.672 / 0.477 / 0.864;
Joint-Fisher B16R16 0.123 / 0.720 / 0.353 / 0.872; R32 0.126 / 0.718 / 0.337 / 0.872. The joint recipe ties R32 at half the sidecar
(wins layers 0 and 13, loses layer 33); the sequential recipe lowers the Fisher loss without improving the routing metrics.
