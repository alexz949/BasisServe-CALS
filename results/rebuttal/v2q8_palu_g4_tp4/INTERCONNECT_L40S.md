# L40S node 1: TP4 grid paused on a PCIe bottleneck

2026-11-10, Vast.ai 4x L40S instance. Repository at `84789fd` plus the uncommitted changes listed under
"Deviations". This note explains why the formal grid from `RUN_L40S.md` was paused after 18 cells and
how to check a replacement node before running it again.

## Summary

- Validation and smoke passed. The formal grid then ran with NCCL P2P enabled, and was paused after 18
  of its 42 eager cells, because this node's interconnect turned out to dominate the measurement.
- All four GPUs sit behind one PCIe switch whose uplink to the CPU is **Gen4 x8**; the CPU root port
  itself is only x8 wide. All GPU-to-GPU traffic crosses that one link. Peer-to-peer copies run at
  12.6 GB/s on every pair. Aggregate host-to-device bandwidth stays at 13 GB/s whether one GPU copies or
  all four do.
- A 4-GPU NCCL all-reduce of 256 KiB, the size of one decode all-reduce at batch 32, takes **292 µs**.
  A decode step issues about 64 of them, so roughly **19 ms of the 36 ms dense step at batch 32 is
  communication**.
- Part of this is fixable in software. About 120 µs of the 292 µs comes from NCCL choosing its LL
  protocol for small messages. With `NCCL_PROTO=Simple` the same all-reduce takes 171 µs, and the
  dense batch-32 decode step drops from 35.96 ms to **28.53 ms**. Larger messages (batch 64 and 128,
  prefill) do not improve. See "Follow-up".
- The GPU-to-GPU limit is most likely host configuration (ACS redirection), not a hardware ceiling.
  Only the host operator can change it.
- GPU 0 negotiated x8 instead of x16, but this has no measurable effect: every GPU pair measures the
  same, with or without GPU 0.
- Consequences for the grid:
  - The arms rank in the same order as on the A5000.
  - The gaps between them are compressed by the shared communication cost.
  - The PaLU-B (sharded) penalty is inflated, because its extra cost is almost entirely communication.
  - TTFT reflects the link more than the model.
- **Recommendation:**
  - Rerun on a node that passes the interconnect check in the appendix, or on this node after the
    host disables ACS redirection and the check passes.
  - Run the same check on the A5000 node, so we know whether the two runs are comparable at all.
  - Results from this node as it stands are valid for this interconnect only. Do not present them as
    representative of 4x L40S.

## Machine

| | |
|---|---|
| GPUs | 4x NVIDIA L40S (46 GB), driver 595.91.07, 0 uncorrected ECC errors |
| CPU | 2x AMD EPYC 7763; all four GPUs on NUMA node 0 |
| Software | vLLM 0.30.0, torch 2.13.0+cu129, Triton 3.7.1, Transformers 5.22.2 (reference: 5.17.0), NCCL 2.29.7 |
| Environment | Unprivileged container, no Slurm, `CUDA_VISIBLE_DEVICES=0,1,2,3` |

## What raised the question

- Prefill ran at about 4,300 tokens/s. The dense batch-32 request cell had a TTFT p50 of 23.4 s.
- The GPUs reported 100 % utilization while drawing only 140–158 W of a 350 W limit. That pattern
  means they were mostly waiting, not computing.
- Decode at batch 32 was only 3–7 % faster than on the A5000.
- `nvidia-smi` reported GPU 0 at PCIe x8 and the other three GPUs at x16.

## Measurements

The raw logs are `logs/pcie_nccl_debug_l40s.log` and `logs/interconnect_l40s_node1.log` (the second
was produced by the appendix script).

**PCIe chain** (sysfs). Every GPU hangs off switch `22:xx` → switch upstream port `21:00.0` → CPU root
port `20:03.1`.

| Link | Width (current/max) | Speed |
|---|---|---|
| GPU 0 `23:00.0` | **8/16** | Gen4 |
| GPU 1–3 `24/25/26:00.0` | 16/16 | Gen4 |
| Switch upstream port `21:00.0` | **8/16** | Gen4 |
| CPU root port `20:03.1` | **8/8** | Gen4 |

**Copy bandwidth** (256 MiB, pinned memory).

| Test | Result |
|---|---|
| Host→device, one GPU at a time | 12.8–13.0 GB/s on every GPU, GPU 0 included |
| Host→device, 1 / 2 / 4 GPUs at once (aggregate) | 12.8 / 13.0 / 13.1 GB/s, i.e. flat |
| Peer-to-peer copy, all 12 directions | 12.59 GB/s on every pair |
| Peer-to-peer copy, 256 KiB | 39–42 µs on every pair |

**NCCL all-reduce** (BF16, `torch.distributed`, NCCL 2.29.7). With P2P enabled, NCCL reports
`via P2P/CUMEM` on every channel, so it does not fall back to shared memory.

| Group | P2P | 256 KiB | 512 KiB | 1 MiB | 64 MiB | Bus bandwidth at 64 MiB |
|---|---|---|---|---|---|---|
| 4 GPUs | on | 292 µs | 315 µs | 566 µs | 37.1 ms | 2.7 GB/s |
| 4 GPUs | off | 330 µs | 541 µs | 1041 µs | 65.2 ms | 1.5 GB/s |
| any pair (6 pairs, with or without GPU 0) | on | 100 µs | 116–117 µs | 208–210 µs | 12.3–12.4 ms | 5.4–5.5 GB/s |
| any pair | off | 119–144 µs | 209–247 µs | 410–444 µs | 24.3–24.5 ms | 2.7–2.8 GB/s |

## Interpretation

- **One shared x8 uplink.** Aggregate host-to-device bandwidth does not grow with the number of GPUs.
  The uplink therefore carries all host traffic for all four GPUs, at about 13 GB/s.
- **P2P traffic also goes up through the CPU.** Three observations point that way:
  - Peer-to-peer bandwidth equals the uplink bandwidth.
  - Every pair measures identically.
  - Four-GPU bus bandwidth is exactly half the two-GPU figure, which keeps the aggregate near 11 GB/s.

  Together they mean GPU-to-GPU traffic is routed up to the root complex instead of being turned around
  inside the switch. This is common when ACS/IOMMU redirection is active under virtualization. That is
  an inference: the ACS settings cannot be inspected from inside this container.
- **GPU 0's x8 link is not the bottleneck.** The shared uplink is x8 as well, so GPU 0 is no slower
  than the others. Every measurement above confirms this.

## Follow-up: what software can and cannot fix

Raw output: `logs/nccl_tests_l40s_node1.log` (nccl-tests 2.21.1 `all_reduce_perf -g 4`, plus the
multi-process torch test, which uses the same `P2P/CUMEM` path as vLLM).

**Two separate limits.** The first version of this note treated them as one.

| Limit | Cause | Fixable from the container |
|---|---|---|
| Host↔GPU aggregate of 13 GB/s | Gen4 x8 uplink | No (hardware) |
| GPU↔GPU capped at 12.6 GB/s, shared by all pairs | P2P routed through the root port, most likely ACS redirection | No, but the host operator can |
| 292 µs for a 256 KiB all-reduce | NCCL picks the LL protocol for small messages | **Yes** |

**ACS could not be read.** The container lacks `CAP_SYS_ADMIN`, so only the first 64 bytes of PCI
configuration space are readable and `lspci -vvv` shows `Capabilities: <access denied>`. The evidence
for redirection is indirect:

- Each GPU and each switch downstream port sits in its own IOMMU group (GPUs: 53–56; ports: 47–50).
  Linux groups devices this way when ACS isolation is enabled on the downstream ports.
- The AMD IOMMU is on (`iommu=pt`).
- Peer-to-peer bandwidth equals the uplink bandwidth.

The host is bare metal (no hypervisor flag; DMI reports a Tencent Star Lake server) and we run in a
Docker container on it. The host operator can confirm with
`setpci -s 22:0N.0 ECAP_ACS+6.w` for N = 0..3; bits `0x04` and `0x08` are the two redirect controls.

**NCCL environment.** No `NCCL_*` variable was set anywhere. NCCL 2.29.7 detects the topology
correctly (GPUs `PIX`, uplink 12 GB/s) and uses P2P.

**Algorithm and protocol** (4-GPU all-reduce, multi-process, P2P on, µs per call):

| Setting | 256 KiB (b32) | 512 KiB (b64) | 1 MiB (b128) | 64 MiB (prefill) |
|---|---|---|---|---|
| default | 291 | 317 | 567 | 37,177 |
| `NCCL_PROTO=Simple` | **171** | 314 | 563 | 37,101 |
| `NCCL_PROTO=LL128` | 165 | 331 | 621 | 39,002 |
| `NCCL_ALGO=Tree` | 180 | 324 | 610 | 36,104 |
| `NCCL_ALGO=Tree NCCL_PROTO=LL128` | **156** | 297 | 585 | 38,208 |

- Forcing `NCCL_PROTO=LL` reproduces the default at 256 KiB (294 µs). On this fabric LL reaches about
  half the bandwidth of the other protocols (1.4 vs 2.7 GB/s).
- `NCCL_PROTO=Simple` is the cleanest setting: it fixes 256 KiB and equals the default everywhere else,
  because the default already uses Simple from 512 KiB up.
- From 512 KiB up, time grows linearly with size at about 2.7 GB/s bus bandwidth. No algorithm,
  protocol or channel setting changes that. This part needs the host-side fix.
- Channel count, `NCCL_P2P_USE_CUDA_MEMCPY` and `NCCL_P2P_READ_ENABLE` made no visible difference.

**In-model confirmation** (dense, batch 32, decode, eager, P2P on; one cell, three repetitions):

| | default | `NCCL_PROTO=Simple` |
|---|---|---|
| Decode step p50 | 35.96 ms | **28.53 ms** |
| Repetitions | 36.20 / 35.91 / 35.96 | 28.53 / 28.53 / 28.52 |

The 7.4 ms saving matches the microbenchmark prediction of 7.7 ms (64 × 120 µs). The cell is in
`debug_nccl_proto/simple/`. Not yet measured with this setting: the other arms, PaLU-B in
particular, and batches 64 and 128.

## Effect on the benchmark

The numbers in this section are for the default NCCL settings, as used by the 18 completed cells.

Communication per decode step is estimated as 64 × the measured 4-GPU all-reduce time. These are
microbenchmark numbers with calls issued back to back, so treat them as approximate.

| Batch | All-reduce size | Per call, P2P on | Per step, P2P on | Per step, P2P off | Measured dense decode step (P2P on) |
|---|---|---|---|---|---|
| 32 | 256 KiB | 291 µs | 18.6 ms | 20.9 ms | 35.96 ms |
| 64 | 512 KiB | 314 µs | 20.1 ms | 35.2 ms | 47.14 ms |
| 128 | 1 MiB | 566 µs | 36.2 ms | 66.3 ms | not measured |

1. **Arm ordering matches the A5000:** basis_v64 < basis_v96 < dense < palu_g4_r64 < palu_g4_r96 ≪ sharded.
2. **Gaps between arms are compressed.** At batch 32, dense is 7.1 ms slower than basis_v64. Both pay
   the same ~19 ms of communication on top of that. On a node with a healthy interconnect, the
   relative BasisKV advantage would be noticeably larger.
3. **The PaLU-B penalty is inflated.** At batch 32 the sharded R96 arm takes 67.0 ms against 38.5 ms
   for PaLU-A R96. The extra 28.5 ms per step is the rank-pair exchange, which is pure communication.
   On this fabric even a 256 KiB two-GPU all-reduce costs 100 µs. A reviewer could fairly object that
   this node overstates the cost of Scheme B.
4. **Prefill is link-bound.** One 64 MiB all-reduce takes 37 ms, and an 8192-token chunk needs 64 of
   them, so communication alone is about 2.4 s per chunk. TTFT on this node says little about the model.
5. **P2P off would be worse.** The protocol default is `NCCL_P2P_DISABLE=1`, and the A5000 run used it.
   At batch 64 and 128 it roughly doubles the communication time per step on this node.

## Status of the run

**Validation** (P2P off, as in `RUN_L40S.md`): passed.

| Comparison | Top-1 agreement | Mean abs log-prob difference |
|---|---|---|
| dense vLLM vs dense reference | 0.984 | 0.021 |
| PaLU-A vLLM vs PaLU reference | 0.992 | 0.021 |
| PaLU-B vLLM vs PaLU reference | 0.977 | 0.019 |

**Smoke** (P2P on): 7/7 cells `"status": "complete"`, 04:56–05:11 UTC, in `smoke_p2p/eager/`.

**Formal grid, eager, P2P on**: 18 of 42 cells complete, in `formal_p2p/eager/`. Medians over three
repetitions.

| Arm | b32 decode step p50 (ms) | b32 TTFT p50 (s) | b32 TPOT p50 (ms) | b64 decode step p50 (ms) | b64 TTFT p50 (s) | b64 TPOT p50 (ms) |
|---|---|---|---|---|---|---|
| `dense` | 35.96 | 23.37 | 178.8 | 47.14 | 44.20 | 353.8 |
| `basis_v96` | 30.50 | 16.94 | 133.8 | 43.81 | 32.07 | 264.1 |
| `basis_v64` | 28.88 | 15.51 | 123.3 | — | — | — |
| `palu_g4_r96` | 38.54 | 24.00 | 184.0 | — | — | — |
| `palu_g4_r64` | 36.48 | 23.78 | 180.7 | — | — | — |
| `palu_g4_sharded_r96` | 66.98 | 25.01 | 218.0 | — | — | — |
| `palu_g4_sharded_r64` | 65.31 | 24.70 | 214.4 | — | — | — |

The cell running at pause time, `basis_v64` b64 request, was stopped mid-run. It was moved to
`formal_p2p/_interrupted/` together with its log, so that a resumed grid reruns it from scratch. The
benchmark appends to `request_trials.jsonl`, so the cell could not simply be resumed in place.

### Deviations from `RUN_L40S.md` (agreed with the requester)

1. **NCCL P2P enabled** (`NCCL_P2P_DISABLE=0`) for smoke and the formal grid. Validation ran with
   P2P off; the transport does not change the numerics.
2. **Grid scope:**
   - Eager pass: all 7 arms × batches 32/64/128 × request and decode (42 cells).
   - CUDA Graphs pass: decode only, for the 5 non-sharded arms (15 cells), in place of the full
     CUDA Graphs pass.
   - Results go to `formal_p2p/` and `smoke_p2p/` and never mix with protocol-default runs.
3. **Transformers 5.22.2** instead of 5.17.0. This is what the documented install command resolves
   to today. Every cell records its versions.
4. **Code changes** (uncommitted). The defaults are unchanged, so protocol runs behave exactly as before.
   - `launch_vllm_bench.sh`: `NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}`.
   - `bench_llama_vllm.py`:
     - The assertion accepts `NCCL_P2P_DISABLE` of 0 or 1.
     - `nccl_p2p_disabled` in `config.json` now records the actual setting; it was hard-coded `true`.
     - The smoke `dense` cell started before this fix, so it still records `true`.
   - `run_palu_g4_grid.sh`: new `GRID_EXPERIMENTS` override, default `"request decode"`.
   - New: `check_interconnect.sh`, `check_interconnect_copy.py`, `check_interconnect_allreduce.py`.

## Next steps

1. On the replacement node, run the interconnect check below before anything else. Set up the full
   environment and run the grid only if it prints `OVERALL PASS`.
2. Run the same check on the A5000 node. Without it we cannot tell how much of the A5000 timing was
   communication either.
3. Decide the P2P and `NCCL_PROTO` settings for the new run before starting it. The A5000 reference
   used P2P off and default NCCL. The decode estimate lines of the check show what P2P costs on that
   node; whether LL is also slow there needs the 256 KiB comparison above.
4. Keep the 18 cells from this node as a reference point for a PCIe-limited deployment. They are not
   suitable for the main comparison.

## Appendix: interconnect check

`benchmarks/system/check_interconnect.sh` reproduces every measurement above in one command, then
prints a verdict. It takes 3–4 minutes and needs the four GPUs to be idle; do not run it next to a
benchmark. It changes nothing on the machine.

### Running it

From a clone that contains the script:

```bash
cd BasisServe-CALS
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Before the benchmark environment exists: any Python with torch + CUDA.
# On Vast.ai PyTorch images that is /venv/main/bin/python.
PYTHON=/venv/main/bin/python bash benchmarks/system/check_interconnect.sh results/interconnect_<node>.log

# After step 2 of RUN_L40S.md: uses the basis environment, i.e. the same NCCL as the benchmark.
BASIS_ARTIFACTS=/abs/path/to/basis_artifacts bash benchmarks/system/check_interconnect.sh results/interconnect_<node>.log
```

The NCCL version shipped with the interpreter's torch can differ from the benchmark's. For the record
that goes with a grid, rerun the check with the `basis` environment once it is installed.

### What it measures

1. GPU list, `nvidia-smi topo -m`, the P2P capability matrix, and the PCIe link chain from each GPU up
   to the CPU root port (read from sysfs). When idle the current link speed drops to Gen1; the width
   does not change.
2. Host↔device bandwidth per GPU, and aggregate host→device bandwidth with 1, 2 and 4 GPUs copying at
   once.
3. Peer-to-peer copy bandwidth (256 MiB) and latency (256 KiB) for every ordered pair.
4. NCCL all-reduce, BF16, at 256 KiB to 64 MiB:
   - groups: all four GPUs, and every pair;
   - each with `NCCL_P2P_DISABLE=0` and `=1`;
   - with the transport NCCL actually chose.

### Verdict

| Check | Pass if | Node 1 |
|---|---|---|
| Peer access on every pair | all true | PASS |
| NCCL uses P2P when enabled | transport `via P2P…`, no `SHM` | PASS (`P2P/CUMEM`) |
| P2P copy, every pair | ≥ 20 GB/s | **FAIL** (12.58 GB/s) |
| 4-GPU all-reduce, 256 KiB, P2P on | ≤ 100 µs | **FAIL** (291 µs) |
| 4-GPU all-reduce, 64 MiB bus bandwidth, P2P on | ≥ 10 GB/s | **FAIL** (2.71 GB/s) |

The thresholds are practical bars, not vendor specifications. A Gen4 x16 peer path delivers around
25 GB/s, so 20 GB/s leaves headroom. The two all-reduce bars are roughly 3–4 times better than node 1.
A node that clears them spends far less of each decode step in communication.

The script also prints three `info` lines that are not part of the verdict:

- **Aggregate host→device bandwidth by GPU count.** A flat series means one shared upstream link.
- **Estimated decode all-reduce time per step, P2P on.** 64 × the 4-GPU all-reduce time, at batches
  32/64/128. Compare it with the decode step times above.
- **The same estimate with P2P off.** Use the two lines to decide the P2P setting.

Node 1's reference output (`logs/interconnect_l40s_node1.log`):

```
PASS  peer access on every pair: True
PASS  NCCL uses P2P when enabled: via P2P/CUMEM
FAIL  P2P copy >= 20 GB/s on every pair: min 12.58 GB/s
FAIL  4-GPU all-reduce 256 KiB <= 100 us (P2P on): 291.2 us
FAIL  4-GPU all-reduce 64 MiB busbw >= 10 GB/s (P2P on): 2.71 GB/s
info  concurrent host->device aggregate by GPU count: {'1': 12.82, '2': 13.01, '4': 13.07} GB/s (flat = one shared upstream link)
info  decode all-reduce time per step, 64 x 4-GPU all-reduce, P2P on: b32 18.6 ms, b64 20.1 ms, b128 36.2 ms
info  decode all-reduce time per step, 64 x 4-GPU all-reduce, P2P off: b32 20.9 ms, b64 35.2 ms, b128 66.3 ms
OVERALL FAIL
```

If a node fails, do not start the grid. Keep the log and send it back with the node's description
(provider, host, GPU model).
