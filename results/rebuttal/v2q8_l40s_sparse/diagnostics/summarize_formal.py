import json, statistics as st, sys
from pathlib import Path
R = Path("/workspace/BasisServe-CALS/results/rebuttal/v2q8_l40s_sparse/formal")
G = 2**30
ARMS = ("basis_koff_v128", "basis_koff_v128_reuse", "shadowkv_voff_v128", "lrqk_koff_v128")
for exp, tag in (("exp1", "b1_t130048"), ("exp2", "b16_t65536")):
    print(f"\n######## {exp} {tag}")
    for arm in ARMS:
        print(f"\n== {arm}")
        req = [json.loads(p.read_text()) for p in sorted((R/exp).glob(f"{arm}_request_{tag}_c*/result.json"))]
        sty = [json.loads(p.read_text()) for p in sorted((R/exp).glob(f"{arm}_steady_{tag}_c*/result.json"))]
        brk = [json.loads(p.read_text()) for p in sorted((R/exp).glob(f"{arm}_breakdown_{tag}_c*/result.json"))]
        if not req: print("  no result"); continue
        for d in req:
            ms = [r["measurement"] for r in d["ranks"]]
            print(f"  req c{d['cohort']}: request {max(m['request_seconds'] for m in ms):.2f}s  batch_ready {max(m['batch_ready_seconds'] for m in ms):.2f}s"
                  f"  ttft_mean {st.fmean(ms[0]['ttft_seconds']):.2f}s ttft_max {max(ms[0]['ttft_seconds']):.2f}s placement {max(m['placement_seconds'] for m in ms):.2f}s"
                  f"  decode {max(m['decode_seconds'] for m in ms):.2f}s {max(m['decode_mean_ms_per_step'] for m in ms):.2f}ms/step"
                  f"  peakGPU {max(m['memory_after_prefill']['peak_allocated'] for m in ms)/G:.2f} afterdec {max(m['memory_after_decode']['allocated'] for m in ms)/G:.2f}"
                  f"  devused {max(m['memory_after_decode']['device_used'] for m in ms)/G:.2f} rss {max(m['memory_after_decode']['process_rss'] for m in ms)/G:.2f}"
                  f"  agree {d['cross_rank_tokens_agree']}")
        for d in sty:
            rs = d["ranks"]
            wall = [max(s) for s in zip(*[r["measurement"]["wall_ms"] for r in rs])]
            cuda = [max(s) for s in zip(*[r["measurement"]["cuda_ms"] for r in rs])]
            o = sorted(wall)
            mem = rs[0]["measurement"]["memory"]
            comm = rs[0]["measurement"].get("decode_step_communication") or []
            print(f"  steady c{d['cohort']}: wall mean {st.fmean(wall):.2f} med {st.median(wall):.2f} p95 {o[int(.95*(len(o)-1))]:.2f} cuda med {st.median(cuda):.2f}"
                  f"  tok/s {d['batch']*1000/st.fmean(wall):.1f}  prefill {max(r['measurement']['prefill_seconds'] for r in rs):.2f}s"
                  f"  decode_ready {max(r['measurement']['memory']['decode_ready']['allocated'] for r in rs)/G:.2f} peak_prefill {max(r['measurement']['memory']['after_prefill']['peak_allocated'] for r in rs)/G:.2f}"
                  f"  inv gpu/rank {max(r['inventory']['gpu_backing_bytes'] for r in rs)/G:.2f} host/rank {max(r['inventory']['host_backing_bytes'] for r in rs)/G:.2f} host_all {sum(r['inventory']['host_backing_bytes'] for r in rs)/G:.2f}"
                  f"  rss {max(r['measurement']['memory']['decode_end']['process_rss'] for r in rs)/G:.2f}"
                  + (f"  comm_ms med {st.median(comm):.2f}" if comm and isinstance(comm[0], (int,float)) else ""))
        for d in brk:
            m = d["ranks"][0]["measurement"]
            print("  breakdown prefill (s/request): " + ", ".join(f"{k} {v['seconds_per_request']:.3f}" for k, v in m["prefill"].items() if isinstance(v, dict)))
            print("  breakdown decode (ms/step):    " + ", ".join(f"{k} {v['ms_per_step']:.2f}" for k, v in m["decode"].items() if isinstance(v, dict)))
