"""Tables for the page-granularity sweep: overall AMR, router-oracle gap, per-layer spread."""
import json, sys, statistics
for src in sys.argv[1:]:
    d = json.load(open(src)); P, budgets, o, pl = d['page_sizes'], d['budgets'], d['overall'], d['per_layer']
    layers = sorted({int(k.split('|')[0]) for k in pl})
    print(f"== {src}  windows={len(d['windows'])} queries/window={d['queries_per_window']} layers={len(layers)}")
    for b in budgets:
        print(f"-- B={b}")
        print(f"{'P':>8}" + ''.join(f"{p:>9d}" for p in P))
        for s in ('oracle', 'router'):
            print(f"{s:>8}" + ''.join(f"{100*o[f'{s}|P{p}|B{b}']:9.2f}" for p in P))
        for sc in ('oracle', 'router'):
            print(f"{sc+' -L0':>8}" + ''.join(f"{100*statistics.mean(pl[f'{l}|{sc}|P{p}|B{b}'] for l in layers if l != 0):9.2f}" for p in P) + '   (excluding layer 0)')
        print(f"{'gap':>8}" + ''.join(f"{100*(o[f'oracle|P{p}|B{b}']-o[f'router|P{p}|B{b}']):9.2f}" for p in P))
        print(f"{'vs P32':>8}" + ''.join(f"{100*(o[f'router|P{p}|B{b}']-o[f'router|P32|B{b}']):9.2f}" for p in P) + '   (router, relative to P=32)')
        # per-layer spread of the router at each P (min / max over layers)
        for p in (1, 8, 32):
            vals = [100*pl[f'{l}|router|P{p}|B{b}'] for l in layers]
            gaps = [100*(pl[f'{l}|oracle|P{p}|B{b}']-pl[f'{l}|router|P{p}|B{b}']) for l in layers]
            print(f"   P={p:<2d} router per-layer min/median/max {min(vals):6.2f}/{statistics.median(vals):6.2f}/{max(vals):6.2f}   gap max {max(gaps):5.2f} (layer {layers[gaps.index(max(gaps))]})")
