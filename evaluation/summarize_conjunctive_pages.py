"""Merge diagnose_conjunctive_pages shards and print the all-required-page recall tables."""
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
shards = [json.loads(p.read_text()) for p in sorted(root.glob('shard_*.json'))]
arms = list(shards[0]['arms'])
print(f'{root.name}: {sum(s["prompts"] for s in shards)} prompts, arms {arms}, LRQK top-k {shards[0]["lrqk_topk"]}')
for arm in arms:
    layers = shards[0]['arms'][arm]['layers']
    results = [r for s in shards for r in s['arms'][arm]['results']]
    n = len(results); correct = sum(r['score'] >= 1 for r in results)
    per_layer = {L: dict(any=0, all=0, n=0, miss=[]) for L in layers}
    star = dict(any=0, all=0); union = dict(any=0, all=0); ge2 = dict(star=0, union=0)
    for r in results:
        req = set(r['required']); found_star = set(); found_union = set()
        for e in r['records']:
            hit = {i['page'] for i in e['required'] if any(i['selected'])}
            found_union |= hit
            if e['step'] == r['star_step']:
                found_star |= hit
                p = per_layer[e['layer']]; p['n'] += 1; p['any'] += bool(hit); p['all'] += hit >= req
                p['miss'] += [min(i['rank']) for i in e['required'] if not any(i['selected'])]
        star['any'] += bool(found_star); star['all'] += found_star >= req; ge2['star'] += len(found_star) >= min(2, len(req))
        union['any'] += bool(found_union); union['all'] += found_union >= req; ge2['union'] += len(found_union) >= min(2, len(req))
    print(f'\n[{arm}] answer accuracy {100*correct/n:.0f}%; required pages per prompt: mean {statistics.mean(len(r["required"]) for r in results):.2f}')
    print(f"  {'layer':>6s} {'>=1 found':>10s} {'all found':>10s} {'median rank of missed required page':>36s}")
    for L in layers:
        p = per_layer[L]
        if p['n']:
            print(f"  {L:6d} {p['any']/p['n']:10.2f} {p['all']/p['n']:10.2f} {('%.0f' % statistics.median(p['miss'])) if p['miss'] else '-':>36s}")
    print(f"  at first-digit step, any layer : >=1 found {star['any']/n:.2f}  >=2 (or all) {ge2['star']/n:.2f}  all found {star['all']/n:.2f}")
    print(f"  union over all decode steps    : >=1 found {union['any']/n:.2f}  >=2 (or all) {ge2['union']/n:.2f}  all found {union['all']/n:.2f}")
