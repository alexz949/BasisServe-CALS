"""Conditional analysis of diagnose_conjunctive_pages shards: accuracy given all/partial/none required pages found,
key-page-found-but-value-page-missed rates for straddling needles, and the ranks of the missed page."""
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1]); shards = [json.loads(p.read_text()) for p in sorted(root.glob('shard_*.json'))]
arms = list(shards[0]['arms']); layers = shards[0]['arms'][arms[0]]['layers']; late = [L for L in layers if L != layers[0]]
print(f'{root.name}: {sum(s["prompts"] for s in shards)} prompts; layers {layers} (conditioning on the later layers {late}, first-digit step)')
for arm in arms:
    results = [r for s in shards for r in s['arms'][arm]['results']]
    buckets = {'all': [], 'partial': [], 'none': []}; straddle = dict(n=0, key_only=0, value_only=0, both=0, none=0); miss_rank_value = []
    for r in results:
        req = r['required']; found = set()
        for e in r['records']:
            if e['step'] == r['star_step'] and e['layer'] in late:
                found |= {i['page'] for i in e['required'] if any(i['selected'])}
        state = 'all' if found >= set(req) else ('partial' if found else 'none'); buckets[state].append(r['score'] >= 1)
        if len(req) == 2:
            straddle['n'] += 1; first, second = req  # the line runs left to right: key tokens first, value digits last
            k = first in found; v = second in found
            straddle['both' if k and v else 'key_only' if k else 'value_only' if v else 'none'] += 1
            if k and not v:
                ranks = [min(i['rank']) for e in r['records'] if e['step'] == r['star_step'] and e['layer'] in late for i in e['required'] if i['page'] == second]
                if ranks: miss_rank_value.append(min(ranks))
    n = len(results)
    print(f'\n[{arm}] accuracy {100*sum(r["score"]>=1 for r in results)/n:.0f}%')
    for k, v in buckets.items():
        print(f"  required pages {k:7s} found in >=1 late layer: {len(v):3d} prompts ({100*len(v)/n:.0f}%), accuracy given that {100*statistics.mean(v):.0f}%" if v else f'  required pages {k:7s}: 0 prompts')
    if straddle['n']:
        s = straddle; print(f"  straddling needles (2 pages): {s['n']} prompts -> both pages {s['both']}, key page only {s['key_only']}, value page only {s['value_only']}, neither {s['none']}"
              + (f"; when only the key page is found the value page's best late-layer rank has median {statistics.median(miss_rank_value):.0f} (budget 61)" if miss_rank_value else ''))
