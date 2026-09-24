"""Calibration-side routing quality of the page router for several rank allocations, on held-out windows only.

For every attention layer, every held-out window and a set of query positions, the deployed V96 model is replayed
once and, at each query, the routed 2048-token support (sink 32 + recent 64 + 61 pages of 32, exactly as the
runtime's `page_support`) is computed for each bank and for the exact-K page router. Reported per bank:
  mass          exact softmax mass on the routed support (mean over query heads, groups, queries)
  page_recall   overlap of the routed 61 pages with the exact-K top-61 pages
  page_kl       KL(exact group page distribution || router group page distribution) over routed pages
  required      (synthetic windows) recall of the pages holding the records a tail question needs, and the
                fraction of queries whose required pages are all inside the support
Nothing here touches RULER prompts: --windows gives the C4 validation windows the fits never used, --synthetic a
freshly generated retrieval bank whose metadata records where each record and question sits.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar, conditional_routing_query_projector
from evaluation.diagnose_conjunctive_pages import page_group_scores
from evaluation.fit_k_routing_streaming import install_deployed_teacher
from evaluation.k_routing_config import routing_config, routing_position_embeddings
from evaluation.llama_sink_recent_routing import page_support
from evaluation.v96kl_common import configure, read_json, sha256

PAGE, BUDGET, RECENT, ROUTED_PAGES = 32, 2048, 64, 61


def load_bank(path, layers):
    bank = {}
    for layer in layers:
        t = load_file(str(path / f'layer_{layer:03d}.safetensors'))
        base_tag = next(k for k in t if k.startswith('base_left_')).removeprefix('base_left_')
        res_tag = next(k for k in t if k.startswith('residual_encoder_')).removeprefix('residual_encoder_')
        f = {n: t[f'{n}_{base_tag}'].cuda() for n in ('base_left', 'base_right', 'base_bias')}
        f.update({n: t[f'{n}_{res_tag}'].cuda() for n in ('residual_encoder', 'residual_query')})
        f['base_rank'] = int(base_tag[1:])
        f['projector'] = (f['residual_query'] if f['base_rank'] == 0 else conditional_routing_query_projector(f['residual_query'].cpu())).cuda()
        bank[layer] = f
    return bank


def sidecar_for(f, v_codes, k_post, cos, sin):
    if f['base_rank'] == 0:
        return torch.einsum('bhtd,hdr->bhtr', k_post, f['residual_encoder'].to(k_post.dtype))
    return build_conditional_routing_sidecar(v_codes, k_post, base_left=f['base_left'], base_right=f['base_right'],
                                             base_bias=f['base_bias'], residual_encoder=f['residual_encoder'], cos=cos, sin=sin)


def support_metrics(scores, exact, position, required_pages):
    """scores/exact: [groups, group_heads, length] for one query (length = position + 1)."""
    groups, group_heads, length = exact.shape
    historical = length - RECENT
    ids, valid = page_support(scores[None])                       # [1, groups, 2048]
    ids, valid = ids[0], valid[0]
    routed = ids[:, :BUDGET - RECENT] // PAGE                       # [groups, 62] (page 0 pinned + 61 routed)
    probs = torch.softmax(exact, dim=-1)                            # exact per-head token mass
    gathered = probs.gather(2, ids.clamp_min(0)[:, None, :].expand(groups, group_heads, -1)) * valid[:, None, :]
    mass = gathered.sum(-1).mean().item()
    exact_group = page_group_scores(exact[None], historical)        # [groups, pages]
    router_group = page_group_scores(scores[None], historical)
    exact_top = exact_group.topk(min(ROUTED_PAGES, exact_group.shape[1] - 1), dim=-1).indices
    recall = statistics.mean(len(set(routed[g].tolist()) & set(exact_top[g].tolist())) / exact_top.shape[1] for g in range(groups))
    p = exact_group / exact_group.sum(-1, keepdim=True).clamp_min(1e-12)
    q = router_group / router_group.sum(-1, keepdim=True).clamp_min(1e-12)
    kl = (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(-1).mean().item()
    out = dict(mass=mass, page_recall=recall, page_kl=kl)
    if required_pages:
        recent_pages = set(range(historical // PAGE, (length - 1) // PAGE + 1))
        hits = []
        for g in range(groups):
            have = set(routed[g].tolist()) | {0} | recent_pages
            hits.append(len(required_pages & have) / len(required_pages))
        out['required_recall'] = statistics.mean(hits)
        out['required_all'] = float(all(h == 1.0 for h in hits))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--windows', type=Path, required=True, help='mixed calibration bank dir; its validation_ids are used')
    p.add_argument('--synthetic', type=Path, help='fresh synthetic retrieval bank dir (windows.safetensors + metadata.jsonl)')
    p.add_argument('--bank', action='append', default=[], help='name=dir of a router bank (repeatable)')
    p.add_argument('--sequence-length', type=int, required=True)
    p.add_argument('--rope', choices=('native', 'yarn2', 'yarn4'), required=True)
    p.add_argument('--native-audit', type=Path)
    p.add_argument('--wo-bank', type=Path)
    p.add_argument('--queries-per-window', type=int, default=64)
    p.add_argument('--synthetic-queries', type=int, default=24)
    p.add_argument('--synthetic-offsets', type=int, nargs='+', default=[-8, -6, -4, -2, -1, 0, 1],
                   help='query positions relative to the first token of the first required value (-1 = the step that generates it)')
    p.add_argument('--no-c4', action='store_true', help='skip the C4 validation windows')
    p.add_argument('--probe-question', type=int, default=0, help='also probe every N-th token of the question line (offsets recorded as 1000+distance from the question start)')
    p.add_argument('--prepend-bos', action='store_true', help='replace the first token of every window by BOS (RULER prompts start with BOS; raw calibration windows do not)')
    p.add_argument('--chunk', type=int, default=32)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    configure()
    identity = read_json(args.identity)
    config = routing_config(identity, rope=args.rope, sequence_length=args.sequence_length)
    if config.model_type == 'nemotron_h':
        from evaluation import nemotron_h_triton_mamba as triton_mamba
        triton_mamba.install()
    model = AutoModelForCausalLM.from_pretrained(identity['model'], config=config, dtype=torch.bfloat16,
                                                 attn_implementation='sdpa', local_files_only=True).eval().cuda()
    if config.model_type == 'nemotron_h':
        triton_mamba.restore_dt_limit(model)
    install_deployed_teacher(model, SimpleNamespace(dense_v=False, native_audit=args.native_audit, wo_bank=args.wo_bank,
                                                    identity=args.identity), identity)
    modules = dict(c1_attention_layers(model))
    layers = sorted(modules)
    heads, groups, dim = config.num_attention_heads, config.num_key_value_heads, identity['head_dim']
    group_heads = heads // groups
    if config.model_type == 'llama':
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    elif config.model_type == 'qwen3':
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    stats_cos, stats_sin = routing_position_embeddings(config, args.sequence_length, torch.device('cuda'))
    banks = {}
    for spec in args.bank:
        name, path = spec.split('=', 1)
        banks[name] = load_bank(Path(path), layers)
    names = ['exact'] + list(banks)
    datasets = []
    wm = read_json(args.windows / 'manifest.json')
    windows = load_file(str(args.windows / 'windows.safetensors'))['input_ids']
    assert windows.shape[1] >= args.sequence_length
    grid = torch.linspace(4096, args.sequence_length - 1, args.queries_per_window).long().tolist()
    for index in ([] if args.no_c4 else wm['validation_ids']):
        datasets.append(('c4_validation', int(index), windows[int(index), :args.sequence_length], [(pos, None, None, 0) for pos in grid]))
    if args.synthetic:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(identity['model'], local_files_only=True)
        sw = load_file(str(args.synthetic / 'windows.safetensors'))['input_ids']
        rows = [json.loads(line) for line in (args.synthetic / 'metadata.jsonl').read_text().splitlines()]
        skipped = 0
        shift = 0
        for i, row in enumerate(rows):
            queries = []
            for qrow in row['queries'][:args.synthetic_queries]:
                # `query_token_position` is the start of the Question line; anchor on where the answer text begins.
                # Inside the window the answer follows 'Answer:' so its first token carries a leading space.
                answer_heads = [tokenizer(text, add_special_tokens=False)['input_ids'][:3] for text in (' ' + qrow['answer'], qrow['answer'])]
                start = qrow['query_token_position']; tokens_ = sw[i].tolist(); anchor = None
                for off in range(0, 400):
                    if tokens_[start + off:start + off + 3] in answer_heads:
                        anchor = start + off; break
                if anchor is None:
                    skipped += 1; continue
                # The retrieval step is the generation of the first record's value: anchor on where that value's
                # tokens start inside the answer, and require only that record's pages (the mk2 diagnostic's analogue).
                first = row['records'][qrow['support_records'][0]]
                first_value = first.get('value') or first.get('target')   # aggregation records carry no generated value
                if not first_value:
                    skipped += 1; continue
                value_heads = [tokenizer(text, add_special_tokens=False)['input_ids'][:2] for text in (' ' + first_value, first_value)]
                vpos = None
                for off in range(0, 160):
                    if tokens_[anchor + off:anchor + off + 2] in value_heads:
                        vpos = anchor + off; break
                if vpos is None:
                    skipped += 1; continue
                required = set(range(first['token_start'] // PAGE, (first['token_end'] - 1) // PAGE + 1))
                for off in args.synthetic_offsets:
                    pos = vpos + off
                    if 0 < pos < args.sequence_length:
                        queries.append((pos, required, qrow['query_token_position'], off))
                if args.probe_question:
                    for pos in range(start + 1, anchor, args.probe_question):
                        queries.append((pos, required, qrow['query_token_position'], 1000 + pos - start))
            datasets.append(('synthetic', i, sw[i, :args.sequence_length], queries))
    if args.prepend_bos:
        bos = model.config.bos_token_id if getattr(model.config, 'bos_token_id', None) is not None else AutoTokenizer.from_pretrained(identity['model'], local_files_only=True).bos_token_id
        assert bos is not None
        datasets = [(d, w, torch.cat((torch.tensor([bos], dtype=t.dtype), t[:-1])), [(pos + 1, req, qid, off) for pos, req, qid, off in q]) for d, w, t, q in datasets]
        print('prepended BOS', bos, '(positions shifted by one; page ids unchanged for pages after the first)', flush=True)
        print(f'synthetic queries without a located answer: {skipped}', flush=True)
    from transformers import AutoTokenizer
    print(f'layers {len(layers)}, banks {names}, windows {len(datasets)}', flush=True)
    results = []   # dict(dataset, window, layer, bank, metrics...)
    active = {}

    def capture(attention, positional, kwargs, layer=None):
        x = kwargs['hidden_states'] if 'hidden_states' in kwargs else positional[0]
        n = x.shape[1]
        k = attention.k_proj(x).view(1, n, groups, dim)
        q = attention.q_proj(x).view(1, n, heads, dim)
        v_codes = attention.v_proj(x).view(1, n, groups, -1).transpose(1, 2)
        if config.model_type == 'qwen3':
            q, k = attention.q_norm(q), attention.k_norm(k)
        if config.model_type != 'nemotron_h':
            cos, sin = kwargs['position_embeddings']
            q, k_post = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        else:
            cos, sin = stats_cos, stats_sin
            q, k_post = q.transpose(1, 2), k.transpose(1, 2)
        scaling = attention.scaling
        queries = active['queries']
        positions = [pos for pos, _, _, _ in queries]
        sidecars = {name: sidecar_for(f[layer], v_codes, k_post, cos, sin) for name, f in banks.items()}
        k_rep = k_post.repeat_interleave(group_heads, dim=1)
        for start in range(0, len(positions), args.chunk):
            pos = positions[start:start + args.chunk]
            q_sel = q[:, :, pos].float()                                                        # [1, heads, Q, dim]
            exact = (torch.einsum('bhqd,bhtd->bhqt', q_sel, k_rep.float()) * scaling)[0].view(groups, group_heads, len(pos), n)
            scores = {}
            for name, f in banks.items():
                codes = torch.einsum('bhqd,hdr->bhqr', q_sel, f[layer]['projector'].float())[0].view(groups, group_heads, len(pos), -1)
                scores[name] = torch.einsum('gcqr,gtr->gcqt', codes, sidecars[name][0].float()) * scaling
            for j, position in enumerate(pos):
                length = position + 1
                ex = exact[:, :, j, :length]
                _, required, qid, offset = queries[start + j]
                for name in names:
                    sc = ex if name == 'exact' else scores[name][:, :, j, :length]
                    results.append(dict(dataset=active['dataset'], window=active['window'], layer=layer, bank=name,
                                        position=position, qid=qid, offset=offset, **support_metrics(sc, ex, position, required)))
            del exact, scores, q_sel
        del sidecars, k_rep
        torch.cuda.empty_cache()

    handles = [modules[layer].register_forward_pre_hook(lambda m, a, kw, layer=layer: capture(m, a, kw, layer=layer), with_kwargs=True)
               for layer in layers]
    with torch.inference_mode():
        for dataset, window, tokens, queries in datasets:
            active.update(dataset=dataset, window=window, queries=queries)
            out = model.model(tokens[None].long().cuda(), use_cache=False)
            del out
            print(dict(dataset=dataset, window=window, queries=len(queries), results=len(results)), flush=True)
    for h in handles:
        h.remove()
    summary = {}
    for dataset in sorted({r['dataset'] for r in results}):
        summary[dataset] = {}
        for name in names:
            rows = [r for r in results if r['dataset'] == dataset and r['bank'] == name]
            metrics = [m for m in ('mass', 'page_recall', 'page_kl', 'required_recall', 'required_all') if m in rows[0]]
            summary[dataset][name] = {m: statistics.mean(r[m] for r in rows) for m in metrics}
            if 'required_all' in metrics:
                # per (window, query, layer): was every required page in the support at ANY of the probed steps?
                by_query = {}
                for r in rows: by_query.setdefault((r['window'], r['qid'], r['layer']), []).append(r['required_all'])
                summary[dataset][name]['required_all_any_step'] = statistics.mean(max(v) for v in by_query.values())
                summary[dataset][name]['required_all_by_offset'] = {str(o): statistics.mean(r['required_all'] for r in rows if r['offset'] == o) for o in sorted({r['offset'] for r in rows})}
            summary[dataset][name]['per_layer'] = {str(l): {m: statistics.mean(r[m] for r in rows if r['layer'] == l) for m in metrics} for l in layers}
    for dataset, table in summary.items():
        print(f'\n== {dataset}: mean over {len(layers)} layers')
        scalar = [m for m, v in next(iter(table.values())).items() if isinstance(v, float)]
        print(f"   {'bank':10s}" + ''.join(f'{m:>22s}' for m in scalar))
        for name, row in table.items():
            print(f'   {name:10s}' + ''.join(f'{row[m]:22.4f}' for m in scalar))
            if 'required_all_by_offset' in row:
                print(f'   {"":10s} required_all by offset: ' + ', '.join(f'{o}:{v:.3f}' for o, v in row['required_all_by_offset'].items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Plain json.dump: reruns of this diagnostic overwrite their previous output.
    args.output.write_text(json.dumps(dict(identity=str(args.identity), identity_sha256=sha256(args.identity), windows=str(args.windows),
                                 synthetic=str(args.synthetic) if args.synthetic else None, banks=args.bank, rope=args.rope,
                                 sequence_length=args.sequence_length, queries_per_window=args.queries_per_window,
                                 synthetic_queries=args.synthetic_queries, synthetic_offsets=args.synthetic_offsets,
                                 summary=summary, records=results), indent=1))
    print('written', args.output, flush=True)


if __name__ == '__main__':
    main()
