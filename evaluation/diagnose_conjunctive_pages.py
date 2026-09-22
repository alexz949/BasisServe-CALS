"""All-required-page recall on niah_multikey_2: does the router retrieve every page the needle lives on?

For each prompt the needle line (key + value) is located in the token stream; its pages are the *required* pages.
Three routers are run with recording hooks on the same prompts: B16R16 (offline sidecar), exact_sparse (exact QK
page mass = the Full-K oracle under the same page budget) and LRQK (online token-level top-k). Per decode step and
attention layer we record, for every KV group, the rank of each required page under the router's own score and
whether it was selected; LRQK is scored at the token level (best rank of any needle token per query head, min over
the group). Pages of distractor lines sharing the needle key's prefix or suffix word are tracked the same way.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import eval_k_routing_ruler as E
from evaluation import llama_sink_recent_routing as sink_routing
from basisserve.core import c1_lrqk
from basisserve.checkpoint import c1_lrqk_qwen3 as lrqk_adapter
from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from evaluation.ruler_v1 import ruler_prompt, sample_score
from evaluation.v96kl_common import configure, read_json

PAGE = 32
STATE = dict(step=0, call=0, records=None, layers=None, arm=None, targets=None)


def page_group_scores(scores, historical):
    """Replicate _selected_pages: per-head log page mass over the historical prefix, GQA max over the group."""
    batch, groups, heads, _ = scores.shape
    proxy = scores[..., :historical].float()
    page_count = math.ceil(historical / PAGE)
    padding = page_count * PAGE - historical
    if padding:
        proxy = torch.nn.functional.pad(proxy, (0, padding), value=-torch.inf)
    log_mass = torch.logsumexp(proxy.reshape(batch, groups, heads, page_count, PAGE), dim=-1)
    log_mass[..., 0] = -torch.inf  # page 0 is pinned, never routed
    normalized = torch.softmax(log_mass, dim=-1)  # per-head page mass over routed pages, as in _selected_pages
    return normalized.max(dim=2).values[0]  # GQA max over the group's heads -> [groups, page_count]


def record_pages(group_scores, selected_pages, layer):
    targets = STATE['targets']
    rank_order = group_scores.clone()
    rank_order[:, 0] = -torch.inf  # page 0 is pinned, not routed
    order = rank_order.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, order.shape[1] + 1, device=order.device).expand_as(order))
    entry = dict(layer=layer, step=STATE['step'])
    for name, pages in targets.items():
        entry[name] = [dict(page=p, rank=ranks[:, p].tolist(), selected=selected_pages[:, p].tolist()) for p in pages if p < ranks.shape[1]]
    STATE['records'].append(entry)


def page_support_hook(scores, budget=2048):
    ids, valid = ORIGINAL_PAGE_SUPPORT(scores, budget)
    batch, groups, heads, length = scores.shape
    if length > budget:
        historical = length - 64
        gs = page_group_scores(scores, historical)
        page_count = gs.shape[1]
        routed_pages = ids[0, :, :budget - 64].reshape(groups, -1, PAGE)[:, :, 0] // PAGE  # [groups, 61]
        if STATE.get('force'):
            # Oracle-union control: guarantee every required page is in the routed set of every group by
            # evicting the lowest-scoring routed page; the 2048 budget is unchanged.
            for p_req in STATE['targets']['required']:
                for g in range(groups):
                    if (routed_pages[g] == p_req).any():
                        continue
                    candidates = gs[g, routed_pages[g]].clone(); candidates[routed_pages[g] == 0] = torch.inf  # never evict the pinned sink page
                    victim = candidates.argmin()
                    routed_pages[g, victim] = p_req
            routed_pages = routed_pages.sort(dim=-1).values
            tokens = (routed_pages[..., None] * PAGE + torch.arange(PAGE, device=scores.device)).flatten(-2)
            ids = torch.cat((tokens[None], ids[:, :, budget - 64:]), -1)
            valid = torch.cat((ids[:, :, :budget - 64] < historical, valid[:, :, budget - 64:]), -1)
        selected = torch.zeros(groups, page_count, dtype=torch.bool, device=scores.device)
        selected.scatter_(1, routed_pages, True)
        record_pages(gs, selected, STATE['layers'][STATE['call'] % len(STATE['layers'])])
    STATE['call'] += 1
    return ids, valid


def select_tokens_hook(qcode, kcode, config):
    out = ORIGINAL_SELECT_TOKENS(qcode, kcode, config)
    batch, heads, length, _ = kcode.shape
    historical = length - min(length, config.recent)
    scores = (kcode[:, :, :historical] @ qcode.transpose(-1, -2)).squeeze(-1)[0]  # [heads, historical]
    order = scores.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, historical + 1, device=order.device).expand_as(order))
    selected_tokens = torch.zeros(heads, length, dtype=torch.bool, device=kcode.device)
    selected_tokens.scatter_(1, out[0], True)
    groups = STATE['kv_heads']; per_group = heads // groups
    entry = dict(layer=STATE['layers'][STATE['call'] % len(STATE['layers'])], step=STATE['step'])
    for name, spans in STATE['token_targets'].items():
        items = []
        for page, toks in spans:
            toks = [t for t in toks if t < historical]
            if not toks:
                continue
            r = ranks[:, toks].min(dim=-1).values.reshape(groups, per_group).min(dim=-1).values  # best needle-token rank per group
            s = selected_tokens[:, toks].any(dim=-1).reshape(groups, per_group).any(dim=-1)
            items.append(dict(page=page, rank=r.tolist(), selected=s.tolist()))
        entry[name] = items
    STATE['records'].append(entry)
    STATE['call'] += 1
    return out


ORIGINAL_PAGE_SUPPORT, ORIGINAL_SELECT_TOKENS = sink_routing.page_support, c1_lrqk.select_tokens


def render(args, tokenizer, source):
    if args.frozen_prompts is not None:
        text = tokenizer.apply_chat_template([dict(role='user', content=source['input'])], tokenize=False,
                                             add_generation_prompt=True) + str(source.get('answer_prefix', ''))
    elif args.prompt_layout == 'chat_nn_no_prefix':
        messages = ([{'role': 'system', 'content': args.system_prompt}] if args.system_prompt else []) + [{'role': 'user', 'content': str(source['input'])}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + '\n\n'
    else:
        messages = ([{'role': 'system', 'content': args.system_prompt}] if args.system_prompt else []) + [{'role': 'user', 'content': ruler_prompt(source)}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return text, enc['input_ids'], enc['offset_mapping']


def locate(text, offsets, source):
    answer = str(source['outputs'][0])
    lines = [l for l in source['input'].split('\n') if answer in l and 'special magic' in l]
    assert len(lines) == 1, lines
    needle = lines[0]
    key = needle.split(' for ')[1].split(' is:')[0]
    prefix, suffix = key.split('-', 1)
    start = text.index(needle); end = start + len(needle)
    def span_tokens(a, b):
        return [i for i, (s, e) in enumerate(offsets) if e > a and s < b and e > s]
    needle_tokens = span_tokens(start, end)
    ks = text.index(key, start); vs = text.index(answer, start)
    key_tokens, value_tokens = span_tokens(ks, ks + len(key)), span_tokens(vs, vs + len(answer))
    required = sorted({t // PAGE for t in needle_tokens})
    def other_pages(pred):
        pages = set()
        for l in source['input'].split('\n'):
            if l is needle or answer in l or ' for ' not in l or ' is:' not in l:
                continue
            k = l.split(' for ')[1].split(' is:')[0]
            if '-' in k and pred(k.split('-', 1)):
                s = text.find(l)
                if s >= 0:
                    pages.update(t // PAGE for t in span_tokens(s, s + len(l)))
        return sorted(pages - set(required))
    prefix_pages = other_pages(lambda ps: ps[0] == prefix)
    suffix_pages = other_pages(lambda ps: ps[1] == suffix)
    return dict(key=key, answer=answer, needle_tokens=needle_tokens, key_tokens=key_tokens, value_tokens=value_tokens,
                required=required, prefix_pages=prefix_pages, suffix_pages=suffix_pages)


@torch.inference_mode()
def generate(model, tokenizer, ids, arm, cap):
    torch.manual_seed(0)
    cache = (lrqk_adapter.C1LRQKCache(config=model.config) if arm == 'lrqk' else E.RoutingCache(model.config))
    device = model.get_input_embeddings().weight.device
    STATE.update(step=0, call=0)
    out = model(input_ids=torch.tensor([ids], device=device), past_key_values=cache, use_cache=True, logits_to_keep=1)
    generated = [int(out.logits[0, -1].argmax())]
    eos = model.config.eos_token_id; eos = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    while len(generated) < cap and generated[-1] not in eos:
        STATE.update(step=len(generated), call=0)
        mask = torch.ones(1, 1, 1, cache.get_seq_length() + 1, device=device, dtype=torch.bool)
        if model.config.model_type == 'nemotron_h':
            mask = {'full_attention': mask, 'linear_attention': None}
        out = model(input_ids=torch.tensor([[generated[-1]]], device=device), past_key_values=cache, attention_mask=mask, use_cache=True, logits_to_keep=1)
        generated.append(int(out.logits[0, -1].argmax()))
    return generated


def first_digit_step(tokenizer, generated, answer):
    text = ''
    for s, t in enumerate(generated):
        text += tokenizer.decode([t], skip_special_tokens=True)
        if answer[:2] in text:
            return s
    return len(generated) - 1


def summarize_arm(results, layers, arm):
    per_layer = {L: dict(any=0, all=0, n=0, miss_rank=[]) for L in layers}
    anyl = dict(any=0, all=0, n=0); correct = 0
    for r in results:
        correct += r['score'] >= 1
        step = r['star_step']; found_any_layer = set(); required = r['required']
        for L in layers:
            recs = [e for e in r['records'] if e['layer'] == L and e['step'] == step]
            if not recs:
                continue
            e = recs[0]
            hit = {item['page'] for item in e['required'] if any(item['selected'])}
            found_any_layer |= hit
            p = per_layer[L]; p['n'] += 1; p['any'] += bool(hit); p['all'] += hit >= set(required)
            for item in e['required']:
                if not any(item['selected']):
                    p['miss_rank'].append(min(item['rank']))
        anyl['n'] += 1; anyl['any'] += bool(found_any_layer); anyl['all'] += found_any_layer >= set(required)
    lines = [f'  [{arm}] answer accuracy {correct}/{len(results)}; required pages/prompt mean {statistics.mean(len(r["required"]) for r in results):.2f}']
    for L in layers:
        p = per_layer[L]
        if p['n']:
            mr = f"{statistics.median(p['miss_rank']):.0f}" if p['miss_rank'] else '-'
            lines.append(f"    layer {L:2d}: >=1 required page found {p['any']/p['n']:.2f}  all found {p['all']/p['n']:.2f}  median rank of missed required page {mr} (budget 61 routed pages)")
    lines.append(f"    any layer: >=1 found {anyl['any']/anyl['n']:.2f}  all found {anyl['all']/anyl['n']:.2f}")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('identity', 'data', 'bank'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--native-audit', type=Path); p.add_argument('--wo-bank', type=Path); p.add_argument('--loki-bank', type=Path)
    p.add_argument('--frozen-prompts', type=Path, help='directory with prompts.json/prompts.safetensors (Llama frozen ids)')
    p.add_argument('--sequence-length', type=int, required=True)
    p.add_argument('--rope', choices=('native', 'yarn2', 'yarn4'), required=True)
    p.add_argument('--samples-per-task', type=int, default=100)
    p.add_argument('--task', default='niah_multikey_2')
    p.add_argument('--arms', default='ours,exact_sparse,lrqk')
    p.add_argument('--lrqk-topk', type=int, default=832)
    p.add_argument('--chat-template', action='store_true'); p.add_argument('--system-prompt'); p.add_argument('--prompt-layout', choices=('completion', 'chat_nn_no_prefix'), default='completion')
    p.add_argument('--limit', type=int); p.add_argument('--shard-index', type=int, default=0); p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--max-new-tokens', type=int, default=24)
    p.add_argument('--force-required', action='store_true', help='oracle-union control: force the required pages into every routed set')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); configure()
    args.arm, args.stage, args.dense_v, args.official_lrqk = 'ours', 'evaluate', False, False
    args.router_fit_count, args.router_diagnostic_count = 32, 0
    args.loki_topk, args.loki_recent = E.LOKI_TOPK, E.LOKI_RECENT
    arms = tuple(args.arms.split(','))
    E.ARMS = tuple(dict.fromkeys(('full',) + arms))
    E.LRQK_TOPK = args.lrqk_topk
    identity = read_json(args.identity)
    tokenizer = AutoTokenizer.from_pretrained(identity['model'], local_files_only=True)
    identity, manifest, rows, bank, _, spec = E.inputs(args, tokenizer, task_names=(args.task,), samples_per_task=args.samples_per_task)
    if args.frozen_prompts is not None:
        frozen = read_json(args.frozen_prompts / 'prompts.json')['rows']; tokens = load_file(str(args.frozen_prompts / 'prompts.safetensors'))
        by = {r['index']: r for r in frozen if r['task'] == args.task}
        assert len(by) == len(rows)
        for row, (index, fr) in zip(rows, sorted(by.items()), strict=True):
            assert fr['ordinal'] == row['ordinal'] and fr['answers'] == row['answers']
            row['input_ids'] = tokens[str(index)].tolist()
    sources = [json.loads(line) for line in (args.data / args.task / 'validation.jsonl').read_text().splitlines()]
    rows = rows[args.shard_index::args.num_shards][:args.limit]
    config = E.routing_config(identity, rope=args.rope, sequence_length=args.sequence_length)
    STATE['kv_heads'] = identity['hkv']; STATE['force'] = args.force_required
    report = dict(status='complete', task=args.task, arms={}, prompts=len(rows), lrqk_topk=args.lrqk_topk, force_required=args.force_required)
    for arm in arms:
        model = E.load_evaluation_model(identity, config)
        if config.model_type == 'nemotron_h':
            E.install_native_runtime(model, args, spec['native'])
        E.install(model, Path(identity['checkpoint']), manifest, arm, bank)
        STATE['layers'] = [i for i, _ in c1_attention_layers(model)]
        sink_routing.page_support = page_support_hook if arm != 'lrqk' else ORIGINAL_PAGE_SUPPORT
        c1_lrqk.select_tokens = select_tokens_hook if arm == 'lrqk' else ORIGINAL_SELECT_TOKENS
        results = []
        for row in rows:
            source = sources[row['ordinal']]
            text, ids, offsets = render(args, tokenizer, source)
            assert ids == row['input_ids'], 'rendered prompt differs from the evaluated prompt'
            loc = locate(text, offsets, source)
            STATE['targets'] = dict(required=loc['required'], prefix_pages=loc['prefix_pages'][:16], suffix_pages=loc['suffix_pages'][:16])
            STATE['token_targets'] = dict(required=[(pg, [t for t in loc['needle_tokens'] if t // PAGE == pg]) for pg in loc['required']],
                                          key_tokens=[(-1, loc['key_tokens'])], value_tokens=[(-2, loc['value_tokens'])])
            STATE['records'] = []
            generated = generate(model, tokenizer, ids, arm, args.max_new_tokens)
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            score = sample_score(prediction, row['answers'], row['match_type'])
            star = first_digit_step(tokenizer, generated, loc['answer'])
            results.append(dict(index=row['index'], key=loc['key'], answer=loc['answer'], required=loc['required'], prefix_pages=len(loc['prefix_pages']),
                                suffix_pages=len(loc['suffix_pages']), score=score, prediction=prediction[:80], star_step=star, records=STATE['records']))
            print(dict(arm=arm, index=row['index'], score=score, required=loc['required'], star_step=star, prediction=prediction[:50]), flush=True)
        print(summarize_arm(results, STATE['layers'], arm), flush=True)
        report['arms'][arm] = dict(layers=STATE['layers'], results=results)
        del model; torch.cuda.empty_cache()
    sink_routing.page_support, c1_lrqk.select_tokens = ORIGINAL_PAGE_SUPPORT, ORIGINAL_SELECT_TOKENS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
