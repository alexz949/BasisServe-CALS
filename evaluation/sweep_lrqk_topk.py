"""Measure LRQK's physical token union per KV group for a per-query-head top-k on a few RULER prompts.

LRQK selects top-k tokens per query head; under GQA the KV group fetches the union over its query heads,
so the attention budget it actually uses is larger than k + recent. This runs the same evaluator path
(`eval_k_routing_ruler`) on one prompt per task for each requested top-k and reports the mean union, so a
top-k can be chosen whose union matches the hard 2048-token budget of the page router.
"""
import argparse
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import eval_k_routing_ruler as E
import json
from evaluation.v96kl_common import configure, read_json
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--native-audit', type=Path)
    p.add_argument('--wo-bank', type=Path)
    p.add_argument('--loki-bank', type=Path)
    p.add_argument('--sequence-length', type=int, required=True)
    p.add_argument('--rope', choices=('native', 'yarn2', 'yarn4'), required=True)
    p.add_argument('--samples-per-task', type=int, required=True)
    p.add_argument('--tasks', default=','.join(E.DEFAULT_TASK_NAMES))
    p.add_argument('--chat-template', action='store_true')
    p.add_argument('--system-prompt')
    p.add_argument('--prompt-layout', choices=('completion', 'chat_nn_no_prefix'), default='completion')
    p.add_argument('--topk', type=int, nargs='+', required=True)
    p.add_argument('--prompts-per-task', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    configure()
    args.arm, args.stage, args.arms, args.dense_v, args.official_lrqk = 'lrqk', 'evaluate', 'lrqk', False, False
    args.router_fit_count, args.router_diagnostic_count = 0, 0
    args.loki_topk, args.loki_recent = E.LOKI_TOPK, E.LOKI_RECENT
    E.ARMS = ('lrqk',)
    task_names = tuple(task.name for task in E.parse_tasks(args.tasks))
    identity = read_json(args.identity)
    tokenizer = AutoTokenizer.from_pretrained(identity['model'], local_files_only=True)
    report = {}
    for topk in args.topk:
        E.LRQK_TOPK = topk
        args.lrqk_topk = topk
        identity, manifest, rows, bank, _, spec = E.inputs(args, tokenizer, task_names=task_names,
                                                          samples_per_task=args.samples_per_task)
        selected = [row for row in rows if row['ordinal'] < args.prompts_per_task]
        config = E.routing_config(identity, rope=args.rope, sequence_length=args.sequence_length)
        model = E.load_evaluation_model(identity, config)
        if config.model_type == 'nemotron_h':
            E.install_native_runtime(model, args, spec['native'])
        E.install(model, Path(identity['checkpoint']), manifest, 'lrqk', bank)
        unions, seconds = [], []
        for row in selected:
            started = time.monotonic()
            with torch.inference_mode():
                _, _, stats, _ = E.generate(model, tokenizer, row, 'lrqk', min(8, row['maximum_tokens']))
            seconds.append(time.monotonic() - started)
            per_layer = [statistics.mean(x for group in layer['physical_union_per_kv_group'] for x in group) for layer in stats]
            unions.append(statistics.mean(per_layer))
            print(dict(topk=topk, task=row['task'], union=round(unions[-1]), seconds=round(seconds[-1], 1)), flush=True)
        report[str(topk)] = dict(selected_per_query_head=stats[0]['selected_per_query_head'],
                                 mean_physical_union_per_kv_group=statistics.mean(unions),
                                 per_prompt={row['task']: u for row, u in zip(selected, unions, strict=True)},
                                 ratio_to_2048=statistics.mean(unions) / 2048, prompts=len(selected),
                                 mean_seconds=statistics.mean(seconds))
        print('TOPK', topk, 'mean union', round(statistics.mean(unions)), f'= {statistics.mean(unions) / 2048:.3f} x 2048', flush=True)
        del model
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(dict(status='complete', model=identity['model'], sequence_length=args.sequence_length, topk=report), indent=1, default=float))


if __name__ == '__main__':
    main()
