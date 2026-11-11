"""Profile-only operator ranges and exclusive CUDA-event accounting.

Ranges are installed after warmup only in diagnostic processes. Captured graph
replays do not reproduce Python ranges: unmatched events remain unattributed.
GPU duration sums include overlap and must never be interpreted as E2E time.
"""
from collections import defaultdict
from functools import wraps
import gzip
import json
import re


def tagged(function, category):
    @wraps(function)
    def call(*args, **kwargs):
        from torch.profiler import record_function
        with record_function('kcuu::' + category):
            return function(*args, **kwargs)
    return call


def install_ranges(model):
    from vllm.distributed import get_tp_group
    group = get_tp_group()
    group.all_reduce = tagged(group.all_reduce, 'allreduce')
    for layer in model.model.layers:
        attention = layer.self_attn
        attention.attn.impl.forward = tagged(attention.attn.impl.forward, 'attention')
        if hasattr(attention, 'o_proj'):
            attention.o_proj.forward = tagged(attention.o_proj.forward, 'output_projection')
    if hasattr(model, 'c1_boundary'):
        boundary = model.c1_boundary
        boundary._decode = tagged(boundary._decode, 'decoder')
        boundary.local_reduce = tagged(boundary.local_reduce, 'decoder')
        # Python resolves __call__ on the class; this is a fresh profile-only process.
        cls = type(boundary)
        cls.__call__ = tagged(cls.__call__, 'output_boundary')
        for prepared in boundary.prepared.values():
            prepared.gather_inplace_fast = tagged(prepared.gather_inplace_fast, 'allgather')
        if boundary.symmetric is not None:
            nccl = boundary.symmetric[0]
            nccl.all_gather = tagged(nccl.all_gather, 'allgather')


def category_for(event, scopes, cpu_name):
    name = event['name'].lower()
    if 'allgather' in name or 'all_gather' in name or 'allgather' in scopes:
        return 'allgather'
    if 'allreduce' in name or 'all_reduce' in name or 'allreduce' in scopes:
        return 'allreduce'
    # Classify only copies inside the output boundary as its layout conversion.
    output_scope = any(s in scopes for s in ('output_boundary', 'decoder', 'output_projection'))
    if output_scope and ('copy' in cpu_name.lower() or 'memcpy' in name):
        return 'layout_conversion'
    for scope in reversed(scopes):
        if scope in ('attention', 'decoder', 'output_projection'):
            return scope
    if 'diffkv' in name or 'flash_fwd' in name or 'flash_attn' in name:
        return 'attention'
    return 'unattributed'


def summarize_events(events):
    # External IDs associate GPU events with their CPU launch/operator events.
    # Build nesting per CPU thread; timestamps on CUDA streams are not CPU ranges.
    threads = defaultdict(list)
    gpu = []
    for event in events:
        if event.get('ph') != 'X' or 'dur' not in event:
            continue
        cat = event.get('cat', '')
        if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            gpu.append(event)
        elif cat in ('cpu_op', 'user_annotation', 'cuda_runtime', 'cuda_driver'):
            threads[(event.get('pid'), event.get('tid'))].append(event)
    external = {}
    for thread in threads.values():
        stack = []
        for event in sorted(thread, key=lambda e: (e['ts'], -e['dur'])):
            while stack and stack[-1]['ts'] + stack[-1]['dur'] <= event['ts']:
                stack.pop()
            scopes = [e['name'].split('::', 1)[1] for e in stack if e['name'].startswith('kcuu::')]
            if event['name'].startswith('kcuu::'):
                scopes.append(event['name'].split('::', 1)[1])
            eid = event.get('args', {}).get('External id')
            if eid is not None and event.get('cat') == 'cpu_op':
                external[eid] = (scopes, event['name'])
            stack.append(event)
    totals = defaultdict(float)
    kernels = defaultdict(lambda: dict(calls=0, gpu_sum_us=0.0))
    for event in gpu:
        scopes, cpu_name = external.get(event.get('args', {}).get('External id'), ([], ''))
        category = category_for(event, scopes, cpu_name)
        totals[category] += event['dur']
        key = (category, event['name'])
        kernels[key]['calls'] += 1
        kernels[key]['gpu_sum_us'] += event['dur']
    total = sum(totals.values())
    return dict(gpu_events=len(gpu), gpu_sum_us=total, categories_gpu_sum_us=dict(totals),
        attributed_fraction=1-totals.get('unattributed', 0)/total if total else 0,
        kernels=[dict(category=c, name=n, **v) for (c,n),v in sorted(kernels.items())],
        notes='Each CUDA event counted once per rank. Sums include overlap, are not E2E or critical-path time. '
              'Decoder/output_projection scopes include projection-associated work; inspect kernel rows for GEMM-only time. '
              'Graph replay lacks Python operator ranges; unclassified kernels remain unattributed. '
              'Do not sum ranks or report missing categories as measured zero.')


def summarize_trace(path):
    with gzip.open(path, 'rt') as handle:
        trace = json.load(handle)
    match = re.search(r'_rank(\d+)\.', path.name)
    assert match is not None, path
    return dict(rank=int(match.group(1)), trace=str(path), **summarize_events(trace['traceEvents']))
