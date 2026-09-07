"""Window-sharded PaLU V Fisher on exactly the existing C1 32 x 32K fit tokens."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import sys
import time

import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation import build_llama31_8b_palu_m_checkpoint as builder
from evaluation.collect_gqa_palu_fisher import (
    FORMAT, OFFICIAL_PALU_COMMIT, _allocate_exact_official,
    _chunked_official_palu_loss_and_backward, _configure_projection_gradients,
)


def validate_inputs(args):
    c1 = json.loads((args.c1_checkpoint / 'results.json').read_text())
    fit = c1['fit_config']
    assert c1['status'] == 'complete' and fit['fit_windows'] == 32 and fit['validation_windows'] == 4
    snapshot = Path(fit['snapshot_dir']) / 'manifest.json'
    assert builder._sha256(snapshot) == fit['snapshot_manifest_sha256']
    capture = json.loads(snapshot.read_text())['calibration']
    source = Path(capture['windows_file'])
    assert builder._sha256(source) == capture['windows_sha256']
    model_metadata = builder._model_metadata(args.model)
    assert model_metadata['config_sha256'] == fit['model_config_sha256']
    manifest = json.loads(args.windows.with_name('manifest.json').read_text())
    assert builder._sha256(args.windows) == manifest['artifact']['sha256']
    assert manifest['model']['config_sha256'] == model_metadata['config_sha256']
    tokens = load_file(str(args.windows))['input_ids']
    source_tokens = load_file(str(source))['input_ids']
    assert tokens.shape == (32, 32768) and source_tokens.shape == (36, 32768)
    assert torch.equal(tokens, source_tokens[:32])
    whitening_path = args.whitening_dir / 'manifest.json'
    whitening = json.loads(whitening_path.read_text())
    assert whitening['samples'] == 32 and whitening['sequence_length'] == 32768
    assert whitening['model'] == model_metadata
    assert whitening['windows']['sha256'] == builder._sha256(args.windows)
    assert builder._sha256(args.whitening_dir / whitening['artifact']['file']) == whitening['artifact']['sha256']
    protocol = dict(format='basisserve.palu_c1matched_fisher.v1', model=model_metadata,
        c1_manifest_sha256=builder._sha256(args.c1_checkpoint / 'results.json'),
        c1_snapshot_manifest_sha256=builder._sha256(snapshot), c1_windows_sha256=builder._sha256(source),
        windows_sha256=builder._sha256(args.windows), windows_manifest_sha256=builder._sha256(args.windows.with_name('manifest.json')),
        whitening_manifest_sha256=builder._sha256(whitening_path), whitening_sha256=whitening['artifact']['sha256'],
        exact_c1_fit_tokens_verified=True, heldout_windows_used=False, whitening_reused=True,
        samples=32, sequence_length=32768, loss_chunk_size=args.loss_chunk_size, num_shards=args.num_shards,
        loss='existing official-PaLU reproduction: input tokens[:-1], HF labels tokens[1:]; effective targets tokens[2:]',
        statistic='mean over V weight entries of sqrt(mean over windows of squared window-mean-loss gradients))',
        gradients='V projection weights only; no optimizer or weight updates',
        checkpointing='non-reentrant decoder checkpointing enabled in train mode with dropout verified zero',
        requested_retained_ratio=0.625, allocation='existing official Fisher allocator, block32 rounding unchanged',
        torch=torch.__version__, transformers=transformers.__version__,
        code_sha256={name: builder._sha256(ROOT / name) for name in (
            'evaluation/collect_palu_c1matched_fisher.py', 'evaluation/collect_gqa_palu_fisher.py',
            'evaluation/build_llama31_8b_palu_m_checkpoint.py', 'evaluation/reproduce_palu_paper_llama2_distributed.py')})
    return tokens, manifest, protocol


def fisher_scalar(squared_sum, samples):
    assert samples > 0 and torch.isfinite(squared_sum).all() and (squared_sum >= 0).all()
    return float((squared_sum / samples).sqrt().mean())


def merge(args, manifest, protocol):
    states = []
    rows = []
    for shard in range(args.num_shards):
        prefix = args.output_dir / f'shard_{shard:02d}'
        record = json.loads(prefix.with_suffix('.json').read_text())
        assert record['status'] == 'complete' and record['protocol'] == protocol
        assert record['indices'] == list(range(shard, 32, args.num_shards))
        assert builder._sha256(prefix.with_suffix('.safetensors')) == record['artifact_sha256']
        states.append(load_file(str(prefix.with_suffix('.safetensors'))))
        rows.extend(record['rows'])
    assert sorted(r['index'] for r in rows) == list(range(32))
    assert all(set(s) == set(states[0]) for s in states)
    names = [f'model.layers.{l}.self_attn.v_proj' for l in range(36)]
    assert set(states[0]) == set(names)
    scalars = {}
    for name in names:
        total = sum(s[name].double() for s in states)
        assert total.shape == (1024, 4096)
        scalars[name] = fisher_scalar(total, 32)
        assert scalars[name] > 0
    rank_map, rank_sum, total_rank = _allocate_exact_official(scalars, retained_ratio=0.625)
    ranks = [rank_map[n] for n in names]
    result = dict(format=FORMAT, status='complete', command=shlex.join(sys.argv),
        official_palu_commit=OFFICIAL_PALU_COMMIT, model=protocol['model'], matched_protocol=protocol,
        fisher=dict(target='v_proj_only', dataset='allenai/c4', samples=32, sequence_length=32768,
            loss_semantics=protocol['loss'], loss_chunk_size=args.loss_chunk_size,
            aggregation=protocol['statistic'], scalars=scalars,
            mean_loss=sum(r['loss'] for r in rows) / 32),
        allocation=dict(method='official_palu_fisher_uniform_adapted_to_v_only', rank_block_size=32,
            target='v', requested_retained_ratio=0.625, requested_cache_compression_ratio=0.375,
            rank_map=rank_map, layer_ranks=ranks, rank_sum=rank_sum, total_rank=total_rank,
            realized_retained_ratio=rank_sum / total_rank, realized_cache_compression_ratio=1 - rank_sum / total_rank),
        calibration_windows=dict(path=str(args.windows), sha256=protocol['windows_sha256'],
            manifest_sha256=protocol['windows_manifest_sha256'], sampling=manifest.get('packing')),
        rows=sorted(rows, key=lambda r:r['index']), python=sys.executable)
    builder._atomic_json(args.output_dir / 'fisher.json', result)
    summary = dict(status='complete', exact_c1_fit_tokens_verified=True, samples=32, sequence_length=32768,
        whitening_reused=True, layer_ranks=[r[0] for r in ranks],
        rank_histogram=dict(Counter(r[0] for r in ranks)), actual_average_rank=rank_sum / (36 * 8),
        mean_loss=result['fisher']['mean_loss'],
        median_window_seconds=float(torch.tensor([r['seconds'] for r in rows]).median()),
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows), fisher_sha256=builder._sha256(args.output_dir / 'fisher.json'))
    builder._atomic_json(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def collect(args, tokens, protocol):
    assert torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True, attn_implementation='sdpa').to('cuda:0')
    assert model.config.attention_dropout == 0
    assert all(not isinstance(m, torch.nn.Dropout) or m.p == 0 for m in model.modules())
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    selected = _configure_projection_gradients(model, target='v')
    assert len(selected) == 36
    assert all(l.gradient_checkpointing and l.training for l in model.model.layers)
    recomputations = [0] * 36
    handles = []
    for l, layer in enumerate(model.model.layers):
        def count(module, inputs, index=l):
            recomputations[index] += 1
        handles.append(layer.register_forward_pre_hook(count))
    indices = [0] if args.stage == 'smoke' else list(range(args.shard_index, 32, args.num_shards))
    sums = {n: torch.zeros_like(m.weight, dtype=torch.float32, device='cpu') for n,m in selected}
    rows = []
    for index in indices:
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        before = list(recomputations)
        loss = _chunked_official_palu_loss_and_backward(model, tokens[index:index+1].long().to('cuda:0'),
                                                      chunk_size=args.loss_chunk_size)
        assert torch.isfinite(loss)
        assert all(b-a >= 2 for a,b in zip(before, recomputations, strict=True))
        grad_norms = []
        for name, module in selected:
            grad = module.weight.grad
            assert grad is not None and torch.isfinite(grad).all()
            work = grad.detach().float().cpu()
            sums[name].addcmul_(work, work)
            grad_norms.append(float(work.norm()))
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        row = dict(index=index, loss=float(loss.detach()), seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            v_gradient_norms=grad_norms, all_layers_recomputed=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del loss, grad, work
    for h in handles:
        h.remove()
    prefix = args.output_dir / ('smoke' if args.stage == 'smoke' else f'shard_{args.shard_index:02d}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage != 'smoke':
        builder._atomic_safetensors(prefix.with_suffix('.safetensors'), sums)
    builder._atomic_json(prefix.with_suffix('.json'), dict(status='complete', protocol=protocol,
        indices=indices, rows=rows, artifact_sha256=None if args.stage=='smoke' else builder._sha256(prefix.with_suffix('.safetensors')),
        command=shlex.join(sys.argv), python=sys.executable, gpu=torch.cuda.get_device_name(0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('smoke','collect','merge'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--c1-checkpoint', type=Path, default=ROOT/'results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6')
    p.add_argument('--windows', type=Path, default=ROOT/'results/calibration/qwen3_8b_c4_32f_s32768_palu/windows.safetensors')
    p.add_argument('--whitening-dir', type=Path, default=ROOT/'results/calibration/qwen3_8b_c4_32f_s32768_palu_whitening')
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/calibration/palu_c1matched_fisher32k')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    p.add_argument('--loss-chunk-size', type=int, default=128)
    args = p.parse_args()
    assert args.loss_chunk_size > 0 and 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    builder.activate_model_profile('qwen3_8b')
    builder.SEQUENCE_LENGTH = 32768
    tokens, manifest, protocol = validate_inputs(args)
    if args.stage == 'merge':
        merge(args, manifest, protocol)
    else:
        prefix = args.output_dir / ('smoke.json' if args.stage == 'smoke' else f'shard_{args.shard_index:02d}.json')
        if prefix.exists():
            record = json.loads(prefix.read_text())
            assert record['status'] == 'complete' and record['protocol'] == protocol
            print('Already complete:', prefix, flush=True)
            return
        collect(args, tokens, protocol)


if __name__ == '__main__':
    main()
