"""Layer-local continuous ShadowKV adapter output error on the dense teacher tail."""
import argparse
from pathlib import Path
import shlex
import sys
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from basisserve.core.c1_shadowkv import C1ShadowKVState, gather_group
from evaluation.v96kl_common import configure, read_json, write_json, sha256

FIELDS = ('output_squared_error', 'output_energy', 'wo_squared_error', 'wo_energy')


@torch.inference_mode()
def online_errors(q, pre, k, v, cos, sin, wo):
    prefix = 65280
    assert q.shape == (1, 32, 65536, 128)
    state = C1ShadowKVState(pre[:, :, :prefix], k[:, :, :prefix],
                           cos[:, :prefix], sin[:, :prefix], rank=160, budget=2048)
    sums = dict.fromkeys(FIELDS, 0.)
    wo = wo.float()
    vf = v.float()
    maximum_native_difference = 0.
    for position in range(prefix, 65536):
        n = position + 1
        query = q[:, :, position:position+1]
        native = state.decode(query, k[:, :, position:position+1], v[:, :, :n], 128**-.5)
        assert torch.isfinite(native).all()
        ids = state.selected_ids
        fixed = state.fixed_ids.shape[-1]
        routed = ids[:, :, fixed:fixed+state.budget]
        selected_k = torch.cat((state.fixed_key, state.reconstruct(routed), state.generated_key), 2)
        selected_v = gather_group(v, ids)
        assert ids.max() == position and ids.min() >= 0
        assert ids.shape[-1] == 2048 + 384 + 32 + state.steps
        query_fp32 = query[:, :, 0].float().reshape(1, 8, 4, 128)
        dense = ((query_fp32 @ k[:, :, :n].float().transpose(-1, -2)) * 128**-.5).softmax(-1) @ vf[:, :, :n]
        predicted = ((query_fp32 @ selected_k.float().transpose(-1, -2)) * 128**-.5).softmax(-1) @ selected_v.float()
        maximum_native_difference = max(maximum_native_difference,
            float((native.reshape(1, 8, 4, 128).float()-predicted).abs().max()))
        difference = predicted - dense
        values = (difference, dense, difference.reshape(1, -1) @ wo.T, dense.reshape(1, -1) @ wo.T)
        for name, value in zip(FIELDS, values):
            assert torch.isfinite(value).all()
            sums[name] += float(value.double().square().sum())
        if state.steps % 32 == 0:
            print('ONLINE_STEP', state.steps, flush=True)
    return dict(metrics=sums, state=state.statistics(), maximum_native_difference=maximum_native_difference)


def aggregate(windows):
    sums = {m: sum(w['metrics'][m] for w in windows) for m in FIELDS}
    means = {f'{p}_rel_mse': sums[f'{p}_squared_error']/sums[f'{p}_energy'] for p in ('output', 'wo')}
    return sums, means


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['smoke', 'evaluate', 'summarize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    configure()
    root = args.root
    out = root/'shadow_output1024'
    reference = read_json(root/'online_output1024/summary.json')
    identity = read_json(root/'manifests/v128.json')
    assert reference['status'] == 'complete'
    assert sha256(root/'calibration/windows.safetensors') == reference['protocol']['windows_sha256']
    spec = dict(reference_sha256=sha256(root/'online_output1024/summary.json'),
        windows_sha256=reference['protocol']['windows_sha256'], windows=list(range(64, 80)),
        sequence_length=65536, prefill_length=65280, online_decode_tokens=256,
        implementation='C1ShadowKVState adapter; not full official reproduction',
        rank=160, routed=2048, chunk=8, outlier_chunks=48, prompt_local_tokens=32,
        generated_tokens='all 1 through 256, exact K/V', values='Dense V', wo='original',
        arithmetic='FP32 selected QK softmax V with actual reconstructed ShadowKV K; same dense teacher as reference',
        source_sha256={p:sha256(Path(p)) for p in [__file__, 'basisserve/core/c1_shadowkv.py',
            'basisserve/core/compact_v_flash.py']})
    if args.stage == 'summarize':
        reports = [read_json(out/f'layer_{l:03d}.json') for l in range(32)]
        for l, report in enumerate(reports):
            assert report['status'] == 'complete' and report['protocol'] == spec and report['layer'] == l
            assert [w['window'] for w in report['windows']] == list(range(64, 80))
            baseline = read_json(root/'online_output1024'/f'layer_{l:03d}.json')
            for current, old in zip(report['windows'], baseline['windows']):
                for field in ('output_energy', 'wo_energy'):
                    a, b = current['metrics'][field], old['metrics']['exact_k'][field]
                    assert abs(a-b) <= 1e-5*max(abs(b), 1e-12), (l, current['window'], field, a, b)
        means = {m:sum(d['means'][m] for d in reports)/32 for m in ('output_rel_mse','wo_rel_mse')}
        pooled = {f'{p}_rel_mse':sum(d['error_sums'][f'{p}_squared_error'] for d in reports)/sum(d['error_sums'][f'{p}_energy'] for d in reports) for p in ('output','wo')}
        write_json(out/'summary.json', dict(status='complete', protocol=spec, means=means, pooled=pooled,
            comparison_means={**reference['means'], 'shadowkv_adapter':means},
            layers=[dict(layer=d['layer'],means=d['means']) for d in reports]))
        print(means, flush=True)
        return
    assert 0 <= args.shard_index < 4
    if args.stage == 'evaluate':
        smoke = read_json(out/'smoke.json')
        assert smoke['status'] == 'complete' and smoke['protocol'] == spec
    layers = [0] if args.stage == 'smoke' else list(range(8*args.shard_index,8*(args.shard_index+1)))
    model = AutoModelForCausalLM.from_pretrained(identity['model'], dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).eval().cuda()
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    windows = load_file(str(root/'calibration/windows.safetensors'))['input_ids']
    assert tuple(windows.shape) == (80, 65536)
    records = {l:[] for l in layers}
    active = {}
    handles = []
    for layer in layers:
        def capture(module, positional, kwargs, layer=layer):
            x = kwargs['hidden_states']; n = x.shape[1]
            q = module.q_proj(x).view(1,n,32,128).transpose(1,2)
            pre = module.k_proj(x).view(1,n,8,128).transpose(1,2)
            v = module.v_proj(x).view(1,n,8,128).transpose(1,2)
            cos, sin = kwargs['position_embeddings']
            q, k = apply_rotary_pos_emb(q,pre,cos,sin)
            result = online_errors(q,pre,k,v,cos,sin,module.o_proj.weight)
            records[layer].append(dict(window=active['window'], steps=256, **result))
            print(dict(layer=layer,window=active['window'],means=aggregate(records[layer])[1]),flush=True)
        handles.append(model.model.layers[layer].self_attn.register_forward_pre_hook(capture,with_kwargs=True))
    for window in ([64] if args.stage == 'smoke' else range(64,80)):
        active['window'] = window
        result = model.model(windows[window:window+1].long().cuda(),use_cache=False)
        assert torch.isfinite(result.last_hidden_state).all()
        del result
    for handle in handles:
        handle.remove()
    for layer in layers:
        sums, means = aggregate(records[layer])
        path = out/'smoke.json' if args.stage == 'smoke' else out/f'layer_{layer:03d}.json'
        write_json(path,dict(status='complete',layer=layer,protocol=spec,windows=records[layer],
            error_sums=sums,means=means,command=shlex.join(sys.argv),python=sys.executable))


if __name__ == '__main__':
    main()
