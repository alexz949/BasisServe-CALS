"""Layer-local continuous Loki output error on the dense teacher tail."""
import argparse
from pathlib import Path
import shlex
import sys
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from basisserve.core.c1_k_routing_sidecar import build_routing_sidecar
from basisserve.kernels.indexed_sparse_decode_attention import gqa_proxy_scores_triton
from evaluation.v96kl_common import configure, read_json, write_json, sha256

FIELDS = ('output_squared_error', 'output_energy', 'wo_squared_error', 'wo_energy')


@torch.inference_mode()
def online_errors(q, k, v, wo, basis):
    codes = build_routing_sidecar(k, basis)
    qb = basis.repeat_interleave(4, dim=0)
    sums = dict.fromkeys(FIELDS, 0.)
    wo = wo.float()
    vf = v.float()
    union_sum = 0.
    for position in range(65280, 65536):
        n = position + 1
        query = q[:, :, position:position+1]
        code = torch.einsum('bhqd,hdr->bhqr',query,qb)
        scores = gqa_proxy_scores_triton(code[:,:,0].contiguous(),codes[:,:,:n],scale=128**-.5)
        ids = scores.topk(2048,dim=-1,sorted=False).indices.reshape(1,8,4,2048)
        qf = query[:,:,0].float().reshape(1,8,4,128)
        logits = (qf @ k[:,:,:n].float().transpose(-1,-2))*128**-.5
        dense = logits.softmax(-1) @ vf[:,:,:n]
        selected_v = torch.gather(vf[:,:,None].expand(1,8,4,65536,128),3,ids[...,None].expand(1,8,4,2048,128))
        predicted = (logits.gather(-1,ids).softmax(-1)[...,None]*selected_v).sum(-2)
        difference = predicted-dense
        for name,value in zip(FIELDS,(difference,dense,difference.reshape(1,-1)@wo.T,dense.reshape(1,-1)@wo.T)):
            assert torch.isfinite(value).all()
            sums[name] += float(value.double().square().sum())
        sorted_ids = ids.reshape(1,8,-1).sort(-1).values
        union_sum += float((sorted_ids[...,1:]!=sorted_ids[...,:-1]).sum(-1).add(1).float().mean())
        if (position-65280+1)%32==0:print('ONLINE_STEP',position-65280+1,flush=True)
    return dict(metrics=sums,mean_union_tokens_per_kv_head=union_sum/256)


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
    out = root/'loki_output2048'
    reference = read_json(root/'online_output1024/summary.json')
    identity = read_json(root/'manifests/v128.json')
    assert reference['status'] == 'complete'
    assert sha256(root/'calibration/windows.safetensors') == reference['protocol']['windows_sha256']
    manifest=read_json(root/'loki/manifest.json')
    assert manifest['status']=='complete' and manifest['rank']==32
    assert manifest['model_config_sha256']==identity['model_config_sha256']
    for rec in manifest['layers']:assert sha256(root/'loki'/rec['file'])==rec['sha256']
    spec = dict(reference_sha256=sha256(root/'online_output1024/summary.json'),
        pca_sha256=sha256(root/'loki/manifest.json'),windows=list(range(64,80)),
        sequence_length=65536,prefill_length=65280,online_decode_tokens=256,
        rank=32,topk_per_query_head=2048,extra_recent=0,extra_sink=0,physical_gqa_union='uncapped',
        calibration=manifest['coordinate'],values='Dense V',wo='original',
        arithmetic='Triton PCA proxy selection, FP32 exact selected QK softmax V, dense teacher inputs',
        source_sha256={p:sha256(Path(p)) for p in [__file__,'basisserve/core/c1_k_routing_sidecar.py',
            'basisserve/kernels/indexed_sparse_decode_attention.py']})
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
            comparison_means={**reference['means'], 'loki':means},
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
    bases={l:load_file(str(root/'loki'/f'layer_{l:03d}.safetensors'))['projector'].cuda() for l in layers}
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
            result = online_errors(q,k,v,module.o_proj.weight,bases[layer])
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
