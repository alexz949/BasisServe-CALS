"""Independently verify all frozen payload, capture, query and router artifacts."""
import argparse
import hashlib
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoConfig

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.v96kl_common import configure,read_json,write_json,sha256


def attention_payloads(identity, manifest, config):
    assert config.model_type in ('llama', 'qwen3', 'nemotron_h')
    layers = ([i for i, kind in enumerate(config.layers_block_type) if kind == 'full_attention']
        if config.model_type == 'nemotron_h' else list(range(config.num_hidden_layers)))
    assert layers and identity['attention_layers'] == layers
    assert len(identity['layer_ranks']) == len(layers)
    assert sum(identity['layer_ranks']) == 96 * len(layers)
    assert identity['hkv'] == config.num_key_value_heads
    assert identity['hq'] == config.num_attention_heads
    assert identity['hidden_size'] == config.hidden_size
    assert identity['head_dim'] == (getattr(config, 'head_dim', None)
        or config.hidden_size // config.num_attention_heads)
    sources = {row['layer']: row for row in manifest['layers']}
    assert len(sources) == len(manifest['layers']) and set(sources) == set(layers)
    for layer, rank in zip(layers, identity['layer_ranks'], strict=True):
        assert sources[layer]['ranks'] == [rank] * identity['hkv']
    return sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();configure()
    identity=read_json(args.root/'manifests/v96.json')
    checkpoint=Path(identity['checkpoint'])
    manifest=read_json(checkpoint/'manifest.json')
    assert sha256(checkpoint/'manifest.json')==identity['manifest_sha256']
    model_path=Path(identity['model'])
    assert sha256(model_path/'config.json')==identity['model_config_sha256']
    config=AutoConfig.from_pretrained(model_path,trust_remote_code=False,local_files_only=True)
    sources=attention_payloads(identity,manifest,config)
    allocation=read_json(checkpoint/manifest['artifact']['file'])
    assert sha256(checkpoint/manifest['artifact']['file'])==manifest['artifact']['sha256']
    assert allocation['selection']['selected_candidate']=='two_sided_factorized_kl'
    assert allocation['selection']['factorized_method']['exponent']==1
    assert allocation['selection']['selected_schedule']==manifest['compression']['layer_ranks']
    layers=identity['attention_layers']
    assert len(layers)==len(set(layers))==len(identity['layer_ranks'])
    assert sum(identity['layer_ranks'])==96*len(layers)
    files=sorted((args.root/'ours_b16r16').glob('layer_*.safetensors'))
    assert {p.name for p in files}=={f'layer_{i:03d}.safetensors' for i in layers}
    windows=read_json(args.root/'calibration/manifest.json')
    assert windows['status']=='complete'
    assert windows['sha256']==sha256(args.root/'calibration/windows.safetensors')
    records=[]
    common=None
    for layer,rank in zip(layers,identity['layer_ranks'],strict=True):
        source=sources[layer]
        assert source['layer']==layer and source['ranks']==[rank]*identity['hkv']
        assert sha256(checkpoint/source['file'])==source['sha256']
        encoder=load_file(str(checkpoint/source['file']))['value_coordinate_encoders']
        encoder_hash=hashlib.sha256(encoder.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        path=args.root/'ours_b16r16'/f'layer_{layer:03d}.safetensors'
        audit=read_json(path.with_suffix('.json'));t=load_file(str(path))
        assert audit['status']=='complete' and audit['layer']==layer and audit['v_rank']==rank
        assert audit['sha256']==sha256(path)
        spec=audit['protocol']
        if common is None:common=spec
        assert spec==common
        assert spec['v96_manifest_sha256']==identity['manifest_sha256']
        assert spec['windows_sha256']==windows['sha256']
        assert spec['fit_ids']==list(range(64)) and spec['diagnostic_ids']==list(range(64,80))
        assert spec['fit_queries']==64 and spec['diagnostic_queries']==32
        assert spec['base_rank']==spec['residual_rank']==16 and spec['bcd_sweeps']==40
        assert spec['pcg_damping']==spec['pcg_tolerance']==1e-5 and spec['pcg_iterations']==100
        g,h,d=identity['hkv'],identity['hq'],identity['head_dim']
        shapes={'base_left_b16':(g,rank,16),'base_right_b16':(g,16,d),'base_bias_b16':(g,d),
            'residual_encoder_b16_r16':(g,d,16),'residual_query_b16_r16':(h,d,16)}
        assert set(t)==set(shapes)
        for name,shape in shapes.items():
            assert t[name].shape==shape and t[name].dtype==torch.float32 and torch.isfinite(t[name]).all()
        capture=args.root/'calibration'/f'layer_{layer:03d}.json'
        assert sha256(capture)==audit['capture_manifest_sha256']
        capture_spec=read_json(capture)
        assert capture_spec['protocol']==spec and capture_spec['layer']==layer
        if capture_spec.get('storage')=='window_shards':
            seen=set()
            for shard in capture_spec['shards']:
                shard_path=Path(shard['manifest'])
                assert sha256(shard_path)==shard['manifest_sha256']
                shard_record=read_json(shard_path)
                assert shard_record['status']=='complete' and shard_record['layer']==layer
                assert shard_record['protocol']==spec['capture_protocol']
                assert shard_record['window_ids']==shard['window_ids']
                assert shard_record['sha256']==shard['sha256']==sha256(shard_path.with_suffix('.safetensors'))
                assert not seen.intersection(shard['window_ids'])
                seen.update(shard['window_ids'])
            assert seen==set(spec['fit_ids']+spec['diagnostic_ids'])
        else:
            assert capture_spec['sha256']==sha256(capture.with_suffix('.safetensors'))
        qpath=args.root/'qgram'/f'layer_{layer:03d}.json'
        assert sha256(qpath)==audit['query_manifest_sha256']
        q=read_json(qpath)
        assert q['capture_manifest_sha256']==sha256(capture)
        assert q['selection_window_ids']==list(range(64)) and not q['diagnostic_used_for_selection']
        assert q['sha256']==sha256(qpath.with_suffix('.safetensors'))
        queries=load_file(str(qpath.with_suffix('.safetensors')))
        for split,count,windows_count in [('fit',64,64),('diagnostic',32,16)]:
            positions=q['selections'][split]['selected_positions']
            assert positions==sorted(set(positions)) and len(positions)==count
            assert all(sum(p*4//spec['sequence_length']==b for p in positions)==count//4 for b in range(4))
            assert queries[split].shape==(windows_count,count,h,d) and torch.isfinite(queries[split]).all()
        loss=audit['losses']['b16_r16']
        assert len(loss['sweeps'])==40
        metrics=dict(base_fit=audit['reconstruction']['fit']['relative_mse'],
            base_diagnostic=audit['reconstruction']['diagnostic']['relative_mse'],
            residual_fit=loss['fit_page_fisher_nmse'],residual_diagnostic=loss['validation_page_fisher_nmse'])
        assert all(torch.isfinite(torch.tensor(v)) and v>=0 for v in metrics.values())
        records.append(dict(layer=layer,rank=rank,encoder_sha256=encoder_hash,factors_sha256=sha256(path),
            **metrics,seconds=audit['seconds'],
            final_query_maximum_relative_residual=loss['final_query_maximum_relative_residual']))
        print('AUDITED',layer,metrics,flush=True)
    metrics={name:dict(mean=sum(r[name] for r in records)/len(records),
        minimum=min(r[name] for r in records),maximum=max(r[name] for r in records))
        for name in ('base_fit','base_diagnostic','residual_fit','residual_diagnostic')}
    write_json(args.root/'manifests/fit_audit.json',dict(status='complete',attention_layers=layers,
        protocol=common,layers=records,metrics=metrics,
        above_pcg_tolerance=[r['layer'] for r in records if r['final_query_maximum_relative_residual']>1e-5],
        summed_layer_fit_seconds=sum(r['seconds'] for r in records),
        disk_bytes=sum(p.stat().st_size for p in args.root.rglob('*') if p.is_file())))
    print('ALL LAYERS AUDITED',metrics,flush=True)


if __name__=='__main__':main()
