"""Exercise real checkpoint Mamba/attention blocks with prefill and cached decode."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import time

import torch
import causal_conv1d
import mamba_ssm
import selective_scan_cuda
import causal_conv1d_cuda
from safetensors import safe_open
from transformers import AutoConfig,DynamicCache
from transformers.models.nemotron_h import modeling_nemotron_h as native

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.v96kl_common import configure,read_json,write_json,sha256


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();configure();torch.manual_seed(20260828)
    print('DEPENDENCIES',mamba_ssm.__version__,causal_conv1d.__version__,flush=True)
    audit=read_json(args.audit);assert audit['status']=='complete'
    path=Path(audit['model'])
    config=AutoConfig.from_pretrained(path,trust_remote_code=False,local_files_only=True)
    config._attn_implementation='sdpa'
    assert sha256(path/'config.json')==audit['config_sha256']
    assert sha256(Path(inspect.getfile(native.NemotronHForCausalLM)))==audit['implementation_sha256']
    index=read_json(path/'model.safetensors.index.json')['weight_map']
    implementations={}
    for name,package in [('mamba2_chunk_scan','mamba_ssm'),
        ('mamba2_selective_state_update','mamba_ssm'),
        ('causal_conv1d_fn','causal_conv1d'),('causal_conv1d_update','causal_conv1d')]:
        function=getattr(native,name)
        implementation=inspect.getclosurevars(function).nonlocals['implementation']
        implementations[name]=implementation.__module__+'.'+implementation.__name__
        assert implementation.__module__.startswith(package),implementations[name]
    print('FAST IMPLEMENTATIONS',implementations,flush=True)
    reports=[]
    for kind in ('linear_attention','full_attention'):
        entry=next(r for r in audit['layers'] if r['kind']==kind)
        layer=entry['layer'];prefix=f'model.layers.{layer}.'
        with torch.device('meta'):
            block=native.NemotronHBlock(config,layer)
        tensors={}
        for source,target in audit['checkpoint_to_native_keys'].items():
            if target.startswith(prefix):
                with safe_open(str(path/index[source]),framework='pt',device='cpu') as data:
                    tensors[target[len(prefix):]]=data.get_tensor(source).to(device='cuda',dtype=torch.bfloat16)
        loaded=block.load_state_dict(tensors,assign=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        block.eval();del tensors
        assert all(p.device.type=='cuda' and torch.isfinite(p).all() for p in block.parameters())
        for n in (129,2049):
            x=torch.randn(1,n,config.hidden_size,device='cuda',dtype=torch.bfloat16)*0.1
            started=time.monotonic();torch.cuda.reset_peak_memory_stats()
            full=block(x)
            cache=DynamicCache(config=config)
            prefix_output=block(x[:,:-1],past_key_values=cache,use_cache=True)
            last=block(x[:,-1:],past_key_values=cache,use_cache=True)
            assert full.shape==x.shape and last.shape==x[:,-1:].shape
            assert torch.isfinite(full).all() and torch.isfinite(last).all()
            relative_rmse=float((full[:,-1:].float()-last.float()).square().sum().sqrt()
                /full[:,-1:].float().square().sum().sqrt())
            assert relative_rmse<0.01,relative_rmse
            record=dict(layer=layer,kind=kind,tokens=n,relative_rmse=relative_rmse,
                maximum_absolute_error=float((full[:,-1:]-last).abs().max()),
                peak_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started)
            reports.append(record);print('PASS',record,flush=True)
            del x,full,last,prefix_output,cache
        del block
        torch.cuda.empty_cache()
    write_json(args.output,dict(status='complete',audit_sha256=sha256(args.audit),
        implementations=implementations,reports=reports,torch=torch.__version__,
        dependencies=dict(mamba=mamba_ssm.__version__,causal_conv1d=causal_conv1d.__version__,
            selective_scan_cuda_sha256=sha256(selective_scan_cuda.__file__),
            causal_conv1d_cuda_sha256=sha256(causal_conv1d_cuda.__file__)),
        gpu=torch.cuda.get_device_name(0),source_sha256=sha256(__file__),
        full_model_tested=False,original_remote_implementation_compared=False))


if __name__=='__main__':main()
