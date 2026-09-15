"""Fit Loki rank32 on the first two 4096-token segments of each of 64 frozen fit windows."""
import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM
from evaluation.build_qwen3_8b_loki_pca import _fit_pca
from evaluation.v96kl_common import configure,read_json,write_json,sha256

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();configure()
    r=a.root;out=r/'loki128x4k';out.mkdir(exist_ok=True)
    identity=read_json(r/'manifests/v128.json')
    tokens=load_file(str(r/'calibration/windows.safetensors'))['input_ids'][:64,:8192].reshape(128,4096)
    model=AutoModelForCausalLM.from_pretrained(identity['model'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().cuda()
    sums=torch.zeros(32,8,128,dtype=torch.float64,device='cuda');grams=torch.zeros(32,8,128,128,dtype=torch.float64,device='cuda')
    counts=torch.zeros(32,dtype=torch.int64)
    handles=[]
    for l,layer in enumerate(model.model.layers):
        def hook(module,inputs,output,l=l):
            k=output.reshape(1,4096,8,128).permute(0,2,1,3)[0].double()
            sums[l].add_(k.sum(1));grams[l].add_(k.transpose(-1,-2)@k);counts[l]+=4096
        handles.append(layer.self_attn.k_proj.register_forward_hook(hook))
    for w in range(128):
        result=model.model(tokens[w:w+1].long().cuda(),use_cache=False)
        assert torch.isfinite(result.last_hidden_state).all();del result
        print('FIT_WINDOW',w+1,128,flush=True)
    for h in handles:h.remove()
    assert (counts==128*4096).all()
    projector,mean,spectrum,retained=_fit_pca(counts,sums.cpu(),grams.cpu(),rank=32)
    records=[]
    for l in range(32):
        dst=out/f'layer_{l:03d}.safetensors';assert not dst.exists()
        save_file(dict(projector=projector[l].bfloat16(),mean=mean[l],spectrum=spectrum[l]),str(dst))
        records.append(dict(layer=l,file=dst.name,sha256=sha256(dst),retained=retained[l].tolist()))
    write_json(out/'manifest.json',dict(status='complete',layers=records,rank=32,fit_windows=128,sequence_length=4096,
        selection='two consecutive 4096-token segments from the first8192 tokens of frozen fit windows 0..63; each replay resets positions; no diagnostic data',windows_sha256=sha256(r/'calibration/windows.safetensors'),
        coordinate='centered pre-RoPE K PCA; runtime post-RoPE Q/K without centering',model_config_sha256=identity['model_config_sha256'],
        source_sha256=sha256(Path(__file__))))
if __name__=='__main__':main()
