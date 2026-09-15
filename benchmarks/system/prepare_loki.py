"""Derive the repository's pre-RoPE centered PCA from existing fit moments."""
import argparse
import hashlib
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from benchmarks.system.common import metadata,save


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'));a=p.parse_args()
    torch.set_num_threads(2);rows=[];folder=a.output/'loki';folder.mkdir(parents=True,exist_ok=True)
    for layer in range(32):
        source=a.root/'moments'/f'layer_{layer:03d}.safetensors'
        data=load_file(str(source),device='cuda')
        count=int(data['fit_count']);assert count==64*65536
        mean=data['fit_sum_k'].double()/count
        covariance=data['fit_kk'].double()/count-mean[:,:,None]*mean[:,None,:]
        covariance=(covariance+covariance.transpose(-1,-2))*.5
        eigenvalues,vectors=torch.linalg.eigh(covariance)
        basis=vectors[:,:,-32:].flip(-1).contiguous()
        torch.testing.assert_close(basis.transpose(-1,-2)@basis,torch.eye(32,device='cuda',dtype=torch.float64).expand(8,-1,-1),rtol=1e-8,atol=1e-8)
        target=folder/f'layer_{layer:03d}.safetensors';assert not target.exists()
        save_file(dict(key_projector=basis.bfloat16().cpu(),fit_mean=mean.cpu()),str(target))
        rows.append(dict(layer=layer,fit_tokens=count,source=str(source),source_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),
            factor_sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),retained_energy=(eigenvalues[:,-32:].sum(-1)/eigenvalues.sum(-1)).tolist()))
    save(folder/'manifest.json',dict(metadata=metadata(),rank=32,calibration='64 x 64K fit windows, existing moments, diagnostic excluded',
        coordinate='centered pre-RoPE K PCA; repository routing applies basis to uncentered post-RoPE Q/K',rows=rows))
    print(dict(status='complete',layers=len(rows),rank=32),flush=True)

if __name__=='__main__':main()
