"""Verify reconstructed historical Base K across prefill and decode shapes."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from evaluation.compact_routing_sidecar import restore_routing_sidecar
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
device = parser.parse_args().device
torch.backends.cuda.matmul.allow_tf32 = False
for dtype in (torch.float32, torch.bfloat16):
    generator = torch.Generator(device=device).manual_seed(42)
    def random(shape):
        return torch.randn(shape,device=device,dtype=dtype,generator=generator)
    length = 65536
    v, k = random((1,length+2,8,96)).transpose(1,2), random((1,8,length+2,128))
    angles = random((1,length+2,128))
    cos, sin = angles.cos(), angles.sin()
    factors = dict(base_left=random((8,96,16))/96**0.5,
                   base_right=random((8,16,128))/16**0.5,
                   base_bias=random((8,128)))
    residual_encoder = random((8,128,16))/128**0.5
    stored = build_conditional_routing_sidecar(v[:,:,:length],k[:,:,:length],
        **factors,residual_encoder=residual_encoder,cos=cos[:,:length],sin=sin[:,:length])
    cache_v = v[:,:,:length]
    residual = stored[...,128:].clone()
    for count in (length,length+1,length+2):
        if count > length:
            current = build_conditional_routing_sidecar(v[:,:,count-1:count],k[:,:,count-1:count],
                **factors,residual_encoder=residual_encoder,
                cos=cos[:,count-1:count],sin=sin[:,count-1:count])
            stored = torch.cat((stored,current),2)
            cache_v = torch.cat((cache_v,v[:,:,count-1:count]),2)
        restored = restore_routing_sidecar(cache_v[:,:,:length],residual,
            **factors,cos=cos[:,:length],sin=sin[:,:length])
        if count > length:
            restored = torch.cat((restored,stored[:,:,length:]),2)
        error = ((stored.float()-restored.float()).square().sum()/stored.float().square().sum()).sqrt()
        print(device,str(dtype),count,'equal',torch.equal(stored,restored),'relative_RMSE',float(error),flush=True)
        torch.testing.assert_close(restored,stored,rtol=0,atol=0)
