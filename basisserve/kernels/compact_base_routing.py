"""Score a compact [Base4 V-code, Residual16 code] cache without materialized Base K."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scores(Z,Q,QR,B,BIAS,COS,SIN,OUT,N:tl.constexpr,ZS0:tl.constexpr,ZS1:tl.constexpr,
            QS:tl.constexpr,QRS:tl.constexpr,SCALE:tl.constexpr):
    group=tl.program_id(0);t=tl.program_id(1)*32+tl.arange(0,32)
    r=tl.arange(0,16);d=tl.arange(0,128);h=tl.arange(0,16)
    z=tl.load(Z+group*ZS0+t[:,None]*ZS1+r[None,:],(t[:,None]<N)&(r[None,:]<4),0)
    right=tl.load(B+group*4*128+r[:,None]*128+d[None,:],r[:,None]<4,0).to(tl.bfloat16)
    pre=tl.dot(z,right).to(tl.bfloat16)
    bias=tl.load(BIAS+group*128+d).to(tl.bfloat16)
    pre=(pre.to(tl.float32)+bias[None,:].to(tl.float32)).to(tl.bfloat16)
    rotated=tl.gather(pre,tl.broadcast_to(((d+64)%128)[None,:],(32,128)),axis=1)
    rotated=(rotated.to(tl.float32)*tl.where(d[None,:]<64,-1.,1.)).to(tl.bfloat16)
    c=tl.load(COS+t[:,None]*128+d[None,:],t[:,None]<N,0)
    s=tl.load(SIN+t[:,None]*128+d[None,:],t[:,None]<N,0)
    a=(pre.to(tl.float32)*c.to(tl.float32)).to(tl.bfloat16)
    b=(rotated.to(tl.float32)*s.to(tl.float32)).to(tl.bfloat16)
    post=(a.to(tl.float32)+b.to(tl.float32)).to(tl.bfloat16)
    q=tl.load(Q+(group*4+h[None,:])*QS+d[:,None],h[None,:]<4,0)
    base_scores=tl.dot(post,q)
    residual=tl.load(Z+group*ZS0+t[:,None]*ZS1+4+r[None,:],t[:,None]<N,0).to(tl.float32)
    qr=tl.load(QR+(group*4+h[:,None])*QRS+r[None,:],h[:,None]<4,0)
    residual_scores=tl.sum(residual[:,None,:]*qr[None,:,:],axis=2)
    score=(base_scores+residual_scores)*SCALE
    tl.store(OUT+(group*4+h[None,:])*N+t[:,None],score,(t[:,None]<N)&(h[None,:]<4))


def compact_base_scores(q,sidecar,residual_query,right,bias,cos,sin,scale):
    assert q.shape==(1,32,1,128) and sidecar.shape[:2]==(1,8) and sidecar.shape[-1]==20
    assert q.dtype==sidecar.dtype==torch.bfloat16
    qr=torch.einsum('bhd,hdr->bhr',q[:,:,0].float(),residual_query.float()).contiguous()
    n=sidecar.shape[2];out=torch.empty(1,32,n,device=q.device,dtype=torch.float32)
    with torch.cuda.device(q.device):
        _scores[(8,triton.cdiv(n,32))](sidecar,q,qr,right.contiguous(),bias.contiguous(),cos,sin,out,
            n,sidecar.stride(1),sidecar.stride(2),q.stride(1),qr.stride(1),scale,num_warps=4)
    return out.view(1,8,4,n)
