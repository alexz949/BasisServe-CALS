"""Single-CTA V/Base/Residual append and historical Page32 summary update."""
import triton as tr
import triton.language as tl


@tr.jit
def _append(K,V,L,R,BIAS,E,COS,SIN,VC,BC,RC,RING,MN,MX,
            H:tl.constexpr,CAP:tl.constexpr,MCAP:tl.constexpr,POS,SLOT,
            K0:tl.constexpr,K1:tl.constexpr,V0:tl.constexpr,V1:tl.constexpr):
    row=tl.program_id(0);head=row%H
    d=tl.arange(0,128);r=tl.arange(0,16)
    v=tl.load(V+row//H*V0+head*V1+d)
    k=tl.load(K+row//H*K0+head*K1+d)
    left=tl.load(L+head*128*16+d[:,None]*16+r[None,:]).to(tl.float32)
    base=tl.sum(v.to(tl.float32)[:,None]*left,0).to(tl.bfloat16)
    right=tl.load(R+head*16*128+r[:,None]*128+d[None,:]).to(tl.float32)
    pred=tl.sum(base.to(tl.float32)[:,None]*right,0).to(tl.bfloat16)
    bias=tl.load(BIAS+head*128+d)
    pred=(pred.to(tl.float32)+bias.to(tl.float32)).to(tl.bfloat16)
    paired_right=tl.load(R+head*16*128+r[:,None]*128+((d[None,:]+64)%128)).to(tl.float32)
    paired=tl.sum(base.to(tl.float32)[:,None]*paired_right,0).to(tl.bfloat16)
    paired_bias=tl.load(BIAS+head*128+(d+64)%128).to(tl.float32)
    paired=(paired.to(tl.float32)+paired_bias).to(tl.bfloat16).to(tl.float32)*tl.where(d<64,-1.,1.)
    c=tl.load(COS+POS*64+d%64).to(tl.float32)
    s=tl.load(SIN+POS*64+d%64).to(tl.float32)
    a=(pred.to(tl.float32)*c).to(tl.bfloat16).to(tl.float32)
    b=(paired*s).to(tl.bfloat16).to(tl.float32)
    rotated=(a+b).to(tl.bfloat16)
    err=(k.to(tl.float32)-rotated.to(tl.float32)).to(tl.bfloat16)
    enc=tl.load(E+head*128*16+d[:,None]*16+r[None,:]).to(tl.float32)
    residual=tl.sum(err.to(tl.float32)[:,None]*enc,0).to(tl.bfloat16)
    tl.store(VC+(row*CAP+POS)*128+d,v)
    tl.store(BC+(row*CAP+POS)*16+r,base)
    tl.store(RC+(row*CAP+POS)*16+r,residual)
    old=tl.load(RING+(row*64+SLOT)*128+d).to(tl.float32)
    hist=POS-64;off=(row*MCAP+hist//32)*128+d
    lo=tl.load(MN+off).to(tl.float32);hi=tl.load(MX+off).to(tl.float32)
    tl.store(MN+off,tl.where(hist%32==0,old,tl.minimum(lo,old)))
    tl.store(MX+off,tl.where(hist%32==0,old,tl.maximum(hi,old)))
    tl.store(RING+(row*64+SLOT)*128+d,k)


def append(cache,k,v,layer,meta):
    pos=cache.length;f=cache.factors[layer]
    assert cache.width==16 and meta.n==pos-64 and k.shape[2]==1
    _append[(k.shape[0]*k.shape[1],)](k,v,f['base_left_b16'],f['base_right_b16'],f['base_bias_b16'],
        f['encoder'],cache.cos,cache.sin,cache.values[layer],cache.bases[layer],cache.residuals[layer],
        meta.ring,meta.minimum,meta.maximum,k.shape[1],cache.values[layer].shape[2],meta.cap,pos,meta.slot,
        *k.stride()[:2],*v.stride()[:2],num_warps=4,enable_fp_fusion=False)
    meta.n+=1;meta.slot=(meta.slot+1)%64
