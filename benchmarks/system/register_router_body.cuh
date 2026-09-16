// Experimental Page32 router: each warp owns 16 tokens. MMA D register pairs
// are adjacent RoPE coordinates, consumed before the next output tile.
  static_assert(kPageSize == 32 && (kBaseRank == 8 || kBaseRank == 16));
  constexpr int W = REGISTER_WARPS, T = W*16, P = T/32;
  __shared__ __align__(16) __nv_bfloat16 sb[T*kBaseRank];
  __shared__ __align__(16) __nv_bfloat16 sr[128*kBaseRank];
  __shared__ __nv_bfloat16 sq[kQueriesPerKv*128], bias[128];
  __shared__ __nv_bfloat16 rq[kQueriesPerKv*kResidualRank];
  __shared__ float scores[kQueriesPerKv*T];
  int tid=threadIdx.x, warp=tid/32, lane=tid%32, g=lane/4, u=lane%4;
  int64_t groups=(pages+P-1)/P, group=blockIdx.x%groups;
  int64_t kvrow=blockIdx.x/groups, kv_head=kvrow%kv_heads, batch=kvrow/kv_heads;
  int64_t start=group*T;
  for(int i=tid;i<T*kBaseRank;i+=W*32){
    int t=i/kBaseRank,r=i%kBaseRank;
    sb[i]=start+t<tokens ? static_cast<__nv_bfloat16>(base_code[batch*base_stride_batch+kv_head*base_stride_head+(start+t)*base_stride_token+r]) : __float2bfloat16(0.f);
  }
  // Coalesced reads of the original factor; transpose/pair only inside shared.
  for(int i=tid;i<kBaseRank*128;i+=W*32){
    int r=i/128,d=i%128,paircol=(d%64)*2+d/64;
    sr[paircol*kBaseRank+r]=static_cast<__nv_bfloat16>(base_right[kv_head*kBaseRank*128+i]);
  }
  for(int i=tid;i<kQueriesPerKv*128;i+=W*32){
    int h=i/128,d=i%128;
    sq[i]=static_cast<__nv_bfloat16>(query[batch*query_stride_batch+(kv_head*kQueriesPerKv+h)*query_stride_head+d]);
  }
  for(int i=tid;i<128;i+=W*32)bias[i]=static_cast<__nv_bfloat16>(base_bias[kv_head*128+i]);
  for(int i=tid;i<kQueriesPerKv*kResidualRank;i+=W*32)
    rq[i]=static_cast<__nv_bfloat16>(query_code[(batch*kv_heads+kv_head)*kQueriesPerKv*kResidualRank+i]);
  __syncthreads();
  int t0=warp*16+g,t1=t0+8;
  unsigned a0=*reinterpret_cast<unsigned*>(sb+t0*kBaseRank+u*2);
  unsigned a1=*reinterpret_cast<unsigned*>(sb+t1*kBaseRank+u*2);
#if BASIS_BASE_RANK == 16
  unsigned a2=*reinterpret_cast<unsigned*>(sb+t0*kBaseRank+u*2+8);
  unsigned a3=*reinterpret_cast<unsigned*>(sb+t1*kBaseRank+u*2+8);
#endif
  float sums[2][kQueriesPerKv]={};
#pragma unroll 1
  for(int j=0;j<16;++j){
    unsigned b0=*reinterpret_cast<unsigned*>(sr+(j*8+g)*kBaseRank+u*2);
    float d0,d1,d2,d3;
#if BASIS_BASE_RANK == 8
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
      : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
      : "r"(a0),"r"(a1),"r"(b0),"f"(0.f));
#else
    unsigned b1=*reinterpret_cast<unsigned*>(sr+(j*8+g)*kBaseRank+u*2+8);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
      : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"f"(0.f));
#endif
    int f=j*4+u;
    float lo[2]={d0,d2}, hi[2]={d1,d3};
#pragma unroll
    for(int v=0;v<2;++v){
      int64_t token=start+t0+v*8;
      float a=round_bfloat16(round_bfloat16(lo[v])+__bfloat162float(bias[f]));
      float b=round_bfloat16(round_bfloat16(hi[v])+__bfloat162float(bias[f+64]));
      float c=token<tokens?static_cast<float>(rope_cos[token*rope_stride_token+f]):0.f;
      float s=token<tokens?static_cast<float>(rope_sin[token*rope_stride_token+f]):0.f;
      float x=round_bfloat16(round_bfloat16(a*c)-round_bfloat16(b*s));
      float y=round_bfloat16(round_bfloat16(b*c)+round_bfloat16(a*s));
#pragma unroll
      for(int h=0;h<kQueriesPerKv;++h){
        sums[v][h]=fmaf(x,__bfloat162float(sq[h*128+f]),sums[v][h]);
        sums[v][h]=fmaf(y,__bfloat162float(sq[h*128+f+64]),sums[v][h]);
      }
    }
  }
#pragma unroll
  for(int v=0;v<2;++v){
#pragma unroll
    for(int h=0;h<kQueriesPerKv;++h){
      float z=sums[v][h];
      z+=__shfl_xor_sync(0xffffffffu,z,1,4);
      z+=__shfl_xor_sync(0xffffffffu,z,2,4);
      if(u==0)scores[h*T+t0+v*8]=z;
    }
  }
  __syncthreads();
  for(int item=warp;item<P*kQueriesPerKv;item+=W){
    int p=item/kQueriesPerKv,h=item%kQueriesPerKv;
    int64_t page=group*P+p,token=start+p*32+lane;
    float score=-CUDART_INF_F;
    if(token<tokens){
      float r=0.f;
#pragma unroll
      for(int f=0;f<kResidualRank;++f)
        r=fmaf(__bfloat162float(rq[h*kResidualRank+f]),static_cast<float>(residual_code[batch*residual_stride_batch+kv_head*residual_stride_head+token*residual_stride_token+f]),r);
      score=round_bfloat16(round_bfloat16(round_bfloat16(scores[h*T+p*32+lane])+round_bfloat16(r))*scale);
    }
    float maximum=warp_max(score);
    float sum=warp_sum(__expf(score-maximum));
    if(lane==0 && page<pages)output[batch*output_stride_batch+kv_head*output_stride_head+h*output_stride_query+page]=maximum+__logf(sum);
  }
