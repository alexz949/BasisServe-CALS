"""Per-layer cost of BasisKV _write_prefill pieces at TP1 130048 (and TP4 64K local heads): K host write paths
and routing-code computation, on random data with the forward's real strides."""
import statistics, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
ROOT = Path("/workspace/BasisServe-CALS")
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks/system")]
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from benchmarks.system import audit_tp1_request as frozen
sys.path.insert(0, str(frozen.SNAPSHOT))
from table3_page_router import build
router, _ = build(frozen.SNAPSHOT, Path("/workspace/basis_artifacts/sparse/build/p32_v128_t1984_s32"), 32,
                  value_dim=128, routed_tokens=1984, sink_tokens=32)

def timed(fn, reps=5):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
        out.append(1000 * (time.perf_counter() - t))
    return statistics.median(out)

for T, kv in ((130048, 8), (65536, 2)):
    cap = T + 128
    g = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn(1, T, 4096, device="cuda", dtype=torch.bfloat16, generator=g)
    wk = torch.randn(kv * 128, 4096, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    key = F.linear(hidden, wk).view(1, T, kv, 128).transpose(1, 2)
    value = F.linear(hidden, wk).view(1, T, kv, 128).transpose(1, 2)
    del hidden
    pos = torch.arange(T, device="cuda", dtype=torch.float32)
    freq = pos[:, None] / (500000 ** (torch.arange(64, device="cuda") / 64))[None]
    emb = torch.cat((freq, freq), -1)
    cos, sin = emb.cos().to(torch.bfloat16)[None], emb.sin().to(torch.bfloat16)[None]
    _, key = apply_rotary_pos_emb(key, key, cos, sin)
    cos_half, sin_half = cos[0, :, :64].contiguous(), sin[0, :, :64].contiguous()
    print(f"\nT={T} kv={kv}: key strides {key.stride()} contiguous={key.is_contiguous()}  bytes/layer {key.numel()*2/2**20:.0f} MiB")
    host = router.mapped_host_bf16_empty(1, kv, cap, 128)
    host.zero_()
    ref = key.cpu()
    a = timed(lambda: host[0:1, :, :T].copy_(key)); ok_a = torch.equal(host[0:1, :, :T], ref); host.zero_()
    b = timed(lambda: router.append(host, key, 0)); ok_b = torch.equal(host[0:1, :, :T], ref); host.zero_()
    c = timed(lambda: router.append(host, key.contiguous(), 0)); ok_c = torch.equal(host[0:1, :, :T], ref)
    print(f"  K host write  current copy_ {a:7.1f} ms (ok {ok_a}) | archived append (2D strided) {b:7.1f} ms (ok {ok_b})"
          f" | contiguous + append {c:7.1f} ms (ok {ok_c})")
    base_left = torch.randn(kv, 128, 16, device="cuda", dtype=torch.bfloat16)
    base_right = torch.randn(kv, 16, 128, device="cuda", dtype=torch.bfloat16)
    base_bias = torch.randn(kv, 128, device="cuda", dtype=torch.bfloat16)
    residual_encoder = torch.randn(kv, 128, 16, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.empty(1, kv, cap, 128, device="cuda", dtype=torch.bfloat16)
    base_cache = torch.empty(1, kv, cap, 16, device="cuda", dtype=torch.bfloat16)
    residual_cache = torch.empty_like(base_cache)
    def codes():
        for start in range(0, T, 2048):
            stop = min(T, start + 2048)
            chunk_key, chunk_value = key[:, :, start:stop], value[:, :, start:stop]
            base = torch.einsum("bhtd,hdr->bhtr", chunk_value, base_left)
            predicted = (torch.einsum("bhtr,hrd->bhtd", base, base_right) + base_bias[None, :, None, :]).to(torch.bfloat16)
            c_, s_ = cos_half[start:stop][None, None], sin_half[start:stop][None, None]
            first = ((predicted[..., :64] * c_).to(torch.bfloat16) - (predicted[..., 64:] * s_).to(torch.bfloat16)).to(torch.bfloat16)
            second = ((predicted[..., 64:] * c_).to(torch.bfloat16) + (predicted[..., :64] * s_).to(torch.bfloat16)).to(torch.bfloat16)
            residual = torch.einsum("bhtd,hdr->bhtr", (chunk_key - torch.cat((first, second), dim=-1)).to(torch.bfloat16), residual_encoder)
            value_cache[0:1, :, start:stop].copy_(chunk_value)
            base_cache[0:1, :, start:stop].copy_(base)
            residual_cache[0:1, :, start:stop].copy_(residual)
    d = timed(codes)
    print(f"  routing codes + V/base/residual cache writes {d:7.1f} ms")
    print(f"  x32 layers: current {32*(a+d)/1000:.2f} s  (K copy {32*a/1000:.2f} + codes {32*d/1000:.2f}) | with append {32*(b+d)/1000:.2f} s | contiguous+append {32*(c+d)/1000:.2f} s")
    del host, ref, key, value, value_cache
    torch.cuda.empty_cache()
