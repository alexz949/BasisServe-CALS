"""GPU-resident LRQK storage control using the frozen request timing protocol."""

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.system import bench_tp1_offline_request as request
from benchmarks.system.paper_faithful_compat import install_flash_attn_adapter


def upstream_module():
    install_flash_attn_adapter()
    sys.path.insert(0, str(ROOT / "external/LRQK"))
    sys.path.insert(0, str(ROOT / "external/LRQK/cpp_kernel"))
    import lrqk_attention
    return lrqk_attention


def local_class(upstream):
    class Local(upstream.LightAttentionIndicesFactory):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.KVcpu.device = torch.device("cuda:0")
            self.KVcpu.pin_memory = False

    return Local


def gpu_gather(upstream):
    def gather(src, indices, mask):
        assert src.is_cuda and indices.is_cuda and mask.is_cuda
        linear = upstream._take_along_dim_with_mask_python_indices(
            src.shape[1], src.shape[2], indices, mask)
        return torch.index_select(src.view(-1, src.size(-1)), 0, linear)
    return gather


def model(args, audit):
    assert args.method == "lrqk" and args.mode == "request" and audit is None
    upstream = upstream_module()
    upstream.LightAttentionIndicesOffloadPrefill = local_class(upstream)
    upstream.take_along_dim_with_mask_python = gpu_gather(upstream)
    runtime = request.lrqk_model_original(args, None)
    runtime.configuration.update(storage="GPU-local exact K/V", prefill_state_offload=False,
        gather="torch.index_select; unchanged hit/miss and selected indices",
        allocation="upstream capacity and 1.5 growth policy unchanged")
    return runtime


def profile(runtime, prompt, method):
    request.fresh(runtime)
    upstream = upstream_module()
    audit = request.Audit()
    audit.wrap(upstream, "cast_lrqk_prefill", "prompt_factor_fit")
    audit.wrap(upstream.LightAttentionIndicesOffloadPrefill, "prefill", "construction_and_cache_preparation")
    audit.phase = "prefill"
    begin = request.sync_time()
    logits = runtime.prefill(prompt)
    end = request.sync_time()
    first = logits[:, -1].argmax(-1, keepdim=True)
    second = runtime.decode(first)[:, -1].argmax(-1, keepdim=True)
    request.sync_time()
    totals, counts = {}, {}
    for row in audit.records:
        name = row["label"]
        totals[name] = totals.get(name, 0.0) + row["seconds"]
        counts[name] = counts.get(name, 0) + 1
        row["start"] -= begin
        row["stop"] -= begin
    assert all(layer.KVcpu.data.is_cuda and layer.A_K.is_cuda for layer in runtime.state().lwattn.values())
    return dict(component_seconds=totals, component_calls=counts,
        counts_match=counts == {"prompt_factor_fit": 32, "construction_and_cache_preparation": 32},
        construction_including_preparation_seconds=totals["construction_and_cache_preparation"],
        diagnostic_post_prefill_ready_seconds=0.0, profiled_prefill_seconds=end-begin,
        first_two_tokens=torch.cat((first, second), dim=1).tolist(), records=audit.records,
        scope="Separate synchronized profile; fit nested in construction, no post-prefill restore.")


@torch.inference_mode()
def validate_storage():
    upstream = upstream_module()
    cpu_gather = upstream.take_along_dim_with_mask_python
    gpu = gpu_gather(upstream)
    torch.manual_seed(123)
    shape = (1, 2, 256, 16)
    q = torch.randn(1, 4, 256, 16, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    kwargs = dict(num_lite_tokens=8, attn_topk=32, num_key_value_groups=2,
        r=4, max_iter=(2, 2), tol=1e-8, capacity=400)
    cpu = upstream.LightAttentionIndicesFactory(**kwargs)
    local = local_class(upstream)(**kwargs)
    for layer in (cpu, local):
        torch.manual_seed(20260924)
        layer.prefill(q, k, v)
    for step in range(70):
        query = torch.randn(1, 4, 1, 16, device="cuda", dtype=torch.bfloat16)
        key, value = [torch.randn(1, 2, 1, 16, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        upstream.take_along_dim_with_mask_python = cpu_gather
        expected = cpu.decode(query, key, value)
        upstream.take_along_dim_with_mask_python = gpu
        actual = local.decode(query, key, value)
        assert torch.equal(cpu.hit_indices, local.hit_indices)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        assert torch.equal(cpu.A_K, local.A_K)
        assert torch.equal(cpu.KVcpu.data.cuda(), local.KVcpu.data)
    print(json.dumps(dict(status="passed", steps=70, selected_indices="exact match",
        selected_kv="exact match", routing_factors="exact match", full_kv="exact match")))


if __name__ == "__main__":
    if "--validate-storage" in sys.argv:
        validate_storage()
    else:
        request.lrqk_model_original = request.lrqk_model
        request.lrqk_model = model
        request.profile_build = profile
        original_generate = request.generate
        request.generate = lambda runtime, prompt, count, ready_marker=False: original_generate(
            runtime, prompt, count, ready_marker=False)
        request.main()
