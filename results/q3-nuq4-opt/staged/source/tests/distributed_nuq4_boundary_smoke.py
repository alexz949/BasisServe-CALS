"""Changing-input smoke of the actual global-stats and A8 vLLM boundary."""

from pathlib import Path
import os
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from basisserve.core.qwen3_nuq4_artifacts import QwenNUQ4Artifacts
from basisserve.kernels.fp8_wire import quantize_e4m3_static, quantize_e4m3_tensorwise_col_major, scaled_mm_e4m3_static
from basisserve.kernels.nuq4_cache import nuq4_value_stats
from basisserve.vllm.nuq4_boundary import NUQ4Boundary


@torch.inference_mode()
def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl")
    device = torch.device("cuda", rank)
    boundary = NUQ4Boundary(None, device, 4, 4)
    checks = 0
    for nominal in (64, 96):
        archive = QwenNUQ4Artifacts(ROOT / "results/q3-kv4-fp8/formal", nominal)
        for index in (0, 18, 35):
            data = archive.layer(index, rank, device)
            r = data["value_ranks"][rank]
            codes, scale = quantize_e4m3_tensorwise_col_major(data["decoder"])
            layer = SimpleNamespace(decoder_fp8=codes, decoder_scale=scale, a8_scale=data["a8_scale"])
            for rows in (1, 4):
                value = torch.zeros(rows, r, device=device, dtype=torch.bfloat16)
                latent = torch.zeros(rows, 4*r, device=device, dtype=torch.bfloat16)
                parts_v = [torch.empty_like(value) for _ in range(8)]
                parts_a = [torch.empty_like(latent) for _ in range(8)]
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        boundary.value_stats(value)
                        boundary.decode(latent, layer)
                stream.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    observed_stats = boundary.value_stats(value).clone()
                    observed = boundary.decode(latent, layer)
                    # The reusable arena must not alias earlier decoder outputs.
                    second = boundary.decode(latent * 0.5, layer)
                for step in range(3):
                    value.normal_(mean=rank*0.1, std=0.3)
                    latent.normal_(std=0.1 + step*0.1)
                    dist.all_gather(parts_v, value)
                    dist.all_gather(parts_a, latent)
                    v = torch.cat(parts_v, dim=1)
                    expected_stats = nuq4_value_stats(v)
                    a = torch.cat(parts_a, dim=1)
                    expected = scaled_mm_e4m3_static(quantize_e4m3_static(a, layer.a8_scale),
                        codes, left_scale=layer.a8_scale, right_scale=scale)
                    expected_second = scaled_mm_e4m3_static(quantize_e4m3_static(a*0.5, layer.a8_scale),
                        codes, left_scale=layer.a8_scale, right_scale=scale)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(observed_stats, expected_stats, rtol=0, atol=0)
                    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                    torch.testing.assert_close(second, expected_second, rtol=0, atol=0)
                    checks += 1
                del graph, observed, second, observed_stats
    dist.barrier()
    if rank == 0:
        print(dict(status="passed", changing_input_cases=checks,
                   capture_calls=boundary.capture_calls, prepared_plans=len(boundary.prepared)), flush=True)
    boundary.stats_comm.close()
    boundary.wire_comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
