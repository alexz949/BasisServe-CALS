"""TP8 changing-input correctness for the exact vLLM C1 output runtime."""

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from basisserve.vllm.prepared_c1_output import PreparedC1Output


@torch.inference_mode()
def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 8
    device = torch.device("cuda", rank)
    torch.manual_seed(39)
    decoder = torch.randn(2048, 4096, device=device, dtype=torch.bfloat16) / 2048**0.5
    boundary = PreparedC1Output(None, device, 256, 8)
    for batch in (1, 3, 8):
        local = torch.empty(batch, 256, device=device, dtype=torch.bfloat16)
        parts = [torch.empty_like(local) for _ in range(8)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            local.fill_(rank + 1)
            for _ in range(3):
                boundary(local, decoder)
        stream.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = boundary(local, decoder)
            # Sharing the arena across layers must not change an earlier output.
            second_output = boundary(local * 0.5, decoder)
        for step in range(12):
            local.normal_(mean=rank * 0.1 + step * 0.01, std=0.2)
            dist.all_gather(parts, local)
            reference_arena = torch.cat([part.T.contiguous() for part in parts])
            expected = reference_arena.T @ decoder
            graph.replay()
            torch.cuda.synchronize()
            prepared = boundary.prepared[(batch, stream.cuda_stream)]
            torch.testing.assert_close(
                prepared.local_feature_major_view_fast(), local.T * 0.5, rtol=0, atol=0,
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(second_output, expected * 0.5, rtol=0, atol=0)
        del graph, output, second_output
    assert boundary.capture_calls == 6
    if rank == 0:
        print({"status": "passed", "changing_input_replays": 36, **boundary.statistics()}, flush=True)
    boundary.communicator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
