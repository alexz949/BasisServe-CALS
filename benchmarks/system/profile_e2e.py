"""Nsight capture wrapper: unchanged E2E implementation, four steady steps."""
import sys
import torch
import torch.distributed as dist
from benchmarks.system import bench_e2e_tp4 as benchmark


def main():
    assert '--profile' in sys.argv
    original=benchmark.choose
    calls=0
    def choose(logits,vocab_size):
        nonlocal calls
        token=original(logits,vocab_size);calls+=1
        # Call1 selects the prefill token. Start after 32 decode forwards and
        # stop after four further complete forwards plus their token selection.
        if calls==33:
            torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
            dist.barrier();torch.cuda.synchronize()
            torch.cuda.nvtx.range_push('basis_full_decode_window')
        elif calls==37:
            torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
            dist.barrier();torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        return token
    benchmark.choose=choose
    benchmark.main()
    assert calls==256


if __name__=='__main__':main()
