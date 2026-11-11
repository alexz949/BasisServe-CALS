"""Four-rank numerical checks of original factors, attention and gathered output."""

import argparse
import json
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import get_tp_group, init_distributed_environment, initialize_model_parallel

from basisserve.core.qwen3_8b_vllm_c1 import file_sha256
from basisserve.core.tp4_c1_factors import load_manifest, load_layer
from basisserve.kernels.diffkv_decode import diffkv_decode
from basisserve.kernels.diffkv_prefill import diffkv_prefill
from basisserve.vllm.prepared_c1_output import PreparedC1Output
from evaluation.benchmark_hopper_warps import make_case, reference

LAYERS = [0, 31, 63]
CASES = [('prefill', 1, 17), ('decode', 1, 1), ('decode', 32, 1)]


def tp_group(world_size, rank):
    """vLLM's TP group as a worker builds it; group setup reads the current vLLM config."""
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    with set_current_vllm_config(config):
        init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank)
        initialize_model_parallel(tensor_model_parallel_size=world_size)
    return get_tp_group()


def original_factors(root, manifest, layer, rank, device):
    """One rank's factors, bit-identical to the published tensors."""
    encoder, decoder = load_layer(root, manifest, layer, rank, device=device, dtype=torch.bfloat16)
    original = load_file(str(root / manifest['artifacts'][str(layer)]['file']))
    assert torch.equal(encoder.cpu(), original['value_coordinate_encoders'][rank*2:rank*2+2])
    assert torch.equal(decoder.cpu(), original['head_output_decoders'].flatten(0,1))
    return encoder, decoder


def local_attention(encoder, rank, mode, batch, query_length, device):
    """One rank's DiffKV attention over encoded values, checked against FP32 attention."""
    width = encoder.shape[-1]
    torch.manual_seed(311 + rank)
    data = make_case(batch,query_length,33,8,width,decode=mode=='decode')
    # Values use the original encoder for each physical KV head.
    raw = torch.randn(*data['v'].shape[:-1],128,device=device,dtype=torch.bfloat16)
    data['v'].copy_(torch.einsum('pthd,hdr->pthr',raw.float(),encoder.float()).to(torch.bfloat16))
    expected_local = reference(data,query_length)
    keys_before = data['k'].clone()
    if mode == 'decode':
        shape=(batch,16,4)
        diffkv_decode(**data, split_threshold=128,max_sequence_length=33,
            softmax_segm_output=torch.empty(*shape,128,device=device),
            softmax_segm_max=torch.empty(shape,device=device),
            softmax_segm_expsum=torch.empty(shape,device=device))
    else:
        diffkv_prefill(**data)
    assert torch.equal(keys_before,data['k'])
    attention_error=((data['out'].float()-expected_local).norm()/expected_local.norm()).item()
    assert torch.isfinite(data['out']).all() and attention_error < 0.01
    return data, expected_local, attention_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factor-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    group = tp_group(4, rank)
    hosts = [None] * 4
    dist.all_gather_object(hosts, socket.gethostname())
    assert len(set(hosts)) == 1 and group.world_size == 4
    root = args.factor_dir.resolve()
    manifest = load_manifest(root, file_sha256(root / 'results.json'))
    width = manifest['fit_config']['cache_rank_per_head']
    boundary = PreparedC1Output(group.device_group, device, 16 * width, 128)
    records = []
    for layer in LAYERS:
        encoder, decoder = original_factors(root, manifest, layer, rank, device)
        for mode,batch,query_length in CASES:
            data, expected_local, attention_error = local_attention(encoder, rank, mode, batch, query_length, device)
            reference_parts=[torch.empty_like(expected_local, memory_format=torch.contiguous_format) for _ in range(4)]
            dist.all_gather(reference_parts,expected_local.contiguous())
            expected=torch.cat(reference_parts,dim=1).flatten(1) @ decoder.float()
            local=data['out'].flatten(1)
            def run():
                if mode=='decode' and boundary.symmetric is not None:
                    nccl,buffer,source=boundary.symmetric
                    tokens=local.shape[0]
                    arena=buffer[:64*width*tokens].view(64*width,tokens)
                    slot=arena[source*16*width:(source+1)*16*width]
                    slot.copy_(local.T)
                    nccl.all_gather(arena.view(-1),slot.reshape(-1))
                    return boundary._decode(arena,decoder)
                return boundary(local,decoder)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    actual=run()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                actual=run()
            graph.replay()
            torch.cuda.synchronize()
            error=((actual.float()-expected).norm()/expected.norm()).item()
            assert torch.isfinite(actual).all() and error < 0.02
            peers=[torch.empty_like(actual) for _ in range(4)]
            dist.all_gather(peers,actual.contiguous())
            assert all(torch.equal(actual,peer) for peer in peers)
            records.append(dict(layer=layer,mode=mode,batch=batch,value_rank=width,
                                attention_relative_l2=attention_error,output_relative_l2=error,
                                rank_outputs_identical=True,keys_unchanged=True,cuda_graph=True))
    all_records=[None]*4
    dist.all_gather_object(all_records,records)
    if rank==0:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(status='complete',hostname=hosts[0],
            factors_sha256=file_sha256(root/'results.json'),workers=all_records,
            collective=boundary.statistics()),indent=2)+'\n')
        print('VALIDATION COMPLETE',args.output,flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
