"""One fresh-process Qwen3-32B TP4 serving experiment using the Figure 3 path."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

from basisserve.core.qwen3_8b_vllm_c1 import file_sha256
from basisserve.core.tp4_c1_factors import load_manifest
from evaluation.benchmark_vllm_qwen3_8b_c1 import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--arm', choices=['dense', 'v64', 'v96'], required=True)
    parser.add_argument('--factor-dir', type=Path)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--prefill-tokens', type=int, default=4096)
    parser.add_argument('--decode-tokens', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.batch > 0 and args.decode_tokens > 1 and args.repeats > 0
    previous = json.loads(args.output.read_text()) if args.output.exists() else None
    assert previous is None or previous['status'] != 'complete', f'Preserve completed result: {args.output}'
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from basisserve.vllm import register
    register()
    assert torch.cuda.device_count() == 4
    assert all(torch.cuda.get_device_capability(i) == (9, 0) for i in range(4))
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((args.model / 'config.json').read_text())
    assert (cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_key_value_heads'],
            cfg['head_dim'], cfg['num_hidden_layers']) == (5120, 64, 8, 128, 64)
    overrides = {}
    factors = None
    if args.arm != 'dense':
        root = args.factor_dir.resolve()
        sha = file_sha256(root / 'results.json')
        manifest = load_manifest(root, sha)
        assert manifest['fit_config']['cache_rank_per_head'] == int(args.arm[1:])
        assert file_sha256(args.model / 'config.json') == manifest['fit_config']['model_config_sha256']
        factors = dict(path=str(root), manifest_sha256=sha, fit_config=manifest['fit_config'])
        overrides = dict(architectures=['BasisServeQwen3TP4C1ForCausalLM'],
                         basisserve_c1_factor_dir=str(root), basisserve_c1_result_sha256=sha)
    configuration = dict(
        model=str(args.model.resolve()), tensor_parallel_size=4, dtype='bfloat16',
        max_model_len=args.prefill_tokens + args.decode_tokens,
        # vLLM warms up with one step of max_num_batched_tokens; it must fit in max_num_seqs sequences.
        max_num_seqs=args.batch,
        max_num_batched_tokens=min(8192, args.batch * (args.prefill_tokens + args.decode_tokens)),
        enable_chunked_prefill=True, enable_prefix_caching=False, async_scheduling=False,
        gpu_memory_utilization=0.8, enforce_eager=False, disable_custom_all_reduce=False,
        disable_log_stats=False, seed=0,
        compilation_config=dict(mode='NONE', cudagraph_mode='FULL_DECODE_ONLY',
                                cudagraph_capture_sizes=[args.batch], cudagraph_num_of_warmups=1),
        hf_overrides=overrides,
        worker_extension_cls='basisserve.vllm.worker_extension.BasisServeWorkerExtension',
    )
    topology = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
    payload = dict(status='initializing', phase='smoke' if args.smoke else 'formal',
                   command=shlex.join([sys.executable, *sys.argv]), conda_environment='basis',
                   slurm_job_id=os.environ.get('SLURM_JOB_ID'), hostname=socket.gethostname(),
                   arm=args.arm, batch=args.batch, configuration=configuration, factors=factors,
                   model_revision=args.model.resolve().name,
                   versions=dict(torch=torch.__version__, cuda=torch.version.cuda, vllm=vllm.__version__),
                   topology=topology,
                   visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version', '--format=csv'], text=True),
                   p2p_access=[[i == j or torch.cuda.can_device_access_peer(i,j) for j in range(4)] for i in range(4)],
                   source_sha256={str(p):file_sha256(p) for p in [Path(__file__),
                       Path('basisserve/core/tp4_c1_factors.py'), Path('basisserve/vllm/qwen3_tp4_c1.py'),
                       Path('basisserve/vllm/prepared_c1_output.py'), Path('basisserve/vllm/diffkv_attention.py'),
                       Path('basisserve/kernels/diffkv_decode.py'), Path('basisserve/kernels/diffkv_prefill.py')]},
                   runs=[], previous_attempt=previous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(args.output)
    save()
    llm = LLM(**configuration)
    workers = llm.collective_rpc('basisserve_tp4_statistics')
    assert len(workers) == 4 and {r['tp_rank'] for r in workers} == {0,1,2,3}
    assert len({r['hostname'] for r in workers}) == 1
    payload['workers_initial'] = workers
    if args.smoke and args.arm != 'dense':
        payload['weight_validation'] = llm.collective_rpc('basisserve_tp4_validate_weights', args=(str(args.model),))
        save()
    tokenizer = llm.get_tokenizer()
    seed = tokenizer.encode('The capital of France is Paris. Tensor parallelism distributes a model across GPUs. ', add_special_tokens=False)
    tail = (seed * ((args.prefill_tokens + len(seed) - 1) // len(seed)))[:args.prefill_tokens]
    prompts = [{'prompt_token_ids': (tokenizer.encode(f'Request {i}: ', add_special_tokens=False) + tail)[:args.prefill_tokens]} for i in range(args.batch)]
    payload['prompt_sha256'] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    sampling = SamplingParams(temperature=0, max_tokens=args.decode_tokens, ignore_eos=True, detokenize=False)
    payload['status'] = 'running'; save()
    run_batch(llm, prompts, sampling, args.decode_tokens)
    for repeat in range(args.repeats):
        result = run_batch(llm, prompts, sampling, args.decode_tokens)
        result['total_output_tokens'] = sum(len(r['output_token_ids']) for r in result['requests'])
        assert result['total_output_tokens'] == args.batch * args.decode_tokens
        decode_window = (max(r['last_token_ts'] for r in result['requests'])
                         - min(r['first_token_ts'] for r in result['requests']))
        result['decode_window_seconds'] = decode_window
        result['decode_tokens_per_second'] = args.batch * (args.decode_tokens - 1) / decode_window
        result['effective_batch'] = len(result['requests'])
        payload['runs'].append(result); save()
        print('RUN', args.arm, args.batch, repeat, result['wall_seconds'], flush=True)
    # Separate short smoke request verifies selected-token log probabilities are finite.
    if args.smoke:
        outputs = llm.generate([prompts[0]], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, detokenize=False, logprobs=1), use_tqdm=False)
        logprobs = outputs[0].outputs[0].logprobs
        assert len(logprobs) == 4 and all(math.isfinite(value.logprob) for row in logprobs for value in row.values())
        payload['finite_logprobs'] = True
    payload['workers'] = llm.collective_rpc('basisserve_tp4_statistics')
    if args.arm != 'dense':
        payload['c1_workers'] = llm.collective_rpc('basisserve_c1_statistics')
        assert all(r['loaded_value_layers'] == 64 and r['capture_calls'] > 0 and r['value_head_size'] == int(args.arm[1:]) for r in payload['c1_workers'])
    payload['status'] = 'complete'; save()
    print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
