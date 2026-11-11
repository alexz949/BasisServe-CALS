"""One fresh-process 8K/128 TP4 E2E configuration, including request admission."""
from evaluation.kcuu_platform import check_execution, check_devices, identity
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from basisserve.core.qwen3_8b_vllm_c1 import file_sha256
from basisserve.core.tp4_c1_factors import load_manifest
from evaluation.benchmark_vllm_qwen3_8b_c1 import distribution
from evaluation.prepare_kcuu_llama import source_hashes


def run_batch(llm, prompts, sampling):
    before = sum(m.value for m in llm.get_metrics() if m.name == 'vllm:num_preemptions')
    llm.sleep(level=0, mode='keep')
    start = time.perf_counter()
    ids = llm.enqueue(prompts, sampling, use_tqdm=False)
    assert len(ids) == len(prompts)
    llm.wake_up(tags=['scheduling'])
    outputs = llm.wait_for_completion(use_tqdm=False)
    end = time.perf_counter()
    assert len(outputs) == len(prompts)
    requests = []
    for output in outputs:
        m = output.metrics
        tokens = list(output.outputs[0].token_ids)
        assert len(output.prompt_token_ids) == 8192 and len(tokens) == 128
        assert m is not None and not m.is_corrupted
        assert m.last_token_ts >= m.first_token_ts >= m.scheduled_ts >= m.queued_ts
        requests.append(dict(request_id=output.request_id, queued_ts=m.queued_ts,
            scheduled_ts=m.scheduled_ts, first_token_ts=m.first_token_ts, last_token_ts=m.last_token_ts,
            ttft_ms=1000 * (m.first_token_ts - m.queued_ts), output_token_ids=tokens))
    after = sum(m.value for m in llm.get_metrics() if m.name == 'vllm:num_preemptions')
    last_first = max(r['first_token_ts'] for r in requests)
    last_end = max(r['last_token_ts'] for r in requests)
    return dict(wall_seconds=end-start, admission_included=True, total_output_tokens=len(prompts)*128,
                output_tokens_per_second=len(prompts)*128/(end-start), successful_requests=len(outputs),
                ttft_ms=distribution([r['ttft_ms'] for r in requests]), requests=requests,
                preemptions=after-before, last_first_token_ts=last_first,
                last_first_token_since_first_queued_seconds=last_first-min(r['queued_ts'] for r in requests),
                decode_tail_seconds=last_end-last_first,
                decode_tail_note='Active concurrency varies; not fixed-B decode latency.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--factor-dir', type=Path, required=True)
    parser.add_argument('--arm', choices=('dense', 'v64_ag', 'v64_hybrid'), required=True)
    parser.add_argument('--batch', type=int, choices=(32, 64, 128, 256), required=True)
    parser.add_argument('--profile', action='store_true', help='Separate B128 diagnostic run; not an E2E score')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.profile or args.batch == 128
    repeats = 1 if args.profile else 3
    check_execution()
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from basisserve.vllm import register
    register()
    assert torch.cuda.device_count() == 4
    check_devices()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cfg = json.loads((args.model / 'config.json').read_text())
    factor_sha = file_sha256(args.factor_dir / 'results.json')
    manifest = load_manifest(args.factor_dir, factor_sha)
    assert file_sha256(args.model / 'config.json') == manifest['fit_config']['model_config_sha256']
    assert manifest['fit_config']['cache_rank_per_head'] == 64
    overrides = {}
    if args.arm != 'dense':
        architecture = 'BasisServeLlamaTP4C1ForCausalLM' if cfg['model_type'] == 'llama' else 'BasisServeQwen3TP4C1ForCausalLM'
        overrides = dict(architectures=[architecture], basisserve_c1_factor_dir=str(args.factor_dir),
                         basisserve_c1_result_sha256=factor_sha,
                         basisserve_output_mode='hybrid' if args.arm == 'v64_hybrid' else 'ag',
                         basisserve_audit_routes=False)
    config = dict(model=str(args.model), tensor_parallel_size=4, dtype='bfloat16',
        max_model_len=8320, max_num_seqs=args.batch, max_num_batched_tokens=8192,
        enable_chunked_prefill=True, enable_prefix_caching=False, async_scheduling=False,
        gpu_memory_utilization=0.8, enforce_eager=False, disable_custom_all_reduce=False,
        disable_log_stats=False, seed=0,
        compilation_config=dict(mode='NONE', cudagraph_mode='FULL_DECODE_ONLY',
                                cudagraph_capture_sizes=[args.batch], cudagraph_num_of_warmups=1),
        worker_cls='basisserve.vllm.kcuu_worker.KCuuWorker',
        worker_extension_cls='basisserve.vllm.worker_extension.BasisServeWorkerExtension',
        hf_overrides=overrides)
    if args.profile:
        config['profiler_config'] = dict(profiler='torch',
            torch_profiler_dir=str(args.output.resolve().with_suffix('')),
            torch_profiler_with_stack=False, torch_profiler_record_shapes=True,
            torch_profiler_use_gzip=True, detailed_trace_annotation=True)
    previous = json.loads(args.output.read_text()) if args.output.exists() else None
    assert previous is None or previous['status'] != 'complete' or previous['source_sha256'] != source_hashes()
    payload = dict(status='initializing', command=shlex.join([sys.executable, *sys.argv]),
        environment=os.environ.get('CONDA_DEFAULT_ENV'), **identity(),
        arm=args.arm, batch=args.batch, prefill_tokens=8192, output_tokens=128, repeats=repeats, warmups=1,
        profile=args.profile, measurement_role='diagnostic_only' if args.profile else 'formal_e2e',
        configuration=config, model_revision=args.model.name, model_config=cfg,
        model_config_sha256=file_sha256(args.model / 'config.json'), factors_sha256=factor_sha,
        source_sha256=source_hashes(),
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        versions=dict(torch=torch.__version__, vllm=vllm.__version__, cuda=torch.version.cuda),
        gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version', '--format=csv'], text=True),
        topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
        p2p_access=[[i == j or torch.cuda.can_device_access_peer(i,j) for j in range(4)] for i in range(4)],
        runs=[], previous_attempt=previous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2)+'\n')
        temporary.replace(args.output)
    save()
    llm = LLM(**config)
    payload['workers_initial'] = llm.collective_rpc('basisserve_tp4_statistics')
    assert len(payload['workers_initial']) == 4
    assert len({w['hostname'] for w in payload['workers_initial']}) == 1
    tokenizer = llm.get_tokenizer()
    seed = tokenizer.encode('The capital of France is Paris. Tensor parallelism distributes a model across GPUs. ', add_special_tokens=False)
    bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    prompts = []
    for i in range(args.batch):
        prefix = bos + tokenizer.encode(f'Request {i}: ', add_special_tokens=False)
        ids = (prefix + seed * ((8192 + len(seed)-1)//len(seed)))[:8192]
        assert len(ids) == 8192
        prompts.append(dict(prompt_token_ids=ids))
    prompt_file = args.output.parent.parent / 'inputs' / f'{cfg["model_type"]}_b{args.batch}_prompts.json'
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    if prompt_file.exists():
        assert json.loads(prompt_file.read_text()) == prompts
    else:
        prompt_file.write_text(json.dumps(prompts)+'\n')
    payload['prompt_token_ids_file'] = str(prompt_file)
    payload['prompt_sha256'] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    sampling = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True, detokenize=False)
    payload['status'] = 'warming'; save()
    run_batch(llm, prompts, sampling)
    payload['status'] = 'running'; save()
    if args.profile:
        llm.collective_rpc('basisserve_install_profile_ranges')
        trace_dir = Path(config['profiler_config']['torch_profiler_dir'])
        previous_traces = set(trace_dir.glob('*.pt.trace.json.gz'))
        llm.start_profile(profile_prefix=args.output.stem)
    for repeat in range(repeats):
        llm.collective_rpc('basisserve_begin_measurement')
        result = run_batch(llm, prompts, sampling)
        result['repeat'] = repeat
        result['workers'] = llm.collective_rpc('basisserve_measurement_stats')
        first_steps = result['workers'][0]['steps']
        assert all(w['steps'] == first_steps for w in result['workers'])
        active = [s['decode_requests'] for s in first_steps if s['kind'] == 'decode']
        result['pure_decode_active_batch_histogram'] = {str(b):active.count(b) for b in sorted(set(active))}
        result['max_pure_decode_batch'] = max(active, default=0)
        result['target_decode_batch_observed'] = args.batch in active
        result['max_rank_peak_allocated_bytes'] = max(w['peak_allocated_bytes'] for w in result['workers'])
        payload['runs'].append(result); save()
        print('RUN', args.arm, args.batch, repeat, result['wall_seconds'], 'preemptions',result['preemptions'], flush=True)
    if args.profile:
        llm.stop_profile()
        from evaluation.kcuu_profile import summarize_trace
        traces = sorted(set(trace_dir.glob('*.pt.trace.json.gz')) - previous_traces)
        assert len(traces) == 4
        payload['breakdown'] = [summarize_trace(path) for path in traces]
        assert {rank['rank'] for rank in payload['breakdown']} == set(range(4))
        assert all(rank['gpu_events'] > 0 for rank in payload['breakdown'])
    if args.arm != 'dense':
        payload['c1_workers'] = llm.collective_rpc('basisserve_c1_statistics')
        assert all(w['decode_gather'] == 'nccl_symmetric_inplace' for w in payload['c1_workers'])
        assert all(w['loaded_value_layers'] == cfg['num_hidden_layers'] and w['value_head_size'] == 64 for w in payload['c1_workers'])
    payload['status'] = 'complete'; save()
    print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
