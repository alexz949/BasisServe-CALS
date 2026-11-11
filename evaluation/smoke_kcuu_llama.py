"""Real Llama/Qwen TP4 short-prompt smoke; no formal performance measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from basisserve.core.qwen3_8b_vllm_c1 import file_sha256
from basisserve.core.tp4_c1_factors import load_manifest
from evaluation.prepare_kcuu_llama import source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--factor-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('ag', 'hybrid'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ['SLURM_NNODES'] == '1'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from basisserve.vllm import register
    register()
    assert torch.cuda.device_count() == 4
    assert all(torch.cuda.get_device_capability(i) == (9, 0) for i in range(4))
    sha = file_sha256(args.factor_dir / 'results.json')
    manifest = load_manifest(args.factor_dir, sha)
    layers = manifest['fit_config']['num_hidden_layers']
    assert layers in (32, 64)
    architecture = 'BasisServeLlamaTP4C1ForCausalLM' if layers == 32 else 'BasisServeQwen3TP4C1ForCausalLM'
    model_config = json.loads((args.model / 'config.json').read_text())
    assert manifest['fit_config']['cache_rank_per_head'] == 64
    assert file_sha256(args.model / 'config.json') == manifest['fit_config']['model_config_sha256']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logits_path = args.output.with_suffix('.safetensors').resolve()
    payload = dict(status='initializing', mode=args.mode, vocab_size=model_config['vocab_size'],
        command=shlex.join([sys.executable, *sys.argv]), environment=os.environ.get('CONDA_DEFAULT_ENV'),
        job=os.environ['SLURM_JOB_ID'], source_sha256=source_hashes(), model_revision=args.model.name,
        factors_sha256=sha, model_config_sha256=file_sha256(args.model / 'config.json'),
        topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
        gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,uuid', '--format=csv'], text=True),
        peer_access=[[i == j or torch.cuda.can_device_access_peer(i, j) for j in range(4)] for i in range(4)],
        versions=dict(torch=torch.__version__, vllm=vllm.__version__, cuda=torch.version.cuda), runs=[])

    def save():
        args.output.write_text(json.dumps(payload, indent=2) + '\n')

    save()
    config = dict(model=str(args.model), tensor_parallel_size=4, dtype='bfloat16',
        max_model_len=8320, max_num_seqs=4, max_num_batched_tokens=64,
        enable_chunked_prefill=True, enable_prefix_caching=False, async_scheduling=False,
        gpu_memory_utilization=0.8, disable_custom_all_reduce=False, seed=0,
        compilation_config=dict(mode='NONE', cudagraph_mode='FULL_DECODE_ONLY',
                                cudagraph_capture_sizes=[1, 2, 4], cudagraph_num_of_warmups=1),
        worker_cls='basisserve.vllm.kcuu_worker.KCuuWorker',
        worker_extension_cls='basisserve.vllm.worker_extension.BasisServeWorkerExtension',
        hf_overrides=dict(architectures=[architecture],
            basisserve_c1_factor_dir=str(args.factor_dir), basisserve_c1_result_sha256=sha,
            basisserve_output_mode=args.mode, basisserve_audit_routes=True))
    payload['configuration'] = config
    llm = LLM(**config)
    payload['weights'] = llm.collective_rpc('basisserve_tp4_validate_weights', args=(str(args.model),))
    llm.collective_rpc('basisserve_reset_audit')
    llm.collective_rpc('basisserve_capture_logits')
    tokenizer = llm.get_tokenizer()
    seed = tokenizer.encode('The capital of France is Paris. ', add_special_tokens=False)
    all_prompts = []
    for index, lengths in enumerate(((17,), (65,), (17, 129), (17, 33, 65))):
        bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
        prompts = [{'prompt_token_ids': bos +
                    (seed * ((length + len(seed) - 1) // len(seed)))[:length - len(bos)]}
                   for length in lengths]
        assert [len(p['prompt_token_ids']) for p in prompts] == list(lengths)
        all_prompts.append(prompts)
        outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=8,
                               ignore_eos=True, detokenize=False), use_tqdm=False)
        assert len(outputs) == len(prompts) and all(len(o.outputs[0].token_ids) == 8 for o in outputs)
        payload['runs'].append(dict(lengths=lengths, output_ids=[list(o.outputs[0].token_ids) for o in outputs]))
        if index == 0:
            llm.collective_rpc('basisserve_save_logits', args=(str(logits_path),))
        save()
    payload['prompt_sha256'] = hashlib.sha256(json.dumps(all_prompts).encode()).hexdigest()
    payload['logits_file'] = str(logits_path)
    payload['logits_sha256'] = file_sha256(logits_path)
    payload['workers'] = llm.collective_rpc('basisserve_tp4_statistics')
    payload['c1_workers'] = llm.collective_rpc('basisserve_c1_statistics')
    payload['steps'] = llm.collective_rpc('basisserve_step_records')
    assert len({w['hostname'] for w in payload['workers']}) == 1
    assert all(w['loaded_value_layers'] == layers for w in payload['c1_workers'])
    assert all(s == payload['steps'][0] for s in payload['steps'])
    assert {s['kind'] for s in payload['steps'][0]} >= {'prefill', 'mixed', 'decode'}
    for worker, steps in zip(payload['c1_workers'], payload['steps']):
        counts = worker['route_counts']
        for kind in ('prefill', 'mixed', 'decode'):
            route = 'ar' if args.mode == 'hybrid' and kind != 'decode' else 'ag'
            assert counts[f'{kind}_{route}'] == layers * sum(s['kind'] == kind for s in steps)
            assert counts[f'{kind}_{"ag" if route == "ar" else "ar"}'] == 0
        assert any(s['kind'] == 'decode' and 'FULL' in s['graph_mode'] for s in steps)
        assert all('NONE' in s['graph_mode'] for s in steps if s['kind'] in ('prefill', 'mixed'))
    payload['status'] = 'passed'
    save()
    print('PASS TP4 runtime smoke', model_config['model_type'], args.mode, flush=True)


if __name__ == '__main__':
    main()
