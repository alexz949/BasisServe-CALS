"""CPU-only provenance/header checks and reviewable Llama GPU smoke commands."""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from importlib.metadata import version
from safetensors import safe_open
from huggingface_hub import HfApi
from basisserve.core.qwen3_8b_vllm_c1 import file_sha256
from basisserve.core.tp4_c1_factors import load_manifest

MODEL = Path('/u/azhang35/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659')
FACTORS = Path('/projects/bido/azhang35/factor-banks/llama31-8b-instruct/R64-S6-D0')
OUTPUT = Path('results/rebuttal/kcuu_gh200_tp4_8k')
SOURCE_FILES = [
    'basisserve/core/tp4_c1_factors.py', 'basisserve/core/tp4_step.py',
    'basisserve/vllm/tp4_c1_model.py', 'basisserve/vllm/llama_tp4_c1.py',
    'basisserve/vllm/qwen3_tp4_c1.py', 'basisserve/vllm/prepared_c1_output.py',
    'basisserve/vllm/diffkv_attention.py', 'basisserve/vllm/kcuu_worker.py',
    'basisserve/vllm/worker_extension.py', 'basisserve/vllm/__init__.py',
    'basisserve/kernels/diffkv_decode.py', 'basisserve/kernels/diffkv_prefill.py',
    'evaluation/prepare_kcuu_llama.py', 'evaluation/validate_kcuu_tp4.py',
    'evaluation/smoke_kcuu_llama.py', 'evaluation/compare_kcuu_logits.py',
    'evaluation/run_kcuu_llama_smoke.py',
    'evaluation/benchmark_kcuu_tp4.py', 'evaluation/run_kcuu_experiments.py',
    'scripts/gh200_tp4_env.sh', 'tests/test_tp4_c1_factors.py', 'tests/test_kcuu_tp4.py',
]


def source_hashes():
    return {name: file_sha256(Path(name)) for name in SOURCE_FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    for name in SOURCE_FILES:
        if name.endswith('.py'):
            ast.parse(Path(name).read_text(), filename=name)
    cfg = json.loads((MODEL / 'config.json').read_text())
    assert (cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_key_value_heads'],
            cfg['num_hidden_layers']) == (4096, 32, 8, 32)
    assert cfg['rope_scaling']['rope_type'] == 'llama3' and cfg['rope_theta'] == 500000
    manifest = load_manifest(FACTORS, file_sha256(FACTORS / 'results.json'))
    assert file_sha256(MODEL / 'config.json') == manifest['fit_config']['model_config_sha256']
    source = manifest['source']
    assert source['repository'] == 'alexz949/BasisServe-CALS'
    assert source['revision'] == '5a30494ada6aeb5806c7dd7a63e574ebfd163063'
    assert source['path'] == 'ICLR-results/llama31-8b-instruct/c1/factor-banks/R64-S6-D0'
    paths = [source['path'] + '/' + manifest['artifacts'][str(i)]['file'] for i in range(32)]
    remote = HfApi().get_paths_info(source['repository'], paths, revision=source['revision'])
    remote_hashes = {f.path: f.lfs.sha256 for f in remote}
    assert set(remote_hashes) == set(paths)
    artifacts = []
    for layer in range(32):
        entry = manifest['artifacts'][str(layer)]
        path = FACTORS / entry['file']
        assert file_sha256(path) == entry['sha256'] == remote_hashes[paths[layer]]
        with safe_open(str(path), framework='pt', device='cpu') as f:
            assert set(f.keys()) == {'value_coordinate_encoders', 'head_output_decoders'}
            shapes = {key: f.get_slice(key).get_shape() for key in f.keys()}
            assert shapes == {'value_coordinate_encoders': [8, 128, 64],
                              'head_output_decoders': [32, 64, 4096]}
            assert all(f.get_slice(key).get_dtype() == 'BF16' for key in f.keys())
        artifacts.append(dict(layer=layer, file=str(path), sha256=entry['sha256'], shapes=shapes))
    index_path = MODEL / 'model.safetensors.index.json'
    weight_map = json.loads(index_path.read_text())['weight_map']
    shards = []
    for name in sorted(set(weight_map.values())):
        path = MODEL / name
        assert path.is_file()
        with safe_open(str(path), framework='pt', device='cpu') as f:
            for key in (key for key, value in weight_map.items() if value == name):
                assert key in f.keys()
                if key.endswith('.self_attn.q_proj.weight'):
                    assert f.get_slice(key).get_shape() == [4096, 4096]
                if key.endswith(('.self_attn.k_proj.weight', '.self_attn.v_proj.weight')):
                    assert f.get_slice(key).get_shape() == [1024, 4096]
        shards.append(dict(file=name, bytes=path.stat().st_size, cache_blob=path.resolve().name,
                           full_weight_sha256_recomputed=False))
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    common = ['--model', str(MODEL), '--factor-dir', str(FACTORS)]
    commands = [[sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
                 '-m', 'evaluation.validate_kcuu_tp4', '--factor-dir', str(FACTORS),
                 '--output', str(root / 'llama_boundary.json')]]
    for mode in ('ag', 'hybrid'):
        commands.append([sys.executable, '-u', '-m', 'evaluation.smoke_kcuu_llama', *common,
                         '--mode', mode, '--output', str(root / f'llama_{mode}_smoke.json')])
    commands.append([sys.executable, '-m', 'evaluation.compare_kcuu_logits',
                     '--ag', str(root / 'llama_ag_smoke.json'),
                     '--hybrid', str(root / 'llama_hybrid_smoke.json'),
                     '--output', str(root / 'llama_logits_comparison.json')])
    report = dict(status='cpu_prepared_gpu_not_run', environment=os.environ.get('CONDA_DEFAULT_ENV'),
                  command=shlex.join([sys.executable, *sys.argv]), model=str(MODEL),
                  syntax_checked=[name for name in SOURCE_FILES if name.endswith('.py')],
                  software={name: version(name) for name in ('torch', 'vllm', 'triton', 'transformers', 'huggingface_hub')},
                  model_revision=MODEL.name, model_config_sha256=file_sha256(MODEL / 'config.json'),
                  model_index_sha256=file_sha256(index_path), model_shards=shards, model_config=cfg,
                  factors_manifest_sha256=file_sha256(FACTORS / 'results.json'), factors_source=source,
                  factors=artifacts, source_sha256=source_hashes(),
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  geometry=dict(local_query_heads=8, local_kv_heads=2, head_dim=128,
                                local_latent_width=512, gathered_width=2048, hidden_width=4096),
                  gpu_smoke_commands=commands,
                  gpu_smoke_requires=dict(nodes=1, gpus_per_node=4, architecture='SM90'),
                  numerical_gates=dict(attention_relative_l2=0.01, output_relative_l2=0.02,
                      logits_relative_l2=0.02, max_absolute_error='recorded, no independent gate',
                      rationale='Reuse existing BF16 attention/output tolerances; logits 0.02 is a provisional smoke gate, not a quality claim.'),
                  formal_configuration_manifest=[dict(model=model, arm=arm, batch=batch,
                      prefill_tokens=8192, output_tokens=128, max_num_batched_tokens=8192,
                      max_num_seqs=batch, max_model_len=8320, gpu_memory_utilization=0.8,
                      repeats=3, warmups=1, status='not_run')
                      for batch in (64, 16, 32, 128, 256)
                      for model in ('qwen3-32b', 'llama31-8b-instruct')
                      for arm in ('dense', 'v64_ag', 'v64_hybrid')])
    (root / 'llama_preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    for command in commands:
        print(shlex.join(command))
    print('CPU PREPARATION PASSED: 32 factor hashes; GPU validation not executed.')


if __name__ == '__main__':
    main()
