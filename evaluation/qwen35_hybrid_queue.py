"""Continue V-only bank assembly and PPL as independent factor workers finish.

The queue stops after all six V-only evaluations so their quality can be
reviewed before starting the authorized conditional Wo stage.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from evaluation.qwen35_hybrid_common import atomic_save, load_bank
from evaluation.qwen35_hybrid_banks import LAYERS, RANKS


ROOT = Path('results/q35_hybrid')


def gpu():
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
    available = [(int(free), int(index)) for index, free in (line.split(',') for line in lines)]
    memory, index = max(available)
    return index if memory >= 28 * 1024 else None


def launch(module, args, name, *, use_gpu=True, device=None):
    selected = (gpu() if device is None else device) if use_gpu else None
    while use_gpu and selected is None:
        print(json.dumps({'waiting_for_memory': name}), flush=True)
        time.sleep(30)
        selected = gpu()
    env = os.environ.copy()
    if use_gpu:
        env['CUDA_VISIBLE_DEVICES'] = str(selected)
    command = [sys.executable, '-u', '-m', module, *args]
    print(json.dumps({'launch': name, 'gpu': selected, 'command': command}), flush=True)
    log = ROOT / 'logs' / f'{name}.log'
    with log.open('a') as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        print(json.dumps({'name': name, 'pid': process.pid, 'log': str(log)}), flush=True)
        status = process.wait()
    assert status == 0, f'{name} failed with status {status}: {log}'


def wait_factors(ranks, workers):
    needed = [ROOT / 'factors' / f'l{layer:02d}_r{rank:03d}.pt' for layer in LAYERS for rank in ranks if rank != 256]
    while not all(p.exists() for p in needed):
        missing = [p.name for p in needed if not p.exists()]
        live = []
        for pid in workers:
            cmdline = Path(f'/proc/{pid}/cmdline')
            if cmdline.exists() and b'evaluation.run_qwen35_hybrid' in cmdline.read_bytes():
                live.append(pid)
        assert live, f'All factor workers stopped; missing {missing}'
        print(json.dumps({'waiting_for_factors': missing, 'live_worker_pids': live}), flush=True)
        time.sleep(30)


def evaluate(bank, name, device=None):
    path = ROOT / f'{name}_ppl.json'
    if path.exists():
        result = json.loads(path.read_text())
        assert len(result['datasets']) == 2
        return
    launch('evaluation.run_qwen35_hybrid', ['evaluate', '--bank', str(bank), '--output', str(path)], name + '_ppl', device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker-pids', required=True)
    p.add_argument('--profile-gpus', type=int, default=8)
    args = p.parse_args()
    workers = list(map(int, args.worker_pids.split(',')))
    assert args.profile_gpus > 0
    for rank in (64, 80, 96):
        wait_factors([rank], workers)
        bank = ROOT / 'banks' / f'c1_uniform_v{rank}.pt'
        if not bank.exists():
            launch('evaluation.qwen35_hybrid_banks', ['assemble', '--uniform', '--anchor', str(rank),
                '--output', str(ROOT / 'banks')], f'assemble_uniform_v{rank}', use_gpu=False)
        load_bank(bank)
        evaluate(bank, f'c1_uniform_v{rank}')
    # Profile only after all candidate banks have frozen, so every assembled
    # checkpoint can be validated against the same immutable candidate set.
    wait_factors(RANKS, workers)
    def profile_rank(rank, device):
        bank = ROOT / 'banks' / f'c1_twosided_v{rank}.pt'
        if not bank.exists():
            launch('evaluation.qwen35_hybrid_banks', ['assemble', '--anchor', str(rank),
                '--output', str(ROOT / 'banks')], f'assemble_twosided_v{rank}', use_gpu=False)
        load_bank(bank)
        evaluate(bank, f'c1_twosided_v{rank}', device=device)
        if not (ROOT / 'kl' / f'confirm_v{rank}.json').exists():
            launch('evaluation.qwen35_hybrid_banks', ['confirm', '--anchor', str(rank),
                '--output', str(ROOT / 'kl')], f'confirm_v{rank}', device=device)
    devices = []
    while not devices:
        lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
        candidates = sorted([(int(free), int(index)) for index, free in (line.split(',') for line in lines)], reverse=True)
        devices = [index for free, index in candidates if free >= 28 * 1024][:args.profile_gpus]
        if not devices:
            time.sleep(30)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(launch, 'evaluation.qwen35_hybrid_banks',
            ['profile', '--all-anchors', '--shard-index', str(index), '--num-shards', str(len(devices)),
             '--output', str(ROOT / 'kl')], f'kl_shard{index}of{len(devices)}', device=device)
            for index, device in enumerate(devices)]
        for future in futures:
            future.result()
    # One lane owns each GPU, including when fewer than three are available.
    def finish_device(device, ranks):
        for rank in ranks:
            profile_rank(rank, device)
    finish_devices = devices[:3]
    with ThreadPoolExecutor(max_workers=len(finish_devices)) as pool:
        futures = [pool.submit(finish_device, device, (64, 80, 96)[index::len(finish_devices)])
            for index, device in enumerate(finish_devices)]
        for future in futures:
            future.result()
    atomic_save(ROOT / 'v_only_queue_complete.json', {'status': 'six_v_only_evaluations_complete',
        'next': 'inspect V-only quality and then run six frozen-V Wo stages'})


if __name__ == '__main__':
    main()
