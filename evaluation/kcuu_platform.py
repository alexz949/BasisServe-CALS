"""Explicit hardware and execution identity; never fabricate Slurm metadata."""
import os
from pathlib import Path
import socket

PLATFORM = os.environ.get('KCUU_PLATFORM', 'gh200')
assert PLATFORM in ('a100', 'gh200')
A100 = PLATFORM == 'a100'
HUB = Path('/mnt/basis-data/hf')
LLAMA_MODEL = HUB / 'models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
QWEN_MODEL = HUB / 'models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137'
LLAMA_FACTORS = Path('/mnt/basis-data/factors/llama31-8b-instruct/R64-S6-D0')
QWEN_FACTORS = HUB / 'models--alexz949--BasisServe-CALS/snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-32b/c1/factor-banks/R64-S6'


def check_execution():
    assert os.environ.get('CONDA_DEFAULT_ENV') == 'basis'
    if A100:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1,2,3,4'
    else:
        assert os.environ.get('SLURM_NNODES') == '1'


def check_devices():
    import torch
    check_execution()
    assert torch.cuda.device_count() == 4
    expected = (8, 0) if A100 else (9, 0)
    assert all(torch.cuda.get_device_capability(i) == expected for i in range(4))
    assert all(i == j or torch.cuda.can_device_access_peer(i, j)
               for i in range(4) for j in range(4))


def identity():
    return dict(platform=PLATFORM, execution='local' if A100 else 'slurm',
                job=os.environ.get('SLURM_JOB_ID'), hostname=socket.gethostname(),
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                run_id=os.environ.get('KCUU_RUN_ID'))
