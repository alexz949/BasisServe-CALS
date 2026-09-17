"""Explicit numerical settings for reproducible evaluation runs."""

import os

import torch

from evaluation.v96kl_common import configure


NUMERICAL_POLICY = dict(
    model_dtype='bfloat16', float32_matmul_precision='highest',
    cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
    deterministic_algorithms=True, cublas_workspace_config=':4096:8',
)


def configure_deterministic_evaluation():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = NUMERICAL_POLICY['cublas_workspace_config']
    configure()
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
