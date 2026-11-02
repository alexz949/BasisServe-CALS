"""Inspect the failed layer-zero reduced normal equations without writing factors."""

import json
from pathlib import Path

import torch

from basisserve.compression import ValueGeometry
from basisserve.compression import value as fitter
from basisserve.core import gqa_routed_ov_joint as joint


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    root = Path('results/llama31i_v96_64k/covariance')
    geometry = ValueGeometry.from_snapshot(root)
    manifest = json.loads((root/'manifest.json').read_text())
    matrix, _, weight, _ = fitter._load_covariance_layer(root,manifest,0, geometry=geometry)
    covariance = fitter._covariance_matrix_to_blocks(matrix,device='cuda',dtype=torch.float32, geometry=geometry)
    covariance, damping = joint.covariance_with_trace_damping(covariance,relative_damping=1e-7)
    target = fitter._dense_head_targets(weight,device='cuda',dtype=torch.float32, geometry=geometry)
    mapping = fitter._head_to_kv_group(device='cuda', geometry=geometry)
    objective = joint.quadratic_from_target(covariance=covariance,target=target,
        name='layer0_diagnostic',trace_normalize=False)
    initialization = joint.initialize_group_pooled_routed_svd(covariance=covariance,
        target=target,head_to_kv_group=mapping,group_ranks=(96,)*8,covariance_ridge=1e-7)
    encoder = initialization.A_unique
    hessian, rhs = joint._reduced_normal_equations(objective,encoder,mapping)
    hessian = hessian.permute(0,2,1,3).reshape(3072,3072)
    rhs = rhs.reshape(3072,4096)
    print(json.dumps(dict(covariance_damping=damping,
        covariance_finite=bool(torch.isfinite(covariance).all()),
        encoder_finite=bool(torch.isfinite(encoder).all()),
        hessian_finite=bool(torch.isfinite(hessian).all()),
        rhs_finite=bool(torch.isfinite(rhs).all()),
        hessian_absmax=float(hessian.abs().max()),rhs_absmax=float(rhs.abs().max()))),flush=True)
    for dtype in (torch.float32,torch.float64):
        symmetric = .5*(hessian.to(dtype)+hessian.T.to(dtype))
        scale = symmetric.diag().abs().mean()
        identity = torch.eye(3072,device='cuda',dtype=dtype)
        for relative in (0.,1e-12,1e-10,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2):
            system = symmetric+relative*scale*identity
            chol,info = torch.linalg.cholesky_ex(system,check_errors=False)
            record = dict(dtype=str(dtype),relative_jitter=relative,info=int(info),
                chol_finite=bool(torch.isfinite(chol).all()))
            if int(info) == 0:
                solution = torch.cholesky_solve(rhs.to(dtype),chol)
                record.update(solution_finite=bool(torch.isfinite(solution).all()),
                    solution_absmax=float(solution.abs().max()),
                    relative_residual=float(torch.linalg.vector_norm(system@solution-rhs.to(dtype))/torch.linalg.vector_norm(rhs.to(dtype))))
            print(json.dumps(record),flush=True)
    eigenvalues = torch.linalg.eigvalsh(symmetric.cpu())
    print(json.dumps(dict(cpu_eigenvalue_min=float(eigenvalues.min()),
        cpu_eigenvalue_max=float(eigenvalues.max()),diagonal_scale=float(scale))),flush=True)


if __name__ == '__main__':
    main()
