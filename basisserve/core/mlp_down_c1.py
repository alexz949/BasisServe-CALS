"""Shared-code C1 compression for row-parallel MLP ``down_proj`` layers.

For a logical row-parallel weight ``W = [W_0, ..., W_{P-1}]`` this module
implements

    z_p = h_p @ E_p.T
    z = AllReduce(z_p)
    y_hat = z @ D.T

with one source-specific encoder shard ``E_p`` and one replicated decoder
``D``.  The offline fit minimizes the complete MLP output error after source
summation.  Because only ``down_proj`` is changed, post-SwiGLU activations are
fixed and the optimum is the leading eigenspace of the teacher-output second
moment ``E[y y.T]``.  No iterative ALS solve or covariance inverse is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.core.tp_output import LowRankAllReduceOutput


FACTOR_FORMAT = "basisserve.mlp_down_c1.shared_allreduce.v1"


def _canonicalize_columns(matrix: Tensor) -> Tensor:
    pivots = matrix.abs().argmax(dim=0)
    columns = torch.arange(matrix.shape[1], device=matrix.device)
    signs = matrix[pivots, columns].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return matrix * signs.unsqueeze(0)


def _relative_projection_error(covariance: Tensor, basis: Tensor) -> float:
    total = torch.trace(covariance).clamp_min(torch.finfo(covariance.dtype).tiny)
    retained = torch.sum(basis * (covariance @ basis))
    return float(((total - retained).clamp_min(0) / total).item())


@dataclass(frozen=True)
class MLPDownC1Factors:
    """One full logical ``down_proj`` shared-code factorization."""

    input_factor: Tensor  # [intermediate_size, rank]
    output_basis: Tensor  # [hidden_size, rank]
    eigenvalues: Tensor  # [rank]
    metrics: Mapping[str, Any]

    @property
    def rank(self) -> int:
        return int(self.output_basis.shape[1])


def _fit_mlp_down_c1_from_input_cholesky(
    input_cholesky: Tensor,
    target_input_cross_moment: Tensor,
    target_second_moment: Tensor,
    *,
    objective_input_cholesky: Tensor | None,
    rank: int,
    factor_dtype: torch.dtype = torch.bfloat16,
    work_dtype: torch.dtype = torch.float32,
    solver: str,
    absolute_damping: float,
    relative_damping: float,
) -> MLPDownC1Factors:
    if (
        input_cholesky.ndim != 2
        or input_cholesky.shape[0] != input_cholesky.shape[1]
    ):
        raise ValueError("input Cholesky factor must be square")
    input_width = int(input_cholesky.shape[0])
    if target_input_cross_moment.ndim != 2 or int(
        target_input_cross_moment.shape[1]
    ) != input_width:
        raise ValueError("target/input cross moment has incompatible input width")
    output_width = int(target_input_cross_moment.shape[0])
    if tuple(target_second_moment.shape) != (output_width, output_width):
        raise ValueError("target second moment has incompatible output width")
    if not 0 < rank <= min(input_width, output_width):
        raise ValueError("sequential C1 rank is outside the regression geometry")
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("work dtype must be float32 or float64")
    if factor_dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError("unsupported factor dtype")
    if absolute_damping < 0 or relative_damping < 0:
        raise ValueError("sequential C1 damping must be nonnegative")

    device = input_cholesky.device
    cholesky = input_cholesky.to(device=device, dtype=work_dtype)
    objective_cholesky = (
        cholesky
        if objective_input_cholesky is None
        else objective_input_cholesky.to(device=device, dtype=work_dtype)
    )
    if tuple(objective_cholesky.shape) != tuple(cholesky.shape):
        raise ValueError("objective and solve input factors use different geometry")
    c_yx = target_input_cross_moment.to(device=device, dtype=work_dtype)
    c_yy = target_second_moment.to(device=device, dtype=work_dtype)
    c_yy = 0.5 * (c_yy + c_yy.T)
    if not all(
        bool(torch.isfinite(value).all())
        for value in (cholesky, objective_cholesky, c_yx, c_yy)
    ):
        raise FloatingPointError("sequential C1 moments must be finite")
    diagonal = cholesky.diagonal().abs()
    if not bool(torch.all(diagonal > 0)):
        raise torch.linalg.LinAlgError("student-input factor is rank deficient")
    whitened_cross_t = torch.linalg.solve_triangular(
        cholesky,
        c_yx.T.contiguous(),
        upper=False,
    )
    predictable_output = whitened_cross_t.T @ whitened_cross_t
    predictable_output = 0.5 * (predictable_output + predictable_output.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(predictable_output)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    retained_eigenvalues = eigenvalues.index_select(0, order).clamp_min(0)
    output_basis = _canonicalize_columns(eigenvectors.index_select(1, order))

    rhs = c_yx.T @ output_basis
    lower_solution = torch.linalg.solve_triangular(cholesky, rhs, upper=False)
    input_factor = torch.linalg.solve_triangular(
        cholesky.T,
        lower_solution,
        upper=True,
    )
    target_energy = torch.trace(c_yy).clamp_min(torch.finfo(work_dtype).tiny)
    regularized_residual = (
        target_energy - retained_eigenvalues.sum()
    ).clamp_min(0)
    encoded_input = objective_cholesky.T @ input_factor
    latent_covariance = encoded_input.T @ encoded_input
    decoder_gram = output_basis.T @ output_basis
    prediction_energy = torch.sum(latent_covariance * decoder_gram)
    target_prediction_cross = torch.sum(
        output_basis * (c_yx @ input_factor)
    )
    output_residual = (
        target_energy - 2.0 * target_prediction_cross + prediction_energy
    ).clamp_min(0)
    fit_error = float((output_residual / target_energy).item())
    regularized_error = float((regularized_residual / target_energy).item())
    return MLPDownC1Factors(
        input_factor=input_factor.to(dtype=factor_dtype).cpu().contiguous(),
        output_basis=output_basis.to(dtype=factor_dtype).cpu().contiguous(),
        eigenvalues=retained_eigenvalues.float().cpu().contiguous(),
        metrics={
            "algorithm": (
                "sequential_exact_reduced_rank_regression"
                if absolute_damping == 0
                else "sequential_ridge_reduced_rank_regression"
            ),
            "objective": "dense_teacher_mlp_output_from_compressed_student_input",
            "collective": "latent_allreduce",
            "input_width": input_width,
            "output_width": output_width,
            "rank": rank,
            "fit_relative_output_mse": fit_error,
            "fit_relative_regularized_objective": regularized_error,
            "communication_fraction_of_dense_allreduce": rank / output_width,
            "communication_reduction_fraction": 1.0 - rank / output_width,
            "factor_parameter_fraction_of_dense_down_proj": (
                rank * (input_width + output_width) / (input_width * output_width)
            ),
            "covariance_damping": absolute_damping,
            "relative_covariance_damping": relative_damping,
            "solver": solver,
        },
    )


@torch.no_grad()
def fit_mlp_down_c1_cross_moments(
    input_second_moment: Tensor,
    target_input_cross_moment: Tensor,
    target_second_moment: Tensor,
    *,
    rank: int,
    factor_dtype: torch.dtype = torch.bfloat16,
    work_dtype: torch.dtype = torch.float32,
) -> MLPDownC1Factors:
    """Solve sequential closure from ``E[X.T X]`` without damping."""

    if (
        input_second_moment.ndim != 2
        or input_second_moment.shape[0] != input_second_moment.shape[1]
    ):
        raise ValueError("input second moment must be square")
    c_xx = input_second_moment.to(dtype=work_dtype)
    c_xx = 0.5 * (c_xx + c_xx.T)
    cholesky, info = torch.linalg.cholesky_ex(c_xx, check_errors=False)
    if int(info.max().item()) != 0:
        raise torch.linalg.LinAlgError(
            "student-input second moment is not positive definite at zero damping"
        )
    return _fit_mlp_down_c1_from_input_cholesky(
        cholesky,
        target_input_cross_moment,
        target_second_moment,
        objective_input_cholesky=cholesky,
        rank=rank,
        factor_dtype=factor_dtype,
        work_dtype=work_dtype,
        solver="zero_damping_normal_equation_cholesky_reduced_rank_regression",
        absolute_damping=0.0,
        relative_damping=0.0,
    )


@torch.no_grad()
def fit_mlp_down_c1_cross_moments_from_cholesky(
    input_cholesky: Tensor,
    target_input_cross_moment: Tensor,
    target_second_moment: Tensor,
    *,
    objective_input_cholesky: Tensor | None = None,
    absolute_damping: float = 0.0,
    relative_damping: float = 0.0,
    rank: int,
    factor_dtype: torch.dtype = torch.bfloat16,
    work_dtype: torch.dtype = torch.float32,
) -> MLPDownC1Factors:
    """Solve sequential closure from a direct ``X = Q R`` input factor.

    ``input_cholesky`` is the solve factor.  For ridge regression it includes
    the augmented identity rows, while ``objective_input_cholesky`` is the raw
    ``R.T / sqrt(N)`` factor used to report unregularized output MSE.
    """

    return _fit_mlp_down_c1_from_input_cholesky(
        input_cholesky,
        target_input_cross_moment,
        target_second_moment,
        objective_input_cholesky=objective_input_cholesky,
        rank=rank,
        factor_dtype=factor_dtype,
        work_dtype=work_dtype,
        solver=(
            "zero_damping_streaming_tsqr_reduced_rank_regression"
            if absolute_damping == 0
            else "ridge_augmented_streaming_tsqr_reduced_rank_regression"
        ),
        absolute_damping=absolute_damping,
        relative_damping=relative_damping,
    )


@torch.no_grad()
def fit_mlp_down_c1_fixed_encoder_decoder(
    input_factor: Tensor,
    latent_input_cholesky: Tensor,
    target_latent_cross_moment: Tensor,
    target_second_moment: Tensor,
    *,
    factor_dtype: torch.dtype = torch.bfloat16,
    work_dtype: torch.dtype = torch.float32,
) -> MLPDownC1Factors:
    """Refit only the decoder for a fixed activation-aware encoder.

    If ``Z = X @ input_factor``, this solves the zero-damping least-squares
    problem ``min_D E[||Y - Z @ D.T||^2]``.  The lower-triangular input factor
    must satisfy ``C_zz = L @ L.T`` and should come directly from TSQR.
    """

    if input_factor.ndim != 2:
        raise ValueError("fixed MLP encoder must be a matrix")
    input_width, rank = map(int, input_factor.shape)
    if tuple(latent_input_cholesky.shape) != (rank, rank):
        raise ValueError("latent TSQR factor does not match fixed encoder rank")
    if target_latent_cross_moment.ndim != 2 or int(
        target_latent_cross_moment.shape[1]
    ) != rank:
        raise ValueError("target/latent cross moment has incompatible rank")
    output_width = int(target_latent_cross_moment.shape[0])
    if tuple(target_second_moment.shape) != (output_width, output_width):
        raise ValueError("target second moment has incompatible output width")
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("work dtype must be float32 or float64")
    if factor_dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError("unsupported factor dtype")

    device = latent_input_cholesky.device
    cholesky = latent_input_cholesky.to(device=device, dtype=work_dtype)
    c_yz = target_latent_cross_moment.to(device=device, dtype=work_dtype)
    c_yy = target_second_moment.to(device=device, dtype=work_dtype)
    c_yy = 0.5 * (c_yy + c_yy.T)
    if not all(
        bool(torch.isfinite(value).all())
        for value in (input_factor, cholesky, c_yz, c_yy)
    ):
        raise FloatingPointError("fixed-encoder sequential moments must be finite")
    if not bool(torch.all(cholesky.diagonal().abs() > 0)):
        raise torch.linalg.LinAlgError("fixed-encoder latent matrix is rank deficient")

    lower_solution = torch.linalg.solve_triangular(
        cholesky,
        c_yz.T.contiguous(),
        upper=False,
    )
    decoder_coefficient = torch.linalg.solve_triangular(
        cholesky.T,
        lower_solution,
        upper=True,
    )
    output_basis = decoder_coefficient.T.contiguous()
    target_energy = torch.trace(c_yy).clamp_min(torch.finfo(work_dtype).tiny)
    predicted_whitened = cholesky.T @ decoder_coefficient
    prediction_energy = torch.sum(predicted_whitened * predicted_whitened)
    target_prediction_cross = torch.sum(output_basis * c_yz)
    residual = (
        target_energy - 2.0 * target_prediction_cross + prediction_energy
    ).clamp_min(0)
    fit_error = float((residual / target_energy).item())
    return MLPDownC1Factors(
        input_factor=input_factor.to(dtype=factor_dtype).cpu().contiguous(),
        output_basis=output_basis.to(dtype=factor_dtype).cpu().contiguous(),
        eigenvalues=torch.empty(0, dtype=torch.float32),
        metrics={
            "algorithm": "sequential_fixed_encoder_decoder_least_squares",
            "objective": "dense_teacher_mlp_output_from_fixed_student_latent",
            "collective": "latent_allreduce",
            "input_width": input_width,
            "output_width": output_width,
            "rank": rank,
            "fit_relative_output_mse": fit_error,
            "fit_relative_regularized_objective": fit_error,
            "communication_fraction_of_dense_allreduce": rank / output_width,
            "communication_reduction_fraction": 1.0 - rank / output_width,
            "factor_parameter_fraction_of_dense_down_proj": (
                rank * (input_width + output_width) / (input_width * output_width)
            ),
            "covariance_damping": 0.0,
            "relative_covariance_damping": 0.0,
            "solver": "zero_damping_latent_streaming_tsqr_least_squares",
        },
    )


@torch.no_grad()
def fit_mlp_down_c1(
    weight: Tensor,
    fit_output_second_moment: Tensor,
    heldout_output_second_moment: Tensor,
    *,
    rank: int,
    factor_dtype: torch.dtype = torch.bfloat16,
    work_dtype: torch.dtype = torch.float32,
) -> MLPDownC1Factors:
    """Fit the exact shared-decoder optimum for complete MLP output MSE.

    ``weight`` is the logical PyTorch ``down_proj`` weight with shape
    ``[hidden_size, intermediate_size]``.  The two moments are normalized
    teacher-output second moments with shape ``[hidden_size, hidden_size]``.
    """

    if weight.ndim != 2:
        raise ValueError("down_proj weight must be a matrix")
    hidden_size, intermediate_size = map(int, weight.shape)
    expected = (hidden_size, hidden_size)
    if tuple(fit_output_second_moment.shape) != expected or tuple(
        heldout_output_second_moment.shape
    ) != expected:
        raise ValueError("MLP output moments do not match down_proj output width")
    if not 0 < rank <= hidden_size:
        raise ValueError(f"rank must be in [1, {hidden_size}], got {rank}")
    if work_dtype not in (torch.float32, torch.float64):
        raise ValueError("work dtype must be float32 or float64")
    if factor_dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError("unsupported factor dtype")

    device = weight.device
    fit_covariance = fit_output_second_moment.to(device=device, dtype=work_dtype)
    fit_covariance = 0.5 * (fit_covariance + fit_covariance.T)
    heldout_covariance = heldout_output_second_moment.to(
        device=device, dtype=work_dtype
    )
    heldout_covariance = 0.5 * (heldout_covariance + heldout_covariance.T)
    work_weight = weight.detach().to(device=device, dtype=work_dtype)
    if not all(
        bool(torch.isfinite(value).all())
        for value in (fit_covariance, heldout_covariance, work_weight)
    ):
        raise FloatingPointError("MLP weight and output moments must be finite")

    eigenvalues, eigenvectors = torch.linalg.eigh(fit_covariance)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    retained_eigenvalues = eigenvalues.index_select(0, order).clamp_min(0)
    output_basis = _canonicalize_columns(eigenvectors.index_select(1, order))

    # D has orthonormal columns.  For fixed D, E = D.T @ W is the exact
    # least-squares closure for every input, not merely for calibration rows.
    input_factor = work_weight.T @ output_basis
    fit_error = _relative_projection_error(fit_covariance, output_basis)
    heldout_error = _relative_projection_error(heldout_covariance, output_basis)
    stored_input = input_factor.to(dtype=factor_dtype).cpu().contiguous()
    stored_output = output_basis.to(dtype=factor_dtype).cpu().contiguous()

    return MLPDownC1Factors(
        input_factor=stored_input,
        output_basis=stored_output,
        eigenvalues=retained_eigenvalues.float().cpu().contiguous(),
        metrics={
            "algorithm": "teacher_output_pca_exact_shared_decoder",
            "objective": "post_swiglu_complete_mlp_output_mse_after_tp_sum",
            "collective": "latent_allreduce",
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "rank": rank,
            "fit_relative_output_mse": fit_error,
            "heldout_relative_output_mse": heldout_error,
            "communication_fraction_of_dense_allreduce": rank / hidden_size,
            "communication_reduction_fraction": 1.0 - rank / hidden_size,
            "factor_parameter_fraction_of_dense_down_proj": (
                rank * (hidden_size + intermediate_size)
                / (hidden_size * intermediate_size)
            ),
            "damping": 0.0,
            "als_sweeps": 0,
        },
    )


class MLPDownC1Linear(nn.Module):
    """Single-process quality-equivalent shared-code ``down_proj``."""

    def __init__(
        self,
        input_factor: Tensor,
        output_basis: Tensor,
        *,
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        if input_factor.ndim != 2 or output_basis.ndim != 2:
            raise ValueError("MLP C1 factors must be matrices")
        if int(input_factor.shape[1]) != int(output_basis.shape[1]):
            raise ValueError("MLP C1 factors use different ranks")
        self.encoder_weight = nn.Parameter(
            input_factor.detach().T.contiguous(), requires_grad=False
        )
        self.decoder_weight = nn.Parameter(
            output_basis.detach().contiguous(), requires_grad=False
        )
        if bias is None:
            self.register_parameter("bias", None)
        else:
            if tuple(bias.shape) != (int(output_basis.shape[0]),):
                raise ValueError("MLP C1 bias has the wrong shape")
            self.bias = nn.Parameter(bias.detach().contiguous(), requires_grad=False)

    @property
    def in_features(self) -> int:
        return int(self.encoder_weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.decoder_weight.shape[0])

    @property
    def rank(self) -> int:
        return int(self.encoder_weight.shape[0])

    def forward(self, hidden_states: Tensor) -> Tensor:
        latent = F.linear(hidden_states, self.encoder_weight)
        return F.linear(latent, self.decoder_weight, self.bias)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    current = model
    for name in ("model", "language_model"):
        child = getattr(current, name, None)
        if child is not None:
            current = child
    layers = getattr(current, "layers", None)
    if layers is None:
        raise TypeError("could not locate decoder layers")
    return layers


def load_mlp_down_c1_manifest(
    factor_dir: str | Path,
    *,
    model_config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(factor_dir).expanduser().resolve()
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != FACTOR_FORMAT
        or int(manifest.get("schema_version", -1)) != 1
        or manifest.get("status") != "complete"
    ):
        raise ValueError(f"incompatible MLP C1 factor manifest: {path}")
    if model_config_path is not None:
        config_path = Path(model_config_path).expanduser().resolve()
        if _sha256(config_path) != manifest["model"]["config_sha256"]:
            raise ValueError("MLP C1 factors belong to another model config")
    return manifest


def install_mlp_down_c1(
    model: nn.Module,
    factor_dir: str | Path,
    *,
    distributed: bool,
    factor_dtype: torch.dtype | None = None,
) -> list[dict[str, Any]]:
    """Replace every dense MLP ``down_proj`` with its C1 implementation.

    With ``distributed=False`` this installs a two-GEMM quality-equivalent
    module.  With ``distributed=True`` the model must already use row-parallel
    ``down_proj`` weights; each process selects its contiguous encoder shard,
    AllReduces the shared latent, and applies the replicated decoder.
    """

    root = Path(factor_dir).expanduser().resolve()
    model_path = Path(str(getattr(model.config, "_name_or_path", ""))).expanduser()
    config_path = model_path / "config.json"
    manifest = load_mlp_down_c1_manifest(
        root,
        model_config_path=config_path if config_path.is_file() else None,
    )
    layers = _decoder_layers(model)
    artifacts = manifest.get("artifacts", {})
    if set(map(int, artifacts)) != set(range(len(layers))):
        raise ValueError("MLP C1 factors do not cover every decoder layer")
    world_size = dist.get_world_size() if distributed and dist.is_initialized() else 1
    process_rank = dist.get_rank() if distributed and dist.is_initialized() else 0
    if distributed and world_size <= 1:
        raise RuntimeError("distributed MLP C1 requires an initialized process group")
    expected_tp = int(manifest["fit_config"]["tp_size"])
    if distributed and world_size != expected_tp:
        raise ValueError(f"MLP C1 factors require TP{expected_tp}, got TP{world_size}")

    records: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        artifact = artifacts[str(layer_index)]
        path = root / artifact["file"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"MLP C1 artifact hash mismatch at layer {layer_index}")
        tensors = load_file(str(path), device="cpu")
        if set(tensors) != {"input_factor", "output_basis"}:
            raise ValueError(f"unexpected MLP C1 tensors at layer {layer_index}")
        input_factor = tensors["input_factor"]
        output_basis = tensors["output_basis"]
        original = layer.mlp.down_proj
        weight = getattr(original, "weight", None)
        if not isinstance(weight, Tensor) or weight.ndim != 2:
            raise TypeError(f"layer {layer_index} has no linear MLP down projection")
        selected_dtype = factor_dtype or weight.dtype
        output_basis = output_basis.to(device=weight.device, dtype=selected_dtype)
        bias = getattr(original, "bias", None)
        if distributed:
            if int(input_factor.shape[0]) % world_size:
                raise ValueError("MLP encoder input width is not divisible by TP")
            local_width = int(input_factor.shape[0]) // world_size
            if int(weight.shape[1]) != local_width:
                raise ValueError(
                    f"layer {layer_index} TP-local down width {weight.shape[1]} "
                    f"does not match factor shard {local_width}"
                )
            start = process_rank * local_width
            local_input = input_factor[start : start + local_width].to(
                device=weight.device, dtype=selected_dtype
            )
            replacement: nn.Module = LowRankAllReduceOutput(
                local_input,
                output_basis,
                bias=bias,
            )
            runtime = "source_encoder_latent_allreduce_shared_decoder"
        else:
            if int(weight.shape[1]) != int(input_factor.shape[0]):
                raise ValueError(f"layer {layer_index} down_proj input width mismatch")
            replacement = MLPDownC1Linear(
                input_factor.to(device=weight.device, dtype=selected_dtype),
                output_basis,
                bias=bias,
            )
            runtime = "single_process_quality_equivalent_two_gemm"
        replacement.eval()
        layer.mlp.down_proj = replacement
        records.append(
            {
                "layer": layer_index,
                "rank": int(output_basis.shape[1]),
                "factor_file": artifact["file"],
                "factor_sha256": artifact["sha256"],
                "runtime": runtime,
            }
        )
    return records


__all__ = [
    "FACTOR_FORMAT",
    "MLPDownC1Factors",
    "MLPDownC1Linear",
    "fit_mlp_down_c1",
    "fit_mlp_down_c1_cross_moments",
    "fit_mlp_down_c1_cross_moments_from_cholesky",
    "fit_mlp_down_c1_fixed_encoder_decoder",
    "install_mlp_down_c1",
    "load_mlp_down_c1_manifest",
]
