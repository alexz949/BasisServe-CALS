"""POD-guided selection of real activation coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class PODResult:
    basis: Tensor
    singular_values: Tensor
    retained_energy_fraction: float
    relative_projection_residual: float
    method: str
    seed: int

    def diagnostics(self) -> dict[str, Any]:
        return {
            "rank": int(self.basis.shape[1]),
            "available_singular_values": int(self.singular_values.numel()),
            "retained_energy_fraction": self.retained_energy_fraction,
            "relative_projection_residual": self.relative_projection_residual,
            "method": self.method,
            "seed": self.seed,
        }


def _validate_width(width: int, selected_width: int) -> None:
    if width <= 0 or not 0 < selected_width <= width:
        raise ValueError("selected width must lie in [1, width]")


def selected_index_sha256(indices: Tensor) -> str:
    values = indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def random_coordinates(width: int, selected_width: int, *, seed: int) -> Tensor:
    _validate_width(width, selected_width)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(width, generator=generator)[:selected_width].sort().values


def activation_rms_coordinates(snapshots: Tensor, selected_width: int) -> Tensor:
    if snapshots.ndim != 2:
        raise ValueError("snapshots must be [tokens, channels]")
    _validate_width(int(snapshots.shape[1]), selected_width)
    scores = snapshots.float().square().mean(dim=0)
    return torch.topk(scores, selected_width, largest=True, sorted=True).indices.sort().values


@torch.no_grad()
def compute_uncentered_pod(
    snapshots: Tensor,
    rank: int,
    *,
    oversample: int,
    niter: int,
    seed: int,
) -> PODResult:
    if snapshots.ndim != 2 or not 0 < rank <= min(snapshots.shape):
        raise ValueError("invalid POD snapshot shape or rank")
    if oversample < 0 or niter < 0:
        raise ValueError("POD oversampling and iterations must be nonnegative")
    maximum = min(map(int, snapshots.shape))
    q = min(maximum, rank + oversample)
    work = snapshots.float()
    devices = [work.device] if work.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if work.is_cuda:
            torch.cuda.manual_seed_all(seed)
        _, singular_values, vectors = torch.pca_lowrank(
            work,
            q=q,
            center=False,
            niter=niter,
        )
    basis = vectors[:, :rank].contiguous()
    total_energy = torch.sum(work.square(), dtype=torch.float64).clamp_min(1e-300)
    retained_energy = torch.sum(
        singular_values[:rank].square(),
        dtype=torch.float64,
    )
    projection = work @ basis
    projection_energy = torch.sum(projection.square(), dtype=torch.float64)
    residual = (total_energy - projection_energy).clamp_min(0.0)
    return PODResult(
        basis=basis,
        singular_values=singular_values,
        retained_energy_fraction=float(retained_energy / total_energy),
        relative_projection_residual=float(residual / total_energy),
        method="torch.pca_lowrank_uncentered",
        seed=int(seed),
    )


def leverage_scores(pod_basis: Tensor) -> Tensor:
    if pod_basis.ndim != 2 or not torch.isfinite(pod_basis).all():
        raise ValueError("POD basis must be a finite matrix")
    return pod_basis.float().square().sum(dim=1)


def leverage_coordinates(pod_basis: Tensor, selected_width: int) -> Tensor:
    _validate_width(int(pod_basis.shape[0]), selected_width)
    indices = torch.topk(
        leverage_scores(pod_basis),
        selected_width,
        largest=True,
        sorted=True,
    ).indices
    return indices.sort().values.cpu()


def qdeim_coordinates(pod_basis: Tensor) -> Tensor:
    """Return the standard k QRCP pivots of ``V_k.T``."""

    if pod_basis.ndim != 2:
        raise ValueError("POD basis must be [channels, rank]")
    channels, rank = map(int, pod_basis.shape)
    if not 0 < rank <= channels:
        raise ValueError("POD basis rank is invalid")
    from scipy.linalg import qr

    work = np.asarray(
        pod_basis.detach().to(device="cpu", dtype=torch.float64).transpose(0, 1).numpy(),
        order="F",
    )
    _, _, pivots = qr(work, mode="economic", pivoting=True, check_finite=False)
    return torch.from_numpy(np.array(pivots[:rank], dtype=np.int64, copy=True)).sort().values


def qdeim_plus_leverage_coordinates(
    pod_basis: Tensor,
    selected_width: int,
) -> Tensor:
    channels, rank = map(int, pod_basis.shape)
    _validate_width(channels, selected_width)
    if selected_width < rank:
        raise ValueError("QDEIM plus leverage requires selected_width >= POD rank")
    qdeim = qdeim_coordinates(pod_basis)
    if selected_width == rank:
        return qdeim
    chosen = torch.zeros(channels, dtype=torch.bool)
    chosen[qdeim] = True
    leverage_order = torch.argsort(leverage_scores(pod_basis), descending=True).cpu()
    additions = leverage_order[~chosen[leverage_order]][: selected_width - rank]
    return torch.cat((qdeim, additions)).sort().values


@torch.no_grad()
def selection_diagnostics(
    indices: Tensor,
    *,
    width: int,
    pod_basis: Tensor | None = None,
    tp_degree: int | None = None,
) -> dict[str, Any]:
    values = indices.detach().to(device="cpu", dtype=torch.long)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("selected indices must be a nonempty vector")
    if int(values.min()) < 0 or int(values.max()) >= width:
        raise ValueError("selected index is out of range")
    if int(torch.unique(values).numel()) != int(values.numel()):
        raise ValueError("selected indices are not unique")
    result: dict[str, Any] = {
        "selected_width": int(values.numel()),
        "source_width": int(width),
        "selected_index_sha256": selected_index_sha256(values),
        "minimum_index": int(values.min()),
        "maximum_index": int(values.max()),
    }
    if pod_basis is not None:
        selected = pod_basis.index_select(0, values.to(pod_basis.device)).double().cpu()
        singular_values = torch.linalg.svdvals(selected)
        sigma_min = float(singular_values.min())
        sigma_max = float(singular_values.max())
        result.update(
            {
                "pod_selected_sigma_min": sigma_min,
                "pod_selected_sigma_max": sigma_max,
                "pod_selected_condition": sigma_max
                / max(sigma_min, torch.finfo(torch.float64).tiny),
            }
        )
    if tp_degree is not None:
        if tp_degree <= 0 or width % tp_degree:
            raise ValueError("TP degree must positively divide the source width")
        shard_width = width // tp_degree
        shard_ids = torch.div(values, shard_width, rounding_mode="floor")
        result["per_tp_rank_selected_counts"] = torch.bincount(
            shard_ids,
            minlength=tp_degree,
        ).tolist()
        result["tp_degree"] = int(tp_degree)
    return result
