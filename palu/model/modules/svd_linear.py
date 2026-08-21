import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant import Quantizer
import math

from basisserve.core.strong_rrqr import activation_aware_strong_rrqr_factors

try:
    from .hadamard_utils import apply_hadamard
    _HADAMARD_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    apply_hadamard = None
    _HADAMARD_IMPORT_ERROR = exc


def _resolve_recovery_dtype(dtype_value, model_dtype=torch.float32):
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    text = str(dtype_value or "float32").strip().lower()
    if text in {"model", "auto"}:
        return model_dtype
    if text in {"float32", "fp32"}:
        return torch.float32
    if text in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if text in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"Unsupported latent recovery dtype: {dtype_value!r}")


def _resolve_factorization_work_dtype(dtype_value):
    if isinstance(dtype_value, torch.dtype):
        dtype = dtype_value
    else:
        text = str(dtype_value or "float32").strip().lower()
        if text in {"float32", "fp32"}:
            dtype = torch.float32
        elif text in {"float64", "fp64", "double"}:
            dtype = torch.float64
        else:
            raise ValueError(f"Unsupported factorization work dtype: {dtype_value!r}")
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError(f"Unsupported factorization work dtype: {dtype}")
    return dtype


def _resolve_factorization_work_device(device_value, model_device):
    text = str(device_value or "model").strip().lower()
    if text == "model":
        return torch.device(model_device)
    if text == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported factorization work device: {device_value!r}")


def _per_head_whiten_decomposition_from_weight(
    weight,
    scaling_diag_matrix,
    rank,
    *,
    scaling_matrix_inv=None,
):
    original_dtype = weight.dtype
    original_device = weight.device

    if scaling_matrix_inv is None:
        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)

    # Multiply scaling_diag_matrix to weight matrix
    W_scale = torch.matmul(
        weight.detach().to(
            device=scaling_diag_matrix.device,
            dtype=scaling_diag_matrix.dtype,
        ),
        scaling_diag_matrix,
    )

    U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)

    V = torch.matmul(Vt, scaling_matrix_inv)

    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]

    sqrtSigma = torch.sqrt(torch.diag(S))

    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(device=original_device, dtype=original_dtype)
    R = torch.matmul(sqrtSigma, V).to(device=original_device, dtype=original_dtype)

    return L, R

def _per_head_decomposition_from_weight(weight, rank):
    original_dtype = weight.dtype
    original_device = weight.device
    # Get weight matrix decomposed
    U, S, Vt = torch.linalg.svd(weight.detach().to(device="cpu", dtype=torch.float32), full_matrices=False)

    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]

    sqrtSigma = torch.sqrt(torch.diag(S))
    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(device=original_device, dtype=original_dtype)
    R = torch.matmul(sqrtSigma, Vt).to(device=original_device, dtype=original_dtype)
    return L, R

def _stacked_whiten_decomposition_from_weight(weight, scaling_diag_matrix, rank):
    original_dtype = weight.dtype
    try:
        scaling_diag_matrix = scaling_diag_matrix.to(weight.device)
    except AttributeError:
        raise FileExistsError("Cache may not be loaded correctly")

    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))
    weight_scale = torch.matmul(weight.to(torch.float32), scaling_diag_matrix.to(torch.float32))

    U, S, Vt = torch.linalg.svd(weight_scale, full_matrices=False)
    V = torch.matmul(Vt, scaling_matrix_inv)

    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]

    sqrtSigma = torch.sqrt(torch.diag(S))
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, V).to(original_dtype)
    return L, R

def _stacked_decomposition_from_weight(weight, rank):
    original_dtype = weight.dtype
    original_device = weight.device
    U, S, Vt = torch.linalg.svd(weight.detach().to(device="cpu", dtype=torch.float32), full_matrices=False)

    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]

    sqrtSigma = torch.sqrt(torch.diag(S))
    L = torch.matmul(U, sqrtSigma).to(device=original_device, dtype=original_dtype)
    R = torch.matmul(sqrtSigma, Vt).to(device=original_device, dtype=original_dtype)
    return L, R

class HeadwiseLowRankModule(nn.Module):
    """ Headwise low rank module """

    def __init__(
        self,
        ranks,
        in_features,
        out_features,
        bias,
        shared_basis_rank=0,
        latent_recovery_ranks=None,
        latent_recovery_use_diag: bool = False,
        latent_recovery_dtype=torch.float32,
    ):
        super().__init__()


        self.ranks = ranks
        self.num_groups = len(ranks)
        self.in_features = in_features
        self.out_features = out_features
        self.group_dim = out_features // self.num_groups
        self.shared_basis_rank = shared_basis_rank
        self.latent_recovery_ranks = latent_recovery_ranks or [0] * self.num_groups
        self.latent_recovery_use_diag = latent_recovery_use_diag
        self.latent_recovery_dtype = _resolve_recovery_dtype(latent_recovery_dtype)
        self.factorization_method = None
        self.factorization_diagnostics = []

        if (self.group_dim * self.num_groups) != self.out_features:
            raise ValueError(
                f"out_features must be divisible by num_groups (got `out_features`: {self.out_features}"
                f" and `num_groups`: {self.num_groups})."
            )

        self.VT = nn.Linear(in_features, sum(ranks), bias=False)

        Us = []
        for r in ranks:
            Us.append(nn.Linear(r, self.group_dim, bias=bias))

        self.U = nn.ModuleList(Us)
        self.shared_VT = None
        self.shared_U = None
        if self.shared_basis_rank > 0:
            self.shared_VT = nn.Linear(in_features, self.shared_basis_rank, bias=False)
            self.shared_U = nn.ModuleList(
                [nn.Linear(self.shared_basis_rank, self.group_dim, bias=False) for _ in range(self.num_groups)]
            )
        self.latent_recovery_A = nn.ModuleDict()
        self.latent_recovery_B = nn.ModuleDict()
        self.latent_recovery_diag = nn.ParameterDict()
        for i, recovery_rank in enumerate(self.latent_recovery_ranks):
            if recovery_rank > 0:
                # Match FlashSVDTrain ActLoRA semantics:
                # z = VT(x), z <- z + up(down(z)), y = U(z)
                self.latent_recovery_A[str(i)] = nn.Linear(
                    self.ranks[i],
                    recovery_rank,
                    bias=False,
                    device=self.VT.weight.device,
                    dtype=self.latent_recovery_dtype,
                )
                self.latent_recovery_B[str(i)] = nn.Linear(
                    recovery_rank,
                    self.ranks[i],
                    bias=False,
                    device=self.VT.weight.device,
                    dtype=self.latent_recovery_dtype,
                )
                nn.init.normal_(self.latent_recovery_A[str(i)].weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.latent_recovery_B[str(i)].weight)
                if self.latent_recovery_use_diag:
                    self.latent_recovery_diag[str(i)] = nn.Parameter(
                        torch.zeros(
                            self.ranks[i],
                            device=self.VT.weight.device,
                            dtype=self.latent_recovery_dtype,
                        )
                    )


        self.quantized_latents = False
        self.latent_quantizer = None

    def forward(self,
                hidden_states: torch.Tensor):
        low_rank_latents = self.project_to_latent(hidden_states)
        if self.quantized_latents:
            low_rank_latents = self.quantize_latent(low_rank_latents)
        outputs = self.reconstruct(low_rank_latents, hidden_states)
        return outputs


    def project_to_latent(self, hidden_states:  torch.Tensor):
        """
            hidden_states: Tensor of shape (batch_size, seq_len, in_features)
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                "Input tensor should have dimension 3."
            )
        hidden_states = self.VT(hidden_states)
        """
            hidden_states: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        """
        return hidden_states

    def apply_latent_recovery(self, low_rank_latents: torch.Tensor) -> torch.Tensor:
        """Apply latent-space recovery without reconstructing to full output."""
        recovered_latents = []
        total_ranks = 0
        for i in range(self.num_groups):
            low_rank_latent = low_rank_latents[:, :, total_ranks: total_ranks+self.ranks[i]]
            if self.latent_recovery_ranks[i] > 0:
                recovery_input = low_rank_latent.to(self.latent_recovery_A[str(i)].weight.dtype)
                recovery_delta = F.linear(
                    F.linear(recovery_input, self.latent_recovery_A[str(i)].weight),
                    self.latent_recovery_B[str(i)].weight,
                )
                if self.latent_recovery_use_diag:
                    diag_scale = self.latent_recovery_diag[str(i)].view(1, 1, -1)
                    recovery_delta = recovery_delta + (recovery_input * diag_scale)
                low_rank_latent = low_rank_latent + recovery_delta.to(low_rank_latent.dtype)
            recovered_latents.append(low_rank_latent)
            total_ranks += self.ranks[i]
        return torch.cat(recovered_latents, dim=-1)

    def project_to_recovered_latent(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project to latent space and apply recovery before caching."""
        return self.apply_latent_recovery(self.project_to_latent(hidden_states))

    def reconstruct(self, low_rank_latents: torch.Tensor, hidden_states: torch.Tensor = None):
        """
            low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        """
        low_rank_latents = self.apply_latent_recovery(low_rank_latents)
        outputs = []
        total_ranks = 0
        shared_latents = None
        if self.shared_basis_rank > 0:
            if hidden_states is None:
                raise ValueError("hidden_states are required when shared_basis_rank > 0")
            shared_latents = self.shared_VT(hidden_states.to(self.shared_VT.weight.dtype))
        for i in range(self.num_groups):
            low_rank_latent = low_rank_latents[:, :, total_ranks: total_ranks+self.ranks[i]]
            output = self.U[i](low_rank_latent)
            if shared_latents is not None:
                shared_output = self.shared_U[i](shared_latents)
                output = output + shared_output.to(output.dtype)
            outputs.append(output)
            total_ranks += self.ranks[i]

        """
            outputs: Tensor of shape (batch_size, seq_len, out_features)
        """
        return torch.cat(outputs, dim=-1)


    def quantize_latent(self, low_rank_latents: torch.Tensor):
        """
            low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        """
        assert self.latent_quantizer is not None, "Latent quantizer is not initialized."
        fake_quantized_low_rank_latents = []
        total_ranks = 0
        for i in range(self.num_groups):
            low_rank_latent = low_rank_latents[:, :, total_ranks: total_ranks+self.ranks[i]]
            fake_quantized_low_rank_latents.append(self.latent_quantizer(low_rank_latent))
            total_ranks += self.ranks[i]

        """
            fake_quantized_low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ...)
        """
        return torch.cat(fake_quantized_low_rank_latents, dim=-1)


    def configure_latent_quantizer(self,
        n_bits: int,
        group_size: int,
        sym: bool,
        clip_ratio: float,
        hadamard = False
    ):
        #self.latent_quantizer = Quantizer(n_bits, group_size, sym, clip_ratio, hadamard)
        self.latent_quantizer = Quantizer(n_bits, group_size, sym, clip_ratio)
        if hadamard:
            self.fused_hadamard_matrix()
        self.quantized_latents = True


    def fused_hadamard_matrix(self):
        if apply_hadamard is None:
            raise ModuleNotFoundError(
                "fast_hadamard_transform is required when hadamard quantization is enabled."
            ) from _HADAMARD_IMPORT_ERROR
        total_ranks = 0
        for i in range(self.num_groups):
            # Apply Q to VT
            VT_weight_i = self.VT.weight.data[total_ranks: total_ranks+self.ranks[i], :]
            VT_weight_i = apply_hadamard(VT_weight_i.t())
            self.VT.weight.data[total_ranks: total_ranks+self.ranks[i], :] = VT_weight_i.t()
            # Apply Q^T to U
            U_weight_i = self.U[i].weight.data
            U_weight_i = apply_hadamard(U_weight_i)
            self.U[i].weight.data = U_weight_i

            total_ranks += self.ranks[i]

    @staticmethod
    def _compute_budget_preserved_ranks(ranks, in_features, out_features, shared_basis_rank):
        if shared_basis_rank <= 0 or len(ranks) <= 1:
            return list(ranks)
        group_dim = out_features // len(ranks)
        saved_per_group_rank = len(ranks) * (in_features + group_dim)
        shared_basis_cost = in_features + out_features
        rank_delta = math.ceil(shared_basis_rank * shared_basis_cost / saved_per_group_rank)
        return [max(1, rank - rank_delta) for rank in ranks]

    @staticmethod
    def _compute_budget_preserved_ranks_for_latent_recovery(
        ranks,
        in_features,
        out_features,
        latent_recovery_ranks,
    ):
        group_dim = out_features // len(ranks)
        adjusted_ranks = []
        for rank, recovery_rank in zip(ranks, latent_recovery_ranks):
            if recovery_rank <= 0:
                adjusted_ranks.append(rank)
                continue
            # Preserve the approximate parameter budget after adding
            # ActLoRA(z) = up(down(z)) in the latent space.
            numerator = rank * (in_features + group_dim)
            denominator = in_features + group_dim + (2 * recovery_rank)
            adjusted_rank = max(1, numerator // denominator)
            adjusted_ranks.append(min(rank, adjusted_rank))
        return adjusted_ranks

    @staticmethod
    def _init_shared_basis_from_residual(module, residual_weights, scaling_diag_matrix=None):
        if module.shared_basis_rank <= 0:
            return

        residual_stack = residual_weights.reshape(module.out_features, module.in_features)
        if scaling_diag_matrix is not None:
            shared_left, shared_right = _stacked_whiten_decomposition_from_weight(
                residual_stack,
                scaling_diag_matrix,
                module.shared_basis_rank,
            )
        else:
            shared_left, shared_right = _stacked_decomposition_from_weight(
                residual_stack,
                module.shared_basis_rank,
            )

        module.shared_VT.weight.data = shared_right.contiguous()
        for i in range(module.num_groups):
            start = i * module.group_dim
            end = start + module.group_dim
            module.shared_U[i].weight.data = shared_left[start:end, :].contiguous()

    @staticmethod
    def _reset_latent_recovery_like_act_lora(module):
        for i, recovery_rank in enumerate(module.latent_recovery_ranks):
            if recovery_rank <= 0:
                continue
            nn.init.normal_(module.latent_recovery_A[str(i)].weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.latent_recovery_B[str(i)].weight)
            if module.latent_recovery_use_diag:
                nn.init.zeros_(module.latent_recovery_diag[str(i)])

    @staticmethod
    def from_linear_whiten(
        old_module: nn.Linear,
        ranks: list,
        shared_basis_rank: int = 0,
        latent_recovery_ranks=None,
        latent_recovery_use_diag: bool = False,
        preserve_budget: bool = False,
        latent_recovery_dtype=torch.float32,
        factorization: str = "svd",
        srrqr_bound: float = 2.0,
        srrqr_max_swaps: int = 512,
        factorization_work_device: str = "model",
        factorization_work_dtype=torch.float32,
    ):
        if factorization not in {"svd", "srrqr"}:
            raise ValueError(f"Unsupported activation-aware factorization: {factorization!r}")
        latent_recovery_ranks = latent_recovery_ranks or [0] * len(ranks)
        if preserve_budget and any(r > 0 for r in latent_recovery_ranks):
            adjusted_ranks = HeadwiseLowRankModule._compute_budget_preserved_ranks_for_latent_recovery(
                ranks,
                old_module.in_features,
                old_module.out_features,
                latent_recovery_ranks,
            )
        elif preserve_budget:
            adjusted_ranks = HeadwiseLowRankModule._compute_budget_preserved_ranks(
                ranks,
                old_module.in_features,
                old_module.out_features,
                shared_basis_rank,
            )
        else:
            adjusted_ranks = list(ranks)
        latent_recovery_ranks = [
            min(recovery_rank, adjusted_rank)
            for recovery_rank, adjusted_rank in zip(latent_recovery_ranks, adjusted_ranks)
        ]
        recovery_dtype = _resolve_recovery_dtype(latent_recovery_dtype, old_module.weight.dtype)
        new_module = HeadwiseLowRankModule(
            adjusted_ranks,
            old_module.in_features,
            old_module.out_features,
            bias=old_module.bias is not None,
            shared_basis_rank=shared_basis_rank,
            latent_recovery_ranks=latent_recovery_ranks,
            latent_recovery_use_diag=latent_recovery_use_diag,
            latent_recovery_dtype=recovery_dtype,
        )
        new_module = new_module.to(device=old_module.weight.device, dtype=old_module.weight.dtype)
        for key in new_module.latent_recovery_A.keys():
            new_module.latent_recovery_A[key] = new_module.latent_recovery_A[key].to(
                device=old_module.weight.device,
                dtype=recovery_dtype,
            )
            new_module.latent_recovery_B[key] = new_module.latent_recovery_B[key].to(
                device=old_module.weight.device,
                dtype=recovery_dtype,
            )
        for key in new_module.latent_recovery_diag.keys():
            new_module.latent_recovery_diag[key] = nn.Parameter(
                new_module.latent_recovery_diag[key].to(
                    device=old_module.weight.device,
                    dtype=recovery_dtype,
                )
            )
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features)
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)

        wl = []
        wr = []
        factorization_diagnostics = []
        work_device = _resolve_factorization_work_device(
            factorization_work_device,
            old_module.weight.device,
        )
        work_dtype = _resolve_factorization_work_dtype(factorization_work_dtype)
        if factorization == "srrqr" and (
            work_device != old_module.weight.device or work_dtype != torch.float32
        ):
            raise ValueError(
                "factorization work device/dtype overrides currently apply only to "
                "activation-aware SVD"
            )
        scaling_matrix_work = None
        scaling_matrix_inv_work = None
        if factorization == "svd":
            try:
                scaling_matrix_work = old_module.scaling_diag_matrix.detach().to(
                    device=work_device,
                    dtype=work_dtype,
                )
            except AttributeError as exc:
                raise FileExistsError("Cache may not be loaded correctly") from exc
            # The whitening matrix is common to every head in this projection.
            # Compute its inverse once instead of repeating the O(d^3) solve per head.
            scaling_matrix_inv_work = torch.linalg.inv(scaling_matrix_work)
        for i in range(len(ranks)):
            if factorization == "srrqr":
                factors = activation_aware_strong_rrqr_factors(
                    w[i],
                    old_module.scaling_diag_matrix,
                    adjusted_ranks[i],
                    bound=srrqr_bound,
                    max_swaps=srrqr_max_swaps,
                )
                l, r = factors.left, factors.right
                diagnostics = factors.diagnostics.to_dict()
                diagnostics["group_index"] = i
                factorization_diagnostics.append(diagnostics)
            else:
                l, r = _per_head_whiten_decomposition_from_weight(
                    w[i],
                    scaling_matrix_work,
                    adjusted_ranks[i],
                    scaling_matrix_inv=scaling_matrix_inv_work,
                )
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight

        if shared_basis_rank > 0:
            residual_weights = []
            for i in range(len(adjusted_ranks)):
                residual_weights.append(w[i] - (wl[i] @ wr[i]).to(w[i].dtype))
            HeadwiseLowRankModule._init_shared_basis_from_residual(
                new_module,
                torch.stack(residual_weights),
                scaling_diag_matrix=old_module.scaling_diag_matrix,
            )
            new_module.shared_VT = new_module.shared_VT.to(
                device=old_module.weight.device,
                dtype=torch.float32,
            )
            new_module.shared_U = new_module.shared_U.to(
                device=old_module.weight.device,
                dtype=torch.float32,
            )

        if any(rank > 0 for rank in latent_recovery_ranks):
            HeadwiseLowRankModule._reset_latent_recovery_like_act_lora(new_module)

        new_module.factorization_method = f"activation_aware_{factorization}"
        new_module.factorization_work_device = str(work_device)
        new_module.factorization_work_dtype = str(work_dtype)
        new_module.factorization_diagnostics = factorization_diagnostics
        return new_module

    @staticmethod
    def from_linear(
        old_module: nn.Linear,
        ranks: list,
        shared_basis_rank: int = 0,
        latent_recovery_ranks=None,
        latent_recovery_use_diag: bool = False,
        preserve_budget: bool = False,
        latent_recovery_dtype=torch.float32,
    ):
        latent_recovery_ranks = latent_recovery_ranks or [0] * len(ranks)
        if preserve_budget and any(r > 0 for r in latent_recovery_ranks):
            adjusted_ranks = HeadwiseLowRankModule._compute_budget_preserved_ranks_for_latent_recovery(
                ranks,
                old_module.in_features,
                old_module.out_features,
                latent_recovery_ranks,
            )
        elif preserve_budget:
            adjusted_ranks = HeadwiseLowRankModule._compute_budget_preserved_ranks(
                ranks,
                old_module.in_features,
                old_module.out_features,
                shared_basis_rank,
            )
        else:
            adjusted_ranks = list(ranks)
        latent_recovery_ranks = [
            min(recovery_rank, adjusted_rank)
            for recovery_rank, adjusted_rank in zip(latent_recovery_ranks, adjusted_ranks)
        ]
        recovery_dtype = _resolve_recovery_dtype(latent_recovery_dtype, old_module.weight.dtype)
        new_module = HeadwiseLowRankModule(
            adjusted_ranks,
            old_module.in_features,
            old_module.out_features,
            bias=old_module.bias is not None,
            shared_basis_rank=shared_basis_rank,
            latent_recovery_ranks=latent_recovery_ranks,
            latent_recovery_use_diag=latent_recovery_use_diag,
            latent_recovery_dtype=recovery_dtype,
        )
        new_module = new_module.to(device=old_module.weight.device, dtype=old_module.weight.dtype)
        for key in new_module.latent_recovery_A.keys():
            new_module.latent_recovery_A[key] = new_module.latent_recovery_A[key].to(
                device=old_module.weight.device,
                dtype=recovery_dtype,
            )
            new_module.latent_recovery_B[key] = new_module.latent_recovery_B[key].to(
                device=old_module.weight.device,
                dtype=recovery_dtype,
            )
        for key in new_module.latent_recovery_diag.keys():
            new_module.latent_recovery_diag[key] = nn.Parameter(
                new_module.latent_recovery_diag[key].to(
                    device=old_module.weight.device,
                    dtype=recovery_dtype,
                )
            )
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features)
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_decomposition_from_weight(w[i], adjusted_ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]
        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight

        if shared_basis_rank > 0:
            residual_weights = []
            for i in range(len(adjusted_ranks)):
                residual_weights.append(w[i] - (wl[i] @ wr[i]).to(w[i].dtype))
            HeadwiseLowRankModule._init_shared_basis_from_residual(
                new_module,
                torch.stack(residual_weights),
            )
            new_module.shared_VT = new_module.shared_VT.to(
                device=old_module.weight.device,
                dtype=torch.float32,
            )
            new_module.shared_U = new_module.shared_U.to(
                device=old_module.weight.device,
                dtype=torch.float32,
            )

        if any(rank > 0 for rank in latent_recovery_ranks):
            HeadwiseLowRankModule._reset_latent_recovery_like_act_lora(new_module)

        return new_module
