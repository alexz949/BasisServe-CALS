from __future__ import annotations

import torch
from torch import nn

from basisserve.core.nemotron_h_c1 import (
    discover_nemotron_h_c1_targets,
    first_nemotron_h_c1_target_per_kind,
    projection_for_target,
)
from basisserve.core.tp_source_wo_fit import (
    TPSourceWOFitConfig,
    fit_tp_source_wo_c1,
)
from evaluation.run_nemotron_h_c1_wo_pilot import (
    _cuda_placement,
    _model_loading_configuration,
)


class _Mixer(nn.Module):
    def __init__(self, kind: str, input_width: int, output_width: int) -> None:
        super().__init__()
        name = "out_proj" if kind == "linear_attention" else "o_proj"
        setattr(self, name, nn.Linear(input_width, output_width, bias=False))


class _Block(nn.Module):
    def __init__(self, kind: str, input_width: int, output_width: int) -> None:
        super().__init__()
        self.block_type = kind
        self.mixer = _Mixer(kind, input_width, output_width)


class _Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _Block("linear_attention", 24, 16),
                _Block("mlp", 16, 16),
                _Block("full_attention", 16, 16),
            ]
        )


class _CausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Base()


def test_discovers_first_mamba_and_attention_output_projections() -> None:
    model = _CausalLM()
    targets = discover_nemotron_h_c1_targets(model)

    assert [(target.layer_index, target.layer_kind) for target in targets] == [
        (0, "linear_attention"),
        (2, "full_attention"),
    ]
    assert [(target.input_width, target.output_width) for target in targets] == [
        (24, 16),
        (16, 16),
    ]
    mamba, attention = first_nemotron_h_c1_target_per_kind(model)
    assert mamba == targets[0]
    assert attention == targets[1]
    assert projection_for_target(model, attention) is model.model.layers[2].mixer.o_proj


def test_target_layout_handles_rectangular_mamba_output_projection() -> None:
    model = _CausalLM()
    mamba, _ = first_nemotron_h_c1_target_per_kind(model)
    layout = mamba.layout(tp_size=4, source_rank=2)

    assert layout.source_width == 6
    assert layout.gathered_width == 8
    assert layout.reduction_vs_dense_allreduce == 0.75


def test_generic_tp_source_c1_fit_is_finite_and_improves_zero_map() -> None:
    generator = torch.Generator().manual_seed(20260824)
    train = torch.randn(96, 12, generator=generator, dtype=torch.float64)
    heldout = torch.randn(64, 12, generator=generator, dtype=torch.float64)
    weight = torch.randn(7, 12, generator=generator, dtype=torch.float64)
    model = _CausalLM()
    target = discover_nemotron_h_c1_targets(
        model, kinds=("full_attention",)
    )[0]
    layout = target.__class__(
        layer_index=target.layer_index,
        layer_kind=target.layer_kind,
        projection_name=target.projection_name,
        input_width=12,
        output_width=7,
    ).layout(tp_size=3, source_rank=2)

    result = fit_tp_source_wo_c1(
        weight,
        train.transpose(0, 1) @ train / len(train),
        heldout.transpose(0, 1) @ heldout / len(heldout),
        layout,
        config=TPSourceWOFitConfig(encoder_sweeps=0),
        work_dtype=torch.float64,
        factor_dtype=torch.float32,
    )

    assert result.source_encoders.shape == (3, 4, 2)
    assert result.source_decoders.shape == (3, 2, 7)
    assert 0.0 <= result.fit_relative_mse < 1.0
    assert 0.0 <= result.heldout_relative_mse < 1.0
    assert result.selected_boundary == "decoder_only"
    assert result.selected_sweep == 0
    assert result.checkpoints
    assert torch.isfinite(result.source_encoders).all()
    assert torch.isfinite(result.source_decoders).all()


def test_large_model_loading_reserves_solver_gpu_without_cpu_offload() -> None:
    device_map, max_memory = _model_loading_configuration(
        strategy="balanced_low_0",
        solver_device=torch.device("cuda:0"),
        cuda_device_count=2,
        max_memory_per_gpu_gib=76,
    )

    assert device_map == "balanced_low_0"
    assert max_memory == {0: "76GiB", 1: "76GiB"}

    model = _CausalLM()
    model.hf_device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 1,
    }
    assert _cuda_placement(model) == {
        "model.embed_tokens": "0",
        "model.layers.0": "1",
    }


def test_large_model_loading_rejects_one_gpu_and_cpu_offload() -> None:
    try:
        _model_loading_configuration(
            strategy="balanced_low_0",
            solver_device=torch.device("cuda:0"),
            cuda_device_count=1,
            max_memory_per_gpu_gib=76,
        )
    except ValueError as error:
        assert "at least two CUDA devices" in str(error)
    else:
        raise AssertionError("multi-GPU loading unexpectedly accepted one GPU")

    model = _CausalLM()
    model.hf_device_map = {"model.layers.0": "cpu"}
    try:
        _cuda_placement(model)
    except RuntimeError as error:
        assert "offloaded" in str(error)
    else:
        raise AssertionError("CPU offload unexpectedly accepted")
