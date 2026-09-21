import json
from types import SimpleNamespace

from safetensors.torch import save_file
import torch
from torch import nn

from evaluation.eval_nemotron_h_palu_quality import install_palu_v
from evaluation.nemotron_h_palu import (
    CHECKPOINT_FORMAT,
    _balanced_group_factors,
    _group_eigendecomposition,
    allocate_schedules,
)
from evaluation.v96kl_common import sha256
from palu.model.modules.svd_linear import HeadwiseLowRankModule


def test_fisher_schedules_match_global_budgets():
    scalars = {"7": 1.0, "18": 2.0, "29": 3.0, "40": 4.0}
    schedules = allocate_schedules(scalars, num_kv_heads=8, head_dim=128)
    for method, groups in (("mlrd", 8), ("glrd4", 2)):
        for mean_rank, retained in ((64, 0.5), (96, 0.75)):
            schedule = schedules[(method, mean_rank)]
            assert schedule["rank_sum"] == int(schedule["dense_rank_sum"] * retained)
            assert schedule["realized_retained_ratio"] == retained
            assert all(len(ranks) == groups for ranks in schedule["rank_map"].values())
            assert all(
                rank % 32 == 0
                for ranks in schedule["rank_map"].values()
                for rank in ranks
            )


def test_fisher_schedules_stay_exact_with_saturated_layers():
    scalars = {
        "7": 0.0005908617749810219,
        "18": 0.0023561171256005764,
        "29": 0.002587194787338376,
        "40": 0.002647116081789136,
        "51": 0.0016895073931664228,
        "62": 0.001000928576104343,
        "73": 0.0005893005291000009,
        "84": 0.0003579954500310123,
        "95": 0.00033346450072713196,
        "106": 0.0004157528164796531,
    }
    schedules = allocate_schedules(scalars, num_kv_heads=8, head_dim=128)
    for schedule in schedules.values():
        assert schedule["rank_sum"] in (5120, 7680)
        assert schedule["realized_retained_ratio"] in (0.5, 0.75)


def test_balanced_factors_reconstruct_at_full_rank():
    generator = torch.Generator().manual_seed(123)
    weight = torch.randn(12, 17, generator=generator, dtype=torch.float64)
    seed = torch.randn(17, 17, generator=generator, dtype=torch.float64)
    covariance = seed @ seed.mT + torch.eye(17, dtype=torch.float64)
    values, vectors = _group_eigendecomposition(weight, covariance)
    writer, decoder, selected = _balanced_group_factors(
        weight, values, vectors, rank=12
    )
    torch.testing.assert_close(decoder @ writer, weight, rtol=1e-9, atol=1e-9)
    assert torch.all(selected[:-1] >= selected[1:])


class _ToyMixer(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == "full_attention":
            self.v_proj = nn.Linear(5, 8, bias=False)
            self.o_proj = nn.Linear(8, 5, bias=False)
        else:
            self.out_proj = nn.Linear(8, 5, bias=False)


class _ToyLayer(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.block_type = kind
        self.mixer = _ToyMixer(kind)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=[_ToyLayer("linear_attention"), _ToyLayer("full_attention")])


def test_install_replaces_only_attention_v_proj(tmp_path):
    model = _ToyModel()
    dense_v = model.model.layers[1].mixer.v_proj
    dense_o = model.model.layers[1].mixer.o_proj
    dense_mamba_wo = model.model.layers[0].mixer.out_proj
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    factors = {
        "layers.1.v_writer.weight": torch.randn(4, 5, dtype=torch.bfloat16),
        "layers.1.v_decoder.weight": torch.randn(2, 4, 2, dtype=torch.bfloat16),
    }
    artifact = checkpoint / "palu_v_factors.safetensors"
    save_file(factors, artifact)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n")
    manifest = {
        "status": "complete",
        "format": CHECKPOINT_FORMAT,
        "model": {
            "path": str(model_dir),
            "config_sha256": sha256(model_dir / "config.json"),
        },
        "compression": {
            "attention_o_proj": "dense_unchanged",
            "mamba2_wo": "dense_unchanged",
        },
        "artifact": {
            "file": artifact.name,
            "sha256": sha256(artifact),
        },
        "architecture": {
            "full_attention_layers": [1],
            "mamba_layers": [0],
        },
        "layers": [{"layer": 1, "ranks": [2, 2]}],
    }
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    loaded, installed = install_palu_v(model, checkpoint)
    assert loaded == manifest and len(installed) == 1
    assert isinstance(model.model.layers[1].mixer.v_proj, HeadwiseLowRankModule)
    assert model.model.layers[1].mixer.v_proj is not dense_v
    assert model.model.layers[1].mixer.o_proj is dense_o
    assert model.model.layers[0].mixer.out_proj is dense_mamba_wo
    torch.testing.assert_close(
        model.model.layers[1].mixer.v_proj.VT.weight.cpu(),
        factors["layers.1.v_writer.weight"].float(),
    )
