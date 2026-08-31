from types import SimpleNamespace

import torch
from torch import nn

from evaluation import eval_llama2_mha_v25_comparison as comparison


def test_palu_k_m_geometry_and_cache_accounting() -> None:
    assert comparison._palu_geometry("palu_k_m") == (32, 96)
    accounting = comparison._cache_accounting("palu_k_m")
    assert accounting["key_cache_rank_per_layer"] == 32 * 96
    assert accounting["value_cache_rank_per_layer"] == 32 * 128
    assert accounting["key_retained_ratio"] == 0.75
    assert accounting["value_retained_ratio"] == 1.0
    assert accounting["total_kv_reduction_fraction"] == 0.125


def test_install_palu_k_m_replaces_only_key_projection(monkeypatch) -> None:
    attention = SimpleNamespace(
        k_proj=nn.Linear(4, 4, bias=False),
        v_proj=nn.Linear(4, 4, bias=False),
    )
    layer = SimpleNamespace(self_attn=attention)
    model = SimpleNamespace()
    original_key = attention.k_proj
    original_value = attention.v_proj
    replacement = nn.Identity()
    observed = {}

    def fake_factorization(dense, ranks, **kwargs):
        observed["dense"] = dense
        observed["ranks"] = ranks
        observed.update(kwargs)
        replacement.factorization_work_device = kwargs["factorization_work_device"]
        replacement.factorization_work_dtype = kwargs["factorization_work_dtype"]
        return replacement

    monkeypatch.setattr(comparison, "_model_layers", lambda _: [layer])
    monkeypatch.setattr(
        comparison.HeadwiseLowRankModule,
        "from_linear_whiten",
        fake_factorization,
    )
    monkeypatch.setattr(comparison, "NUM_LAYERS", 1)

    records = comparison._install_palu(
        model,
        arm="palu_k_m",
        whitening=[torch.eye(4)],
    )

    assert observed["dense"] is original_key
    assert observed["ranks"] == [96] * 32
    assert attention.k_proj is replacement
    assert attention.v_proj is original_value
    assert records[0]["projection"] == "k_proj"


def test_install_palu_m_still_replaces_only_value_projection(monkeypatch) -> None:
    attention = SimpleNamespace(
        k_proj=nn.Linear(4, 4, bias=False),
        v_proj=nn.Linear(4, 4, bias=False),
    )
    layer = SimpleNamespace(self_attn=attention)
    replacement = nn.Identity()
    replacement.factorization_work_device = "cpu"
    replacement.factorization_work_dtype = "float64"
    original_key = attention.k_proj

    monkeypatch.setattr(comparison, "_model_layers", lambda _: [layer])
    monkeypatch.setattr(
        comparison.HeadwiseLowRankModule,
        "from_linear_whiten",
        lambda *args, **kwargs: replacement,
    )
    monkeypatch.setattr(comparison, "NUM_LAYERS", 1)

    records = comparison._install_palu(
        SimpleNamespace(),
        arm="palu_m",
        whitening=[torch.eye(4)],
    )

    assert attention.k_proj is original_key
    assert attention.v_proj is replacement
    assert records[0]["projection"] == "v_proj"
