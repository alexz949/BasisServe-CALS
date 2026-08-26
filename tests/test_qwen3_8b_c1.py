from __future__ import annotations

from types import SimpleNamespace

from evaluation import allocate_qwen3_32b_c1_tp_source_global_kl as common
from evaluation import eval_qwen3_32b_c1_wikitext as evaluator
from evaluation import fit_llama2_mha_c1_joint as fitter
from evaluation import run_qwen3_32b_c1_layer_global_kl_sharded as layer_global
from evaluation import run_qwen3_32b_c1_tp_source_global_kl_sharded as runtime


def test_qwen3_8b_profile_has_audited_gqa_geometry_and_formats() -> None:
    try:
        fitter.activate_model_profile("qwen3_8b")
        layer_global.activate_model_profile("qwen3_8b")
        evaluator.activate_model_profile("qwen3_8b")

        assert (
            fitter.NUM_LAYERS,
            fitter.NUM_HEADS,
            fitter.NUM_KV_HEADS,
            fitter.HEAD_DIM,
            fitter.HIDDEN_SIZE,
        ) == (36, 32, 8, 128, 4096)
        assert fitter.FORMAT == "basisserve.qwen3_8b.gqa_c1_joint.v1"
        assert common.NUM_LAYERS == 36
        assert common.NUM_QUERY_HEADS == 32
        assert common.NUM_KV_HEADS == 8
        assert common.HEADS_PER_SOURCE == 4
        assert common.QUERY_WIDTH == 4096
        assert runtime.MODEL_LABEL == "Qwen3-8B-Base"
        assert layer_global.FORMAT.endswith("qwen3_8b.gqa_c1.layer_global_kl_allocation.v1")
        assert evaluator.FACTOR_FORMAT == fitter.FORMAT

        config = SimpleNamespace(
            model_type="qwen3",
            num_hidden_layers=36,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        layout = evaluator._layout(config, 64)
        assert layout.rank == 64
    finally:
        fitter.activate_model_profile("qwen3_32b")
        layer_global.activate_model_profile("qwen3_32b")
        evaluator.activate_model_profile("qwen3_32b")


def test_qwen3_8b_layer_dp_preserves_exact_average_rank64() -> None:
    try:
        layer_global.activate_model_profile("qwen3_8b")
        records = []
        for layer in range(common.NUM_LAYERS):
            for rank in (32, 48, 80, 96, 112, 128):
                records.append(
                    {
                        "layer": layer,
                        "candidate_rank": rank,
                        "terminal_kl_delta": {
                            "mean": 1.0,
                            "one_standard_error_ucb": 1.0,
                        },
                    }
                )
        schedule, cost, _ = layer_global._allocate_layer_ranks(
            records,
            candidate_ranks=(32, 48, 64, 80, 96, 112, 128),
            anchor_rank=64,
            cost_key="mean",
        )
        assert schedule == [[64] * 8 for _ in range(36)]
        assert cost == 0.0
        accounting = layer_global._layer_schedule_accounting(
            schedule, anchor_rank=64
        )
        assert accounting["layer_rank_sum"] == 36 * 64
        assert accounting["source_rank_sum"] == 36 * 8 * 64
        assert accounting["dense_reduction"] == 2.0
    finally:
        layer_global.activate_model_profile("qwen3_32b")
