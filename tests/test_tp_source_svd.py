import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from basisserve.core.tp_source_svd import fit_layer, output_losses, weighted_svd
from basisserve.core.tp_source_wo_c1 import TPSourceWOLayout, fold_factors_to_dense_weight
from evaluation.run_tp_source_svd import C4_REVISION, COVARIANCE_FORMAT, FORMAT, identity, install, validate_inputs


def problem(width=16, output=12, rows=80):
    generator = torch.Generator().manual_seed(102)
    x = torch.randn(rows, width, generator=generator, dtype=torch.float64)
    w = torch.randn(width, output, generator=generator, dtype=torch.float64)
    return x, w, x.T @ x / rows


def test_tail_energy_and_full_rank():
    x, w, c = problem()
    e, d, audit = weighted_svd(w, c, 5)
    measured = ((x @ (w - e @ d)).square().sum() / len(x)).item()
    assert measured == pytest.approx(audit['svd_tail_energy'], rel=1e-10)
    assert e.shape == (16, 5) and d.shape == (5, 12)
    assert audit['ridge'] == 0 and audit['support_dimension'] == 16
    e, d, audit = weighted_svd(w, c, 12)
    torch.testing.assert_close(e @ d, w, atol=1e-12, rtol=1e-12)


def test_cross_head_covariance_is_not_pooled():
    w = torch.eye(2, dtype=torch.float64)
    c = torch.tensor([[1., .9], [.9, 1.]], dtype=torch.float64)
    e, d, audit = weighted_svd(w, c, 1)
    assert audit['svd_tail_energy'] == pytest.approx(.1)
    assert abs((e @ d)[0, 1]) > .4
    e0, d0, _ = weighted_svd(w, torch.diag(c.diag()), 1)
    r = w - e0 @ d0
    assert (r * (c @ r)).sum().item() > audit['original_metric_loss'] + .8


def test_rank_deficient_zero_padding_and_unobserved_nullspace():
    c = torch.diag(torch.tensor([4., 1., 0., 0.], dtype=torch.float64))
    w = torch.eye(4, dtype=torch.float64)
    e, d, audit = weighted_svd(w, c, 4)
    assert audit['effective_rank'] == 2
    assert torch.count_nonzero(e[:, 2:]) == torch.count_nonzero(d[2:]) == 0
    assert audit['retained_metric_loss'] < 1e-20
    assert not torch.equal(e @ d, w)
    e, d, audit = weighted_svd(w, torch.zeros_like(c), 3)
    assert audit['support_dimension'] == 0 and audit['svd_tail_energy'] == 0
    assert not torch.count_nonzero(e) and not torch.count_nonzero(d)


def test_discarded_positive_energy_is_reported():
    w = torch.eye(3, dtype=torch.float64)
    c = torch.diag(torch.tensor([2., 1., 1e-9], dtype=torch.float64))
    _, _, audit = weighted_svd(w, c, 3, support_rtol=1e-6)
    assert audit['retained_metric_loss'] < 1e-20
    assert audit['original_metric_loss'] == pytest.approx(1e-9)
    assert audit['discarded_positive_residual_energy'] == pytest.approx(1e-9)
    assert audit['discarded_eigenvalues'] == [1e-9]


def test_materially_negative_and_nonfinite_covariance_rejected():
    w = torch.eye(2, dtype=torch.float64)
    with pytest.raises(AssertionError, match='indefinite'):
        weighted_svd(w, torch.diag(torch.tensor([1., -.1], dtype=torch.float64)), 1)
    with pytest.raises(AssertionError):
        weighted_svd(w, w * float('nan'), 1)
    with pytest.raises(AssertionError, match='symmetric'):
        weighted_svd(w, torch.tensor([[1., 1.], [0., 1.]], dtype=torch.float64), 1)


def test_local_and_final_losses_include_cross_source_terms():
    x, w, c = problem(width=16, output=12)
    layout = TPSourceWOLayout(16, 12, 4, 2)
    approximate = w.T * .3
    loss = output_losses(w.T, approximate, c, layout, len(x))
    residual = w - approximate.T
    per_source = [x[:, p * 4:(p + 1) * 4] @ residual[p * 4:(p + 1) * 4] for p in range(4)]
    assert loss['local_sse'] == pytest.approx(sum(y.square().sum().item() for y in per_source))
    assert loss['final_sse'] == pytest.approx(sum(per_source).square().sum().item())
    assert loss['cross_source_per_row'] == pytest.approx((loss['final_sse'] - loss['local_sse']) / len(x))


@pytest.mark.parametrize('tp_size', [2, 4, 8])
def test_layer_export_once_and_full_rank_endpoint(tp_size):
    x, w, c = problem(width=32, output=32)
    layout = TPSourceWOLayout(32, 32, tp_size, 32 // tp_size)
    factors, audit = fit_layer(w.T, c, c, layout, fit_rows=len(x), heldout_rows=len(x))
    exact = fold_factors_to_dense_weight(factors['source_encoders_fp64'], factors['source_decoders_fp64'], layout)
    torch.testing.assert_close(x @ exact.T, x @ w, rtol=1e-11, atol=1e-11)
    assert torch.equal(factors['materialized_weight_bf16'], exact.bfloat16())
    assert audit['losses']['fit']['fp64']['final_per_row'] < 1e-20
    assert audit['fp64_explicit_vs_materialized_max_abs'] < 1e-11


def test_install_restores_dense_before_each_arm(tmp_path):
    layout = TPSourceWOLayout(8, 8, 2, 2)
    module = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    original = module.weight.detach().clone()
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(o_proj=module))]),
                            config=SimpleNamespace(num_attention_heads=2, head_dim=4))
    config = {'layers': [0], 'tp_size': 2, 'model_identity': {'repo': 'synthetic', 'revision': 'test'}}
    path = tmp_path / 'factors/r2/layer_000.safetensors'
    path.parent.mkdir(parents=True)
    save_file({'materialized_weight_bf16': original * 0}, str(path),
              metadata={'format': FORMAT, 'layout': json.dumps(asdict(layout)),
                        'model_identity': json.dumps(config['model_identity']), 'layer': '0'})
    install(model, [original], tmp_path, 2, config)
    assert not torch.count_nonzero(module.weight)
    install(model, [original], tmp_path, None, config)
    assert torch.equal(module.weight, original)


def test_snapshot_identity_is_not_just_model_geometry():
    base = identity('/cache/models--Qwen--Qwen3-8B-Base/snapshots/abc')
    post = identity('/other/models--Qwen--Qwen3-8B/snapshots/abc')
    assert base != post
    assert base == identity('/other/models--Qwen--Qwen3-8B-Base/snapshots/abc')


def test_fit_only_has_no_invented_heldout_metrics():
    x, w, c = problem()
    layout = TPSourceWOLayout(16, 12, 4, 2)
    _, audit = fit_layer(w.T, c, None, layout, fit_rows=len(x), heldout_rows=0)
    assert set(audit['losses']) == {'fit'}
    assert audit['heldout_status'] == 'not_collected_by_user_request'


def test_input_audit_rejects_wrong_split_or_model(tmp_path):
    model = tmp_path / 'models--Qwen--Qwen3-8B-Base/snapshots/test'
    model.mkdir(parents=True)
    config = {'model_type': 'qwen3', 'hidden_size': 4096, 'num_hidden_layers': 36,
              'num_attention_heads': 32, 'num_key_value_heads': 8, 'head_dim': 128}
    (model / 'config.json').write_text(json.dumps(config))
    for name in ('tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'):
        (model / name).write_text('{}')
    covariance = tmp_path / 'covariance'
    covariance.mkdir()
    manifest = {'format': COVARIANCE_FORMAT, 'status': 'complete', 'phase': 'smoke',
                'model': {'path': str(model)}, 'layers': [0], 'artifacts': {'0': {'file': 'layer_000.safetensors'}},
                'calibration': {'storage': 'normalized_covariance_sufficient_statistics',
                                'fit_windows': 2, 'heldout_windows': 0, 'sequence_length': 2048,
                                'fit_rows': 4096, 'heldout_rows': 0, 'positions_per_window': 2048,
                                'window_count': 2, 'dataset': 'allenai/c4', 'split': 'train',
                                'revision': C4_REVISION, 'moment': 'uncentered_ZtZ_div_N',
                                'windows': [{'document_id': 'a'}, {'document_id': 'b'}]}}
    path = covariance / 'manifest.json'
    path.write_text(json.dumps(manifest))
    validate_inputs(model, covariance, 'smoke')
    manifest['calibration']['split'] = 'validation'
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        validate_inputs(model, covariance, 'smoke')
    manifest['calibration']['split'] = 'train'
    manifest['model']['path'] = str(model).replace('Qwen3-8B-Base', 'Qwen3-8B')
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match='snapshot mismatch'):
        validate_inputs(model, covariance, 'smoke')


def test_shared_ppl_uses_frozen_tokens_and_reports_tail():
    from evaluation.eval_attention_o_proj_collective_ppl import _eval_ppl_fp32_loss

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.config = SimpleNamespace(use_cache=True)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, use_cache=False):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 8))

    model = TinyLM()
    tokens = torch.arange(11) % 8
    result = _eval_ppl_fp32_loss(model, None, dataset='synthetic', split='test', seqlen=4,
                               batch_size=1, max_samples=None, max_tokens=None, input_ids=tokens)
    assert result['ppl'] == pytest.approx(8., rel=1e-6)
    assert result['chunks'] == 2 and result['tokens'] == 6
    assert result['discarded_tail_tokens'] == 3 and result['available_tokens'] == 11
    assert model.config.use_cache is True
