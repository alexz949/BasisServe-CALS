"""Unregularized independent SVD with full within-TP-source second moments."""

import torch

from basisserve.core.tp_source_wo_c1 import (
    TPSourceWOLayout, covariance_to_source_blocks, fold_factors_to_dense_weight,
    weight_to_source_targets,
)


@torch.inference_mode()
def weighted_svd(weight, covariance, rank, *, support_rtol=1e-12,
                 support_atol=0.0, negative_rtol=1e-10):
    """Fit [input, output] weight on the retained positive covariance support."""
    assert weight.dtype == covariance.dtype == torch.float64
    assert weight.ndim == 2 and covariance.shape == (weight.shape[0], weight.shape[0])
    assert 0 < rank <= min(weight.shape)
    assert min(support_rtol, support_atol, negative_rtol) >= 0
    assert torch.isfinite(weight).all() and torch.isfinite(covariance).all()
    symmetric = (covariance + covariance.T) * 0.5
    symmetry_error = (covariance - covariance.T).abs().max().item()
    symmetry_scale = covariance.abs().max().item()
    assert symmetry_error <= 1e-10 * max(symmetry_scale, 1e-30), 'Non-symmetric second moment'
    eigenvalues, vectors = torch.linalg.eigh(symmetric)
    scale = eigenvalues.abs().max().item()
    negative_tolerance = negative_rtol * scale
    assert eigenvalues.min().item() >= -negative_tolerance, 'Materially indefinite second moment'
    threshold = max(support_atol, support_rtol * scale)
    selected = eigenvalues > threshold
    positive = eigenvalues[selected]
    basis = vectors[:, selected]
    support = int(selected.sum())
    effective = min(rank, support, weight.shape[1])
    encoder = weight.new_zeros((weight.shape[0], rank))
    decoder = weight.new_zeros((rank, weight.shape[1]))
    whitened = positive.sqrt()[:, None] * (basis.T @ weight)
    if support:
        u, singular, vh = torch.linalg.svd(whitened, full_matrices=False)
        encoder[:, :effective] = basis @ (u[:, :effective] / positive.sqrt()[:, None])
        decoder[:effective] = singular[:effective, None] * vh[:effective]
        full_rank_loss = (whitened - (u * singular) @ vh).square().sum().item()
    else:
        singular = weight.new_zeros(0)
        full_rank_loss = 0.0
    residual = weight - encoder @ decoder
    tail = singular[effective:].square().sum().item()
    retained_loss = (positive[:, None] * (basis.T @ residual).square()).sum().item()
    target_energy = whitened.square().sum().item()
    audit_tolerance = 1e-9 * max(target_energy, 1e-30)
    assert abs(retained_loss - tail) <= audit_tolerance, 'SVD optimum audit failed'
    assert full_rank_loss <= 1e-18 * max(target_energy, 1e-30), 'Full-support endpoint failed'
    raw_loss = (residual * (symmetric @ residual)).sum().item()
    discarded = eigenvalues[~selected]
    discarded_positive = (eigenvalues > 0) & ~selected
    discarded_positive_loss = (
        eigenvalues[discarded_positive, None]
        * (vectors[:, discarded_positive].T @ residual).square()
    ).sum().item()
    assert torch.isfinite(encoder).all() and torch.isfinite(decoder).all()
    return encoder, decoder, {
        'requested_rank': rank, 'effective_rank': effective, 'support_dimension': support,
        'support_threshold': threshold, 'support_rtol': support_rtol,
        'support_atol': support_atol, 'negative_tolerance': negative_tolerance,
        'minimum_eigenvalue': eigenvalues.min().item(),
        'discarded_eigenvalues': discarded.cpu().tolist(),
        'discarded_positive_eigenvalue_sum': eigenvalues[discarded_positive].sum().item(),
        'discarded_positive_residual_energy': discarded_positive_loss,
        'retained_target_energy': target_energy, 'svd_tail_energy': tail,
        'retained_metric_loss': retained_loss, 'original_metric_loss': raw_loss,
        'original_minus_retained_loss': raw_loss - retained_loss,
        'audit_abs_error': abs(retained_loss - tail), 'audit_abs_tolerance': audit_tolerance,
        'full_rank_retained_metric_loss': full_rank_loss,
        'symmetry_max_abs_error': symmetry_error, 'ridge': 0.0,
        'optimality_scope': 'retained_positive_support_not_silently_full_original_metric',
        'encoder_shape': list(encoder.shape), 'decoder_shape': list(decoder.shape),
    }


@torch.inference_mode()
def output_losses(weight, replacement, covariance, layout, rows):
    """C is Z.T @ Z / rows; report both per-row energy and unnormalized SSE."""
    assert rows > 0 and weight.dtype == replacement.dtype == covariance.dtype == torch.float64
    residual = (weight - replacement).T.contiguous()
    target = weight.T.contiguous()
    blocks = covariance_to_source_blocks(covariance, layout)
    r = residual.reshape(layout.tp_size, layout.source_width, layout.output_width)
    w = target.reshape_as(r)
    local = sum((r[p] * (blocks[p, p] @ r[p])).sum() for p in range(layout.tp_size)).item()
    local_target = sum((w[p] * (blocks[p, p] @ w[p])).sum() for p in range(layout.tp_size)).item()
    final = (residual * (covariance @ residual)).sum().item()
    final_target = (target * (covariance @ target)).sum().item()
    return {
        'rows': rows, 'output_width': layout.output_width,
        'local_per_row': local, 'final_per_row': final,
        'local_sse': local * rows, 'final_sse': final * rows,
        'cross_source_per_row': final - local,
        'local_target_per_row': local_target, 'final_target_per_row': final_target,
        'local_relative_error': local / local_target if local_target > 0 else None,
        'final_relative_error': final / final_target if final_target > 0 else None,
        'normalization': 'per_row=SSE/rows; relative=error_energy/dense_target_energy',
    }


@torch.inference_mode()
def fit_layer(weight, fit_covariance, heldout_covariance, layout, *,
              fit_rows, heldout_rows, support_rtol=1e-12, support_atol=0.0,
              negative_rtol=1e-10):
    assert all(t.dtype == torch.float64 and torch.isfinite(t).all()
               for t in (weight, fit_covariance))
    assert fit_covariance.shape == (layout.input_width, layout.input_width)
    if heldout_covariance is not None:
        assert heldout_covariance.shape == fit_covariance.shape and heldout_covariance.dtype == torch.float64
        assert torch.isfinite(heldout_covariance).all() and heldout_rows > 0
    else:
        assert heldout_rows == 0
    for covariance in (fit_covariance, heldout_covariance):
        if covariance is not None:
            assert torch.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-15)
    targets = weight_to_source_targets(weight, layout)
    fit_blocks = covariance_to_source_blocks(fit_covariance, layout)
    encoders, decoders, audits = [], [], []
    for source in range(layout.tp_size):
        encoder, decoder, audit = weighted_svd(
            targets[source], fit_blocks[source, source], layout.source_rank,
            support_rtol=support_rtol, support_atol=support_atol, negative_rtol=negative_rtol,
        )
        encoders.append(encoder)
        decoders.append(decoder)
        audits.append({'source': source, **audit})
    encoders, decoders = torch.stack(encoders), torch.stack(decoders)
    materialized = fold_factors_to_dense_weight(encoders, decoders, layout)
    exported_weight = materialized.bfloat16()
    assert all(torch.isfinite(t.bfloat16()).all() for t in (materialized, encoders, decoders))
    # Factor-rounding and a single materialized-weight rounding are different paths.
    factor_rounded = fold_factors_to_dense_weight(
        encoders.bfloat16().double(), decoders.bfloat16().double(), layout)
    generator = torch.Generator(device=weight.device).manual_seed(71)
    probe = torch.randn(7, layout.input_width, dtype=torch.float64,
                        device=weight.device, generator=generator)
    sources = probe.reshape(7, layout.tp_size, layout.source_width)
    explicit = torch.einsum('npr,pro->no', torch.einsum('npd,pdr->npr', sources, encoders), decoders)
    dense = probe @ materialized.T
    torch.testing.assert_close(explicit, dense, atol=1e-9, rtol=1e-9)
    bf_sources = sources.bfloat16()
    bf_explicit = sum((bf_sources[:, p] @ encoders[p].bfloat16()) @ decoders[p].bfloat16()
                      for p in range(layout.tp_size))
    bf_dense = probe.bfloat16() @ exported_weight.T
    metrics = {'source_audits': audits, 'geometry': layout.accounting(),
               'fp64_explicit_vs_materialized_max_abs': (explicit - dense).abs().max().item(),
               'probe_type': 'deterministic_normal_inputs_not_heldout_activations',
               'bf16_factor_execution_vs_materialized_probe_max_abs':
                   (bf_explicit.float() - bf_dense.float()).abs().max().item(),
               'materialized_bf16_weight_rounding_frobenius': (exported_weight.double() - materialized).norm().item(),
               'bf16_factor_rounding_weight_frobenius': (factor_rounded - materialized).norm().item(),
               'losses': {}}
    splits = [('fit', fit_covariance, fit_rows)]
    if heldout_covariance is not None:
        splits.append(('heldout', heldout_covariance, heldout_rows))
    else:
        metrics['heldout_status'] = 'not_collected_by_user_request'
    for split, covariance, rows in splits:
        metrics['losses'][split] = {
            name: output_losses(weight, candidate, covariance, layout, rows)
            for name, candidate in [('fp64', materialized),
                                    ('materialized_bf16', exported_weight.double()),
                                    ('bf16_factors_fp64_product', factor_rounded)]
        }
    return {'source_encoders_fp64': encoders, 'source_decoders_fp64': decoders,
            'materialized_weight_bf16': exported_weight}, metrics
