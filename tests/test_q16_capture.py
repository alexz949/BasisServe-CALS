from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb

from scripts.capture_qwen3_8b_q16 import (
    Q16_POSITIONS, Q32_POSITIONS, UNIFORM_Q16_POSITIONS, UNIFORM_Q32_POSITIONS, assert_q8_overlap, assert_q16_overlap, query_positions,
    selected_rotated_queries, verify_document,
)
from evaluation.fit_qwen3_8b_qaware_base_fisher_bank import QUERY_POSITIONS
from evaluation.fit_qwen3_8b_q8_fisher_residual import expanded_queries, parser
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json


def test_q16_nesting_and_overlap_gate():
    assert len(Q16_POSITIONS) == 16 and Q16_POSITIONS[1::2] == QUERY_POSITIONS
    assert Q16_POSITIONS == [24576 + (i + 1) * 512 - 1 for i in range(16)]
    queries = torch.randn(16, 4, 6).bfloat16()
    reference = queries[1::2].float().clone()
    assert_q8_overlap(queries, reference)
    changed = queries.clone()
    changed[0] += 10  # A new position need not equal an existing Q8 position.
    assert_q8_overlap(changed, reference)
    changed[1] += 10
    with TestCase().assertRaises(AssertionError):
        assert_q8_overlap(changed, reference)


def test_q32_contains_every_q16_position_and_checks_values():
    assert query_positions("terminal8k", 32) == Q32_POSITIONS
    assert Q32_POSITIONS == [24576 + (i + 1) * 256 - 1 for i in range(32)]
    assert Q32_POSITIONS[1::2] == Q16_POSITIONS
    queries = torch.randn(3, 32, 4, 6).bfloat16()
    reference = queries[:, 1::2].clone()
    assert_q16_overlap(queries, reference)
    queries[:, 0] += 10
    assert_q16_overlap(queries, reference)
    queries[:, 1] += 10
    with TestCase().assertRaises(AssertionError):
        assert_q16_overlap(queries, reference)


def test_terminal_q64_q128_nesting_and_overlap():
    for count in (64, 128):
        positions = query_positions("terminal8k", count)
        assert len(positions) == count and positions[-1] == 32767
        stride = count // 32
        assert positions[stride - 1::stride] == Q32_POSITIONS
        queries = torch.randn(count, 4, 6).bfloat16()
        stride = count // 16
        reference = queries[stride - 1::stride].clone()
        assert_q16_overlap(queries, reference)
        q8 = queries[[positions.index(p) for p in QUERY_POSITIONS]].clone()
        assert_q8_overlap(queries, q8, positions)
        queries[-1] += 10
        with TestCase().assertRaises(AssertionError):
            assert_q16_overlap(queries, reference)
    with TestCase().assertRaises(AssertionError):
        query_positions("uniform32k", 64)


def test_uniform_q32_nesting_and_both_overlap_gates():
    positions = query_positions("uniform32k", 32)
    assert positions == UNIFORM_Q32_POSITIONS == [1024 * (i + 1) - 1 for i in range(32)]
    assert positions[1::2] == UNIFORM_Q16_POSITIONS
    assert sorted(set(positions) & set(QUERY_POSITIONS)) == QUERY_POSITIONS
    queries = torch.randn(32, 4, 6).bfloat16()
    reference = queries[1::2].clone()
    q8 = queries[[positions.index(p) for p in QUERY_POSITIONS]].clone()
    assert_q16_overlap(queries, reference)
    assert_q8_overlap(queries, q8, positions)
    queries[-1] += 10
    with TestCase().assertRaises(AssertionError):
        assert_q16_overlap(queries, reference)
    with TestCase().assertRaises(AssertionError):
        assert_q8_overlap(queries, q8, positions)


def test_uniform_q16_spans_full_context_and_checks_shared_q8_positions():
    assert query_positions("uniform32k") == UNIFORM_Q16_POSITIONS
    assert UNIFORM_Q16_POSITIONS == [2048 * (i + 1) - 1 for i in range(16)]
    overlap = sorted(set(UNIFORM_Q16_POSITIONS) & set(QUERY_POSITIONS))
    assert overlap == [26623, 28671, 30719, 32767]
    queries = torch.randn(16, 4, 6).bfloat16()
    reference = torch.randn(8, 4, 6).bfloat16()
    for position in overlap:
        queries[UNIFORM_Q16_POSITIONS.index(position)] = reference[QUERY_POSITIONS.index(position)]
    assert_q8_overlap(queries, reference, UNIFORM_Q16_POSITIONS)
    queries[UNIFORM_Q16_POSITIONS.index(overlap[0])] += 1
    with TestCase().assertRaises(AssertionError):
        assert_q8_overlap(queries, reference, UNIFORM_Q16_POSITIONS)


@torch.inference_mode()
def test_selected_rope_matches_full_query_capture():
    torch.manual_seed(671)
    config = Qwen3Config(hidden_size=16, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=4)
    module = Qwen3Attention(config, layer_idx=0).eval()
    hidden = torch.randn(1, 12, 16)
    angle = (torch.arange(12)[:, None] * torch.tensor([.2, .7])[None]).repeat(1, 2)[None]
    embeddings = (angle.cos(), angle.sin())
    query = module.q_norm(module.q_proj(hidden).view(1, 12, 4, 4)).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden).view(1, 12, 2, 4)).transpose(1, 2)
    full, _ = apply_rotary_pos_emb(query, key, *embeddings)
    positions = [2, 5, 8, 11]
    selected = selected_rotated_queries(module, hidden, embeddings, positions)
    torch.testing.assert_close(selected, full.transpose(1, 2)[:, positions], rtol=0, atol=0)


def test_expanded_query_loader_preserves_document_layer_and_split():
    with TemporaryDirectory(prefix="q16-test-") as temporary:
        root = Path(temporary)
        for index in range(80):
            tensor = torch.stack([torch.full((16, 2, 4), index * 3 + layer, dtype=torch.bfloat16)
                                  for layer in range(3)])
            save_file({"queries": tensor}, str(root / f"window_{index:03d}.safetensors"))
        fit, positions = expanded_queries(root, "fit", 2)
        validation, _ = expanded_queries(root, "validation", 1)
        assert positions.tolist() == Q16_POSITIONS
        assert fit.shape == (64, 16, 2, 4) and validation.shape == (16, 16, 2, 4)
        for index in range(64):
            assert torch.all(fit[index] == index * 3 + 2)
        for index in range(16):
            assert torch.all(validation[index] == (index + 64) * 3 + 1)


def test_completed_document_requires_hash_and_overlap_check():
    with TemporaryDirectory(prefix="q16-record-test-") as temporary:
        root = Path(temporary)
        path = root / "window_000.safetensors"
        query = torch.zeros(3, 16, 2, 4, dtype=torch.bfloat16)
        save_file({"queries": query}, str(path))
        spec = {"stored_shape": list(query.shape)}
        record = {"status": "complete", "document": 0, "protocol": spec,
                  "overlap_bitwise_equal": True, "sha256": sha256(path)}
        write_json(path.with_suffix(".json"), record)
        _, actual = verify_document(root, 0, spec)
        assert torch.equal(actual, query)
        record["overlap_bitwise_equal"] = False
        write_json(path.with_suffix(".json"), record)
        with TestCase().assertRaises(AssertionError):
            verify_document(root, 0, spec)


def test_fitter_explicit_capture_option_does_not_change_frozen_base_source():
    args = parser().parse_args(["--model", "/model", "--c1-checkpoint", "/c1",
                               "--query-capture", "/q16", "--output-dir", "/r16q"])
    assert args.query_capture == Path("/q16")
    assert args.initial_bank.name == "q8_qbase_fisher_bank"
