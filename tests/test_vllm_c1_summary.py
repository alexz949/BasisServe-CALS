"""Summary generation for structurally validated TP8 benchmark records."""

import json
import sys

from evaluation.summarize_vllm_qwen3_8b_c1 import main


def test_structural_validation_and_h8_decode_label(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "num_hidden_layers": 64, "num_attention_heads": 64, "hidden_size": 5120,
    }))
    arguments = dict(batch_sizes=[1], prefill_tokens=8192, decode_tokens=128,
                     max_num_batched_tokens=8192, repeats=1, warmups=1,
                     gpu_memory_utilization=0.8)
    run = dict(wall_seconds=4., output_tokens_per_second=32., preemptions=0,
               ttft_ms={"mean": 1000.}, tpot_ms={"mean": 20.})
    for arm in ("dense", "c1"):
        record = dict(
            status="complete", arguments=arguments,
            configuration={"model": str(model)},
            factor_validation="structure", factor_sha256=None,
            torch="test", vllm="test", command="unit-test fixture",
            batches=[{"batch": 1, "runs": [run], "workers": [
                {"decode_specialization": "sm89_qk128_v64_h8"}]}],
        )
        (tmp_path / f"{arm}.json").write_text(json.dumps(record))
    monkeypatch.setattr(sys, "argv", ["summary", "--input-dir", str(tmp_path)])
    main()
    summary = (tmp_path / "summary.md").read_text()
    assert "8192 prompt tokens" in summary
    assert "sm89_qk128_v64_h8" in summary
    assert "no SHA256 checks were performed" in summary
    assert "upstream split-KV" not in summary
    assert "SHA256: `None`" not in summary
