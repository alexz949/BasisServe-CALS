from pathlib import Path


def test_qgram_score_only_source_has_no_fisher_artifact_path():
    source = Path("evaluation/fit_llama_section4_qgram_score_only.py").read_text()
    assert 'root / "fisher"' not in source
    assert 'fisher_artifacts_read": []' in source
    assert "score_statistics" in source
    assert "initialize(fit_stats, 16)" in source
