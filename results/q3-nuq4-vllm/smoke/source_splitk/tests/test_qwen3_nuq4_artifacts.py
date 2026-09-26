from pathlib import Path

import pytest
import torch

from basisserve.core.qwen3_nuq4_artifacts import QwenNUQ4Artifacts

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("rank", [64, 96])
def test_same_frozen_checkpoint_and_decoder_fixtures(rank):
    artifacts = QwenNUQ4Artifacts(ROOT / "results/q3-kv4-fp8/formal", rank)
    fixtures = torch.load(ROOT / f"results/q3-a8-tp8/r{rank}.pt", map_location="cpu", weights_only=False)
    assert sum(map(sum, artifacts.schedule)) == 36*8*rank
    for i in (0, 18, 35):
        actual = artifacts.layer(i, 7, "cpu")
        expected = fixtures["layers"][i]
        assert list(actual["latent_widths"]) == expected["widths"]
        assert actual["encoder"].shape == (128, artifacts.schedule[i][7])
        torch.testing.assert_close(actual["decoder"], expected["decoder"], rtol=0, atol=0)
        torch.testing.assert_close(actual["a8_scale"], expected["scale"], rtol=0, atol=0)
