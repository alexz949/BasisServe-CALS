"""Decode kernels of the sharded PaLU group latent: explicit attention weights over paged Keys
and latent-shard weighted sums. Single GPU; the pair exchange itself is covered by the TP4
validation script."""

import pytest
import torch

pytest.importorskip("vllm")
from basisserve.vllm.palu_shard_attention import apply_weights, decode_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
KV_HEADS, HEADS, BLOCK = 2, 8, 32


@pytest.mark.parametrize("shard,lengths", [(224, [70, 33, 160]), (96, [31, 32, 33, 64, 65, 300]), (256, [1, 1000])])
def test_decode_weights_and_shard_sums_match_reference(shard, lengths):
    torch.manual_seed(2)
    lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    max_blocks = int((lengths.max().item() + BLOCK - 1) // BLOCK)
    total_blocks = max_blocks * len(lengths)
    flat = torch.randn(total_blocks, BLOCK, KV_HEADS * 128 + shard, device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(total_blocks, device="cuda", dtype=torch.int32).view(len(lengths), max_blocks)
    query = torch.randn(len(lengths), HEADS, 128, device="cuda", dtype=torch.bfloat16)
    scale = 128 ** -0.5

    weights = decode_weights(query, flat, table, lengths, int(lengths.max()), KV_HEADS, scale)
    sums = apply_weights(weights, flat, table, lengths, KV_HEADS, shard)

    for seq, length in enumerate(lengths.tolist()):
        slots = (table[seq, :, None] * BLOCK + torch.arange(BLOCK, device="cuda")).flatten()[:length]
        rows = flat.reshape(-1, flat.shape[-1])[slots].float()
        keys, latent = rows[:, :KV_HEADS * 128].view(length, KV_HEADS, 128), rows[:, KV_HEADS * 128:]
        for head in range(HEADS):
            scores = keys[:, head // (HEADS // KV_HEADS)] @ query[seq, head].float() * scale
            expected = torch.softmax(scores, dim=0)
            torch.testing.assert_close(weights[seq, head, :length].float(), expected, rtol=2e-2, atol=1e-3)
            assert not weights[seq, head, length:].any()
            # The kernel consumes the BF16 weights it was given; compare against those.
            torch.testing.assert_close(sums[seq, head], weights[seq, head, :length].float() @ latent,
                                       rtol=2e-2, atol=2e-2)
