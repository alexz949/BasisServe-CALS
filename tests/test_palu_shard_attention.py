"""Decode kernels of the sharded PaLU group latent: explicit attention weights over paged Keys
and latent-shard weighted sums, for both TP4 and TP8 head layouts. Group collectives are
checked separately on CPU; full-model NCCL validation uses validate_palu_g4_vllm.py."""

import pytest
import torch

pytest.importorskip("vllm")
from basisserve.vllm.palu_shard_attention import apply_weights, decode_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
BLOCK = 32


@pytest.mark.parametrize("kv_heads,shard,lengths", [
    (2, 224, [70, 33, 160]), (2, 96, [31, 32, 33, 64, 65, 300]), (2, 256, [1, 1000]),
    (1, 56, [70, 33, 160]), (1, 96, [31, 32, 33, 64, 65, 300]), (1, 104, [1, 1000]),
])
def test_decode_weights_and_shard_sums_match_reference(kv_heads, shard, lengths):
    heads = kv_heads * 4
    torch.manual_seed(2)
    lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    max_blocks = int((lengths.max().item() + BLOCK - 1) // BLOCK)
    total_blocks = max_blocks * len(lengths)
    flat = torch.randn(total_blocks, BLOCK, kv_heads * 128 + shard, device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(total_blocks, device="cuda", dtype=torch.int32).view(len(lengths), max_blocks)
    query = torch.randn(len(lengths), heads, 128, device="cuda", dtype=torch.bfloat16)
    scale = 128 ** -0.5

    weights = decode_weights(query, flat, table, lengths, int(lengths.max()), kv_heads, scale)
    # Production computes all subgroup query owners' partials in a single launch.
    weights = torch.cat([weights * (index + 1) for index in range(4 // kv_heads)], dim=1)
    sums = apply_weights(weights, flat, table, lengths, kv_heads, shard)

    for seq, length in enumerate(lengths.tolist()):
        slots = (table[seq, :, None] * BLOCK + torch.arange(BLOCK, device="cuda")).flatten()[:length]
        rows = flat.reshape(-1, flat.shape[-1])[slots].float()
        keys, latent = rows[:, :kv_heads * 128].view(length, kv_heads, 128), rows[:, kv_heads * 128:]
        for head in range(weights.shape[1]):
            local_head = head % heads
            multiplier = head // heads + 1
            scores = keys[:, local_head // 4] @ query[seq, local_head].float() * scale
            expected = torch.softmax(scores, dim=0) * multiplier
            torch.testing.assert_close(weights[seq, head, :length].float(), expected, rtol=2e-2, atol=1e-3)
            assert not weights[seq, head, length:].any()
            # The kernel consumes the BF16 weights it was given; compare against those.
            torch.testing.assert_close(sums[seq, head], weights[seq, head, :length].float() @ latent,
                                       rtol=2e-2, atol=2e-2)
