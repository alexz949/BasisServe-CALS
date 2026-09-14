"""Check Llama-sized indexed attention against explicit selected-token attention."""
import torch
from evaluation.llama_sink_recent_routing import page_support
from basisserve.kernels.split_indexed_attention import split_indexed_attention


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(93)
    for rank in (64, 80, 96, 112, 128):
        for length in (2049, 65536):
            q = torch.randn(1, 32, 1, 128, device='cuda', dtype=torch.bfloat16)
            k = torch.randn(1, 8, length, 128, device='cuda', dtype=torch.bfloat16)
            v = torch.randn(1, 8, length, rank, device='cuda', dtype=torch.bfloat16)
            scores = (q[:, :, 0].float().reshape(1, 8, 4, 128) @ k.float().mT) / 128**0.5
            ids, valid = page_support(scores)
            selected = ids.masked_fill(~valid, -1).repeat_interleave(4, dim=1)
            actual = split_indexed_attention(q, k, v, selected, scale=128**-0.5)
            mapping = torch.arange(32, device='cuda') // 4
            safe = selected.clamp_min(0)
            keys = k[0, mapping[:, None], safe[0]].float()
            values = v[0, mapping[:, None], safe[0]].float()
            logits = (q[0].float() * keys).sum(-1) / 128**0.5
            logits.masked_fill_(selected[0] < 0, -torch.inf)
            expected = (logits.softmax(-1)[..., None] * values).sum(-2)[None, :, None]
            torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=0.02)
            print(dict(rank=rank, length=length, max_error=float((actual.float()-expected).abs().max()),
                selected_min=int(valid.sum(-1).min()), selected_max=int(valid.sum(-1).max())), flush=True)
    print('PASS sink32/recent64 indexed attention across all five V ranks', flush=True)


if __name__ == '__main__':
    main()
