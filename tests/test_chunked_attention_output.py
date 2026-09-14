import torch
from evaluation.chunked_attention_output import project_attention_output


@torch.inference_mode()
def test_head_major_projection_matches_direct():
    torch.manual_seed(31)
    projection = torch.nn.Linear(8*24, 64, bias=False).to(dtype=torch.bfloat16)
    for length in (1, 1031):
        attention = torch.randn(1, 8, length, 24, dtype=torch.bfloat16)
        expected = projection(attention.transpose(1, 2).contiguous().reshape(1, length, 8*24))
        actual = project_attention_output(attention, projection)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
