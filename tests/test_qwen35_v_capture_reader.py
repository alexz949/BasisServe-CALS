import torch
import pytest
from safetensors.torch import save_file

from basisserve.core.qwen35_gated_v_als import GatedVCapture, GatedVBlock, initialize_gated_v
from evaluation.qwen35_v_capture_reader import WindowCapture


def test_window_reader_matches_materialized_operator(tmp_path):
    torch.manual_seed(17)
    z = torch.randn(34, 4, 4)
    gate = torch.rand_like(z)
    weight = torch.randn(4, 4, 6)
    target = torch.einsum('nhd,hdo->no', z * gate, weight)
    mapping = torch.tensor([0, 0, 1, 1])
    paths = []
    for index in range(2):
        path = tmp_path / f'w{index}.safetensors'
        selection = slice(index*17, (index+1)*17)
        save_file(dict(z=z[selection], gate=gate[selection], target=target[selection], weight=weight), str(path))
        paths.append(path)
    streamed = WindowCapture(paths, mapping)
    dense = GatedVCapture(z, gate, weight, target, mapping)
    torch.testing.assert_close(streamed.z[11:24], z[11:24], rtol=0, atol=0)
    e1, r1 = initialize_gated_v(streamed, 2, chunk_rows=7)
    e2, r2 = initialize_gated_v(dense, 2, chunk_rows=7)
    torch.testing.assert_close(e1, e2, rtol=0, atol=0)
    for block, fixed, variable in [('encoder', r1, e1), ('decoder', e1, r1)]:
        a = GatedVBlock(streamed, fixed, block=block, chunk_rows=7)
        b = GatedVBlock(dense, fixed, block=block, chunk_rows=7)
        torch.testing.assert_close(a.normal(variable), b.normal(variable), rtol=0, atol=0)
        torch.testing.assert_close(a.adjoint(streamed.target), b.adjoint(target), rtol=0, atol=0)
        assert a.loss(variable) == b.loss(variable)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA parity requires GPU')
def test_streamed_als_matches_materialized_cuda(tmp_path):
    from evaluation.fit_qwen35_k_routing_v import fit_one
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    z = torch.randn(34, 4, 4)
    gate = torch.rand_like(z)
    weight = torch.randn(4, 4, 6)
    target = torch.einsum('nhd,hdo->no', z*gate, weight)
    mapping = torch.tensor([0, 0, 1, 1])
    paths = []
    for index in range(2):
        path = tmp_path/f'{index}.safetensors'
        sl = slice(index*17, (index+1)*17)
        save_file(dict(z=z[sl], gate=gate[sl], target=target[sl], weight=weight), str(path))
        paths.append(path)
    a = WindowCapture(paths, mapping)
    b = GatedVCapture(z, gate, weight, target, mapping)
    a.validate(2)
    a.preload_inputs('cuda:0', chunk_rows=7)
    torch.testing.assert_close(a.z.cpu(), z, atol=0, rtol=0)
    torch.testing.assert_close(a.gate.cpu(), gate, atol=0, rtol=0)
    options = dict(sweeps=2, encoder_cg=16, decoder_cg=20, chunk_rows=7, progress=lambda row: None)
    with torch.inference_mode():
        ae, ar, ah = fit_one(a, 2, **options)
        be, br, bh = fit_one(b, 2, **options)
    torch.testing.assert_close(ae, be, rtol=0, atol=0)
    torch.testing.assert_close(ar, br, rtol=0, atol=0)
    assert ah == bh
