import importlib.util
from pathlib import Path
import torch

from basisserve.core.qwen3_tp4_quant_quality import nuq_reconstruct, simulate_padded_wire
from basisserve.kernels.int4_wire import pack_int4, unpack_int4


def test_v_only_fp8_arm_keeps_k_bf16():
    from evaluation.eval_qwen3_tp4_quant_ppl import ARMS
    assert ARMS['v4'] == (False, True, 'bf16')
    assert ARMS['v4-wire8'] == (False, True, 'e4m3')
    assert ARMS['bf16'] == (False, False, 'bf16')
    assert ARMS['kv4-wire8'] == (True, True, 'e4m3')


def test_padded_wire_groups_exclude_padding():
    torch.manual_seed(18)
    value = torch.randn(2, 3, 4096).to(torch.bfloat16)
    heads = value.reshape(-1, 32, 128)
    heads[:, :, 64:] = 1000  # This must never affect quantization scales.
    output = simulate_padded_wire(value, 64).reshape(-1, 32, 128)
    torch.testing.assert_close(output[:, :, 64:], heads[:, :, 64:], rtol=0, atol=0)
    for i in range(4):
        shard = heads[:, i*8:(i+1)*8, :64].reshape(-1, 512)
        expected = unpack_int4(pack_int4(shard), 512).reshape(-1, 8, 64)
        torch.testing.assert_close(output[:, i*8:(i+1)*8, :64], expected, rtol=0, atol=0)


def test_nuq_reference_and_key_sharding():
    source = Path(__file__).resolve().parents[1]/'external/KVQuant/quant/kvquant/simquant_module_quantizer.py'
    spec = importlib.util.spec_from_file_location('kvquant_test', source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    torch.manual_seed(17)
    data = torch.randn(6, 32).to(torch.bfloat16)
    params = (torch.full((32,), 1.5), torch.full((32,), -1.5),
              [torch.linspace(-1, 1, 16).numpy()])
    for dynamic in (False, True):
        axis = -1 if dynamic else 0
        mask = (upstream.get_outliers_dynamic(data.float(), channel=-1, thresh=.99)
                if dynamic else upstream.get_outliers(data.float(), channel=0,
                    outlier_threshold_upper=params[0], outlier_threshold_lower=params[1]))
        expected = upstream.quant_fn_nuq_recon(data.float(), bits=4, qchannel=axis,
            dynamicquantization=dynamic, include_sparse=True, outlier_mask=mask,
            maxval=params[0], minval=params[1], lut=params[2], first_few_fp16=-1).to(data.dtype)
        actual = nuq_reconstruct(upstream, data, params, dynamic=dynamic)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    global_key = nuq_reconstruct(upstream, data, params, dynamic=False)
    shards = [nuq_reconstruct(upstream, data[:, i:i+8],
              (params[0][i:i+8], params[1][i:i+8], params[2]), dynamic=False)
              for i in range(0, 32, 8)]
    torch.testing.assert_close(torch.cat(shards, -1), global_key, rtol=0, atol=0)
