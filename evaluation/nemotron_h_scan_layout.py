"""Pack dt before scan kernels whose stride arithmetic uses 32-bit offsets."""
from functools import wraps
import torch
from evaluation.chunked_prefill_mlp import ChunkedTokenwise


def install_contiguous_dt_scan(native):
    original = native.mamba2_chunk_scan

    @wraps(original)
    def scan(x, dt, *args, **kwargs):
        if x.shape[1] <= 8192:
            return original(x, dt.contiguous(), *args, **kwargs)
        A, B, C, *remaining = args
        assert kwargs.get('seq_idx') is None and kwargs.get('cu_seqlens') is None
        assert kwargs.get('z') is None and not kwargs.get('return_varlen_states', False)
        chunk_size = kwargs.get('chunk_size', remaining[0] if remaining else None)
        assert chunk_size is not None and 8192 % chunk_size == 0
        state = kwargs.get('initial_states')
        options = dict(kwargs, return_final_states=True)
        output = torch.empty_like(x)
        for start in range(0, x.shape[1], 8192):
            stop = start + 8192
            options['initial_states'] = state
            block, state = original(x[:, start:stop], dt[:, start:stop].contiguous(),
                A, B[:, start:stop], C[:, start:stop], *remaining, **options)
            output[:, start:stop].copy_(block)
            del block
        return (output, state) if kwargs.get('return_final_states', False) else output

    native.mamba2_chunk_scan = scan


def install_chunked_mamba_norms(model):
    for layer in model.model.layers:
        if layer.block_type == 'linear_attention':
            layer.mixer.norm = ChunkedTokenwise(layer.mixer.norm)
