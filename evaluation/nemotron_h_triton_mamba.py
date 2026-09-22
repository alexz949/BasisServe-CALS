"""Triton chunk-scan drop-in for transformers' Nemotron-H Mamba2 mixer.

transformers 5.17 falls back to a pure-torch SSD scan when ``mamba_ssm`` is absent; that path
materialises a ``[chunks, chunk, chunk, heads, state]`` tensor and asks for >100 GiB at 4K tokens.
vLLM 0.29 ships Triton SSD kernels that need no compiled CUDA extension, so ``install()`` replaces
``modeling_nemotron_h.mamba2_chunk_scan`` with a wrapper around ``mamba_chunk_scan_combined_varlen``.
The depthwise causal conv and the single-token recurrent update keep their torch implementations,
which are cheap at any length.
"""
import torch
import transformers.models.nemotron_h.modeling_nemotron_h as modeling
from vllm.model_executor.layers.mamba.ops.ssd_combined import mamba_chunk_scan_combined_varlen
from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata

TORCH_CHUNK_SCAN = modeling.mamba2_chunk_scan


def triton_chunk_scan(hidden_states, dt, A, B, C, chunk_size, D=None, dt_bias=None, initial_states=None,
                      dt_softplus=False, dt_limit=(0.0, float('inf')), return_final_states=False, **kwargs):
    batch, length, heads, head_dim = hidden_states.shape
    assert batch == 1 and kwargs.get('z') is None
    device = hidden_states.device
    query_start_loc = torch.tensor([0, length], dtype=torch.int32, device=device)
    cu_chunk_seqlens, last_chunk_indices, seq_idx = compute_varlen_chunk_metadata(query_start_loc, chunk_size)
    if initial_states is not None and initial_states.ndim == 3:
        initial_states = initial_states.unsqueeze(0)
    out = torch.empty((length, heads, head_dim), dtype=hidden_states.dtype, device=device)
    final_state = mamba_chunk_scan_combined_varlen(
        hidden_states[0], dt[0], A, B[0], C[0], chunk_size=chunk_size, cu_seqlens=query_start_loc,
        cu_chunk_seqlens=cu_chunk_seqlens, last_chunk_indices=last_chunk_indices, seq_idx=seq_idx, out=out,
        D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, dt_softplus=dt_softplus, dt_limit=dt_limit,
        state_dtype=torch.float32 if initial_states is None else initial_states.dtype)
    output = out.unsqueeze(0)
    return (output, final_state) if return_final_states else output


def install():
    modeling.mamba2_chunk_scan = triton_chunk_scan


DT_LIMIT = (0.0, float('inf'))


def restore_dt_limit(model):
    """transformers 5.17 clamps every Mamba dt to at least ``config.time_step_min`` (1e-3), which the original
    model and vLLM never do; heads with tiny dt are the long-memory heads, so the clamp erases context beyond
    ~8K tokens. Reset the clamp on every mixer and return how many were patched."""
    patched = 0
    for layer in model.model.layers:
        if layer.block_type == 'linear_attention':
            layer.mixer.time_step_limit = DT_LIMIT
            patched += 1
    assert patched > 0
    return patched


def uninstall():
    modeling.mamba2_chunk_scan = TORCH_CHUNK_SCAN
