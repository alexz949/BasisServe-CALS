"""Split-K indexed attention over resident BF16 K and compact V."""
import torch
import triton
import triton.language as tl

@triton.jit
def _split_indexed_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    selected_ptr,
    output_ptr,
    lse_ptr,
    sequence_length,
    query_stride_batch,
    query_stride_head,
    query_stride_token,
    query_stride_feature,
    key_stride_batch,
    key_stride_head,
    key_stride_token,
    key_stride_feature,
    value_stride_batch,
    value_stride_head,
    value_stride_token,
    value_stride_feature,
    selected_stride_batch,
    selected_stride_head,
    selected_stride_token,
    output_stride_batch,
    output_stride_head,
    output_stride_token,
    output_stride_feature,
    SCALE: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    selected_count,
    BLOCK_QK: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
    BLOCK_SELECTED: tl.constexpr,
    SPLITS: tl.constexpr,
    PER_SPLIT: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    batch_index = row // QUERY_HEADS
    query_head = row % QUERY_HEADS
    kv_head = query_head // HEADS_PER_KV
    qk_offsets = tl.arange(0, BLOCK_QK)
    value_offsets = tl.arange(0, BLOCK_VALUE)
    query = tl.load(
        query_ptr
        + batch_index * query_stride_batch
        + query_head * query_stride_head
        + qk_offsets * query_stride_feature,
        mask=qk_offsets < QK_DIM,
        other=0.0,
    ).to(tl.float32)
    running_maximum = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_VALUE,), dtype=tl.float32)

    for selected_start in range(split * PER_SPLIT, (split + 1) * PER_SPLIT, BLOCK_SELECTED):
        selected_offsets = selected_start + tl.arange(0, BLOCK_SELECTED)
        token_ids = tl.load(
            selected_ptr
            + batch_index * selected_stride_batch
            + query_head * selected_stride_head
            + selected_offsets * selected_stride_token,
            mask=selected_offsets < selected_count,
            other=-1,
        )
        valid = (
            (selected_offsets < selected_count)
            & (token_ids >= 0)
            & (token_ids < sequence_length)
        )
        keys = tl.load(
            key_ptr
            + batch_index * key_stride_batch
            + kv_head * key_stride_head
            + token_ids[:, None] * key_stride_token
            + qk_offsets[None, :] * key_stride_feature,
            mask=valid[:, None] & (qk_offsets[None, :] < QK_DIM),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        block_maximum = tl.max(scores, axis=0)
        has_valid = tl.sum(valid.to(tl.int32), axis=0) > 0
        next_maximum = tl.maximum(running_maximum, block_maximum)
        next_maximum = tl.where(has_valid, next_maximum, running_maximum)
        previous_scale = tl.where(
            has_valid,
            tl.exp(running_maximum - next_maximum),
            1.0,
        )
        probabilities = tl.where(
            valid,
            tl.exp(scores - next_maximum),
            0.0,
        )
        values = tl.load(
            value_ptr
            + batch_index * value_stride_batch
            + kv_head * value_stride_head
            + token_ids[:, None] * value_stride_token
            + value_offsets[None, :] * value_stride_feature,
            mask=valid[:, None] & (value_offsets[None, :] < VALUE_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * values,
            axis=0,
        )
        running_sum = running_sum * previous_scale + tl.sum(
            probabilities,
            axis=0,
        )
        running_maximum = next_maximum

    tl.store(lse_ptr + row * SPLITS + split, tl.where(running_sum > 0, running_maximum + tl.log(running_sum), -float("inf")))
    result = tl.where(running_sum > 0.0, accumulator / running_sum, 0.0)
    tl.store(
        output_ptr
        + batch_index * output_stride_batch
        + query_head * output_stride_head
        + split * output_stride_token
        + value_offsets * output_stride_feature,
        result,
        mask=value_offsets < VALUE_DIM,
    )


@triton.jit
def _merge_kernel(partial, lse, output, R:tl.constexpr, BR:tl.constexpr, SPLITS:tl.constexpr):
    row=tl.program_id(0)
    splits=tl.arange(0,SPLITS)
    dims=tl.arange(0,BR)
    logs=tl.load(lse+row*SPLITS+splits)
    weights=tl.exp(logs-tl.max(logs,0))
    weights=weights/tl.sum(weights,0)
    values=tl.load(partial+(row*SPLITS+splits[:,None])*R+dims[None,:],dims[None,:]<R,0)
    out=tl.sum(values*weights[:,None],0)
    tl.store(output+row*R+dims,out,dims<R)


def split_indexed_attention(q,k,v,ids,*,scale):
    batch,heads,qt,d=q.shape
    rank=v.shape[-1]
    assert qt==1 and ids.shape[:2]==(batch,heads)
    assert q.dtype==k.dtype==v.dtype==torch.bfloat16
    assert q.is_cuda and ids.dtype==torch.int64 and heads%k.shape[1]==0
    assert k.shape[:3]==v.shape[:3] and k.shape[-1]==d
    splits=min(16,triton.next_power_of_2(triton.cdiv(ids.shape[-1],128)))
    per_split=triton.cdiv(triton.cdiv(ids.shape[-1],splits),32)*32
    partial=torch.empty((batch,heads,splits,rank),device=q.device,dtype=torch.float32)
    lse=torch.empty((batch,heads,splits),device=q.device,dtype=torch.float32)
    out=torch.empty((batch,heads,1,rank),device=q.device,dtype=q.dtype)
    with torch.cuda.device(q.device):
        _split_indexed_kernel[(batch*heads,splits)](q,k,v,ids,partial,lse,k.shape[2],
            *q.stride(),*k.stride(),*v.stride(),*ids.stride(),*partial.stride(),
            SCALE=scale,QUERY_HEADS=heads,HEADS_PER_KV=heads//k.shape[1],QK_DIM=d,VALUE_DIM=rank,
            selected_count=ids.shape[-1],BLOCK_QK=triton.next_power_of_2(d),BLOCK_VALUE=triton.next_power_of_2(rank),
            BLOCK_SELECTED=32,SPLITS=splits,PER_SPLIT=per_split,num_warps=4)
        _merge_kernel[(batch*heads,)](partial,lse,out,R=rank,BR=triton.next_power_of_2(rank),SPLITS=splits,num_warps=4)
    return out
