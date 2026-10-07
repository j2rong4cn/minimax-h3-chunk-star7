"""64-row tensor-core attention consuming the existing VEDA tile selection.

Uses independent tensor strides and a single-stage KV loop. QK/PV inputs stay
16-bit; online softmax and dot accumulation are FP32.
"""
import torch
import triton
import triton.language as tl

from .sparse_int8 import key_blocks


@triton.jit
def _attention_kernel(Q, K, V, Index, Count, Valid, Out,
                      qn: tl.constexpr, qh: tl.constexpr,
                      kn: tl.constexpr, kh: tl.constexpr,
                      vn: tl.constexpr, vh: tl.constexpr,
                      on: tl.constexpr, oh: tl.constexpr,
                      ih: tl.constexpr, iq: tl.constexpr,
                      n_queries: tl.constexpr, scale: tl.constexpr):
    query_block, head = tl.program_id(0), tl.program_id(1)
    query_tile = query_block // 2
    rows = query_block * 64 + tl.arange(0, 64)
    columns = tl.arange(0, 64)
    dims = tl.arange(0, 128)
    q = tl.load(Q + rows[:, None] * qn + head * qh + dims[None, :])
    maximum = tl.full((64,), -1.0e30, tl.float32)
    denominator = tl.zeros((64,), tl.float32)
    accumulator = tl.zeros((64, 128), tl.float32)
    kept = tl.load(Count + head * n_queries + query_tile)
    for position in tl.range(0, kept):
        block = tl.load(Index + head * ih + query_tile * iq + position)
        valid = tl.load(Valid + block)
        key_rows = block * 64 + columns
        k = tl.load(K + key_rows[None, :] * kn + head * kh + dims[:, None])
        scores = tl.dot(q, k, out_dtype=tl.float32) * scale
        scores = tl.where(columns[None, :] < valid, scores, -float('inf'))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        probability = tl.exp2(scores - next_maximum[:, None])
        rescale = tl.exp2(maximum - next_maximum)
        v = tl.load(V + key_rows[:, None] * vn + head * vh + dims[None, :])
        accumulator = accumulator * rescale[:, None]
        accumulator += tl.dot(probability.to(v.dtype), v, out_dtype=tl.float32)
        denominator = denominator * rescale + tl.sum(probability, 1)
        maximum = next_maximum
    result = accumulator / tl.where(denominator > 0, denominator, 1)[:, None]
    tl.store(Out + rows[:, None] * on + head * oh + dims[None, :], result)


def attend(q, k, v, index, count, valid_count):
    slots, heads, dim = q.shape
    if (k.shape != q.shape or v.shape != q.shape or slots % 128 or dim != 128
            or q.dtype not in (torch.bfloat16, torch.float16)
            or k.dtype != q.dtype or v.dtype != q.dtype
            or k.device != q.device or v.device != q.device
            or any(value.stride(-1) != 1 for value in (q, k, v))):
        raise ValueError('VEDA 16-bit sparse attention requires equal NHD tensors, D=128, padded N and contiguous head dimensions')
    blocks, counts, valid = key_blocks(index, count, valid_count, 64)
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _attention_kernel[(slots // 64, heads)](
        q, k, v, blocks, counts, valid, output,
        qn=q.stride(0), qh=q.stride(1), kn=k.stride(0), kh=k.stride(1),
        vn=v.stride(0), vh=v.stride(1), on=output.stride(0), oh=output.stride(1),
        ih=blocks.stride(0), iq=blocks.stride(1), n_queries=slots // 128,
        scale=128 ** -.5 * 1.4426950408889634, num_warps=4, num_stages=1)
    return output
