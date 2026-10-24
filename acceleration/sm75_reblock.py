"""SM75-only LMv2 cosine scoring with FP32-normalized FP16 TC operands.

Triton's dot lowering on SM75 uses scalar FMAs for the planner's thin score
matrices. Keep its normalization arithmetic, then use cuBLAS FP16 Tensor Cores
with FP32 output. No global precision flags or model patches are changed.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _normalize(X, INDICES, OUT, N: tl.constexpr, INDIRECT: tl.constexpr,
               ROWS: tl.constexpr):
    batch = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, 128)
    if INDIRECT:
        source_rows = tl.load(INDICES + batch * N + rows, rows < N, 0).to(tl.int64)
    else:
        source_rows = batch * N + rows.to(tl.int64)
    x = tl.load(X + source_rows[:, None] * 128 + columns[None, :], rows[:, None] < N, 0).to(tl.float32)
    norm = tl.maximum(libdevice.sqrt_rn(tl.sum(x * x, 1)), 1.e-12)
    normalized = (x / norm[:, None]).to(tl.float16)
    tl.store(OUT + (batch * N + rows[:, None].to(tl.int64)) * 128 + columns[None, :], normalized, rows[:, None] < N)


def cosine_scores(source, directions, indices=None):
    batch, n = indices.shape if indices is not None else source.shape[:2]
    count = directions.shape[1]
    normalized = torch.empty((batch, n, 128), device=source.device, dtype=torch.float16)
    _normalize[(triton.cdiv(n, 16), batch)](
        source, source if indices is None else indices, normalized,
        N=n, INDIRECT=indices is not None, ROWS=16, num_warps=4)
    padded = 16 if count <= 16 else 32
    direction_half = directions.to(torch.float16)
    if count != padded:
        direction_half = F.pad(direction_half, (0, 0, 0, padded - count))
    result = torch.bmm(normalized, direction_half.transpose(1, 2), out_dtype=torch.float32)
    return result[:, :, :count].contiguous()
