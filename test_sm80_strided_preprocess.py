"""Check strided pooling/quantization math and bounded per-run logging."""
import pathlib
import sys
from unittest.mock import patch

import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import sla_backend as sla
import sm80_diagnostics


def check_logging():
    diagnostics = sm80_diagnostics.RunDiagnostics('test')
    q = torch.empty(1, 7, 519, 128)
    with patch.object(torch.cuda, 'memory_allocated', return_value=1024), \
         patch.object(torch.cuda, 'get_device_capability', return_value=(12, 0)), \
         patch.object(torch.cuda, 'get_device_name', return_value='test GPU'), \
         patch.object(sm80_diagnostics._LOG, 'info') as log:
        for run in range(2):
            for _ in range(100):
                diagnostics.observe('CK->VEDA', (q, q, q), 8192, 'INT8')
            diagnostics.step('VEDA', 1.2)
            diagnostics.finite_outputs = 1
            diagnostics.finish()
            assert log.call_count == 2 * (run + 1)
            assert not diagnostics.calls and not diagnostics.qkv
            assert diagnostics.finite_outputs == 0
        diagnostics.finish()
        assert log.call_count == 4
    print('Two summary lines per run, no per-call logging, reset across consecutive tasks: PASS')


def check_gpu():
    if not torch.cuda.is_available() or sla.triton is None:
        print('Preprocessing GPU tests: SKIP')
        return
    # Mimic fused QKV slices; neither batch, head nor token strides are canonical.
    raw = torch.randn(4, 519, 3 * 9 * 128, device='cuda', dtype=torch.float16)
    x = raw[::2, :, :9 * 128].view(2, 519, 9, 128).transpose(1, 2)[:, 1:8]
    batch, heads, length, dim = x.shape
    assert not x.is_contiguous()
    for block in (16, 64, 128):
        blocks = sla.triton.cdiv(length, block)
        for subtract in (False, True):
            mean = x.mean(dim=-2, keepdim=True, dtype=torch.float32).half() if subtract else None
            placeholder = x if mean is None else mean
            strides = dict(stride_head=x.stride(1), stride_token=x.stride(2),
                           stride_batch=x.stride(0), heads=heads)
            actual = torch.empty(batch, heads, blocks, dim, device='cuda', dtype=torch.float32)
            sla._mean_pool_kernel[(blocks, batch * heads)](x, placeholder, actual, length, dim, block,
                subtract_mean=subtract, **strides, num_warps=4)
            expected = sla._mean_pool_torch(x, block, mean, torch.float32)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
            packed = torch.empty(x.shape, device='cuda', dtype=torch.int8)
            scales = torch.empty(batch, heads, blocks, device='cuda', dtype=torch.float32)
            sla._quantize_per_block_int8_kernel[(blocks, batch * heads)](
                x, placeholder, packed, scales, length, dim, block,
                multiplier=.127, subtract_mean=subtract, **strides, num_warps=4)
            reference, reference_scale = sla._quantize_torch(x, block, .127, mean)
            assert (packed.int() - reference.int()).abs().max() <= 1
            assert packed.is_contiguous()
            torch.testing.assert_close(scales, reference_scale, atol=1e-8, rtol=1e-6)
            contiguous = x.contiguous()
            golden = torch.empty_like(packed)
            golden_scales = torch.empty_like(scales)
            sla._quantize_per_block_int8_kernel[(blocks, batch * heads)](
                contiguous, placeholder, golden, golden_scales, length, dim, block,
                multiplier=.127, subtract_mean=subtract,
                stride_head=contiguous.stride(1), stride_token=contiguous.stride(2),
                stride_batch=contiguous.stride(0), heads=heads, num_warps=4)
            assert torch.equal(packed, golden) and torch.equal(scales, golden_scales)
            print(f'Strided B/H/N, partial tail, block={block}, subtract_mean={subtract}: PASS')


if __name__ == '__main__':
    check_logging()
    check_gpu()
