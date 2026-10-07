"""Same-mask 16-bit kernel checks, including independent strides and empty tiles."""
import pathlib
import sys
import types
from unittest.mock import patch

import torch

ROOT = pathlib.Path(__file__).resolve().parent
package = types.ModuleType('veda_bf16_test')
package.__path__ = [str(ROOT / 'vendor' / 'veda')]
sys.modules[package.__name__] = package
from veda_bf16_test import backends
from veda_bf16_test.backends import base
from veda_bf16_test.core import reference, selection


def check_routing():
    for cc in ((7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)):
        info = types.SimpleNamespace(kind='cuda', family=f'sm{cc[0]}{cc[1]}', cc=cc)
        with patch.dict('os.environ', {'STAR7_VEDA_BACKEND': 'triton-bf16', 'STAR7_VEDA_REFERENCE': ''}):
            assert backends.candidates(info) == (['star7-cuda-int8'] if cc == (7, 5) else ['triton-bf16'])
        with patch.dict('os.environ', {'STAR7_VEDA_BACKEND': '', 'STAR7_VEDA_REFERENCE': ''}):
            assert backends.candidates(info) == (['star7-cuda-int8'] if cc == (7, 5) else ['triton-int8'])
    print('Opt-in BF16 routing, default INT8 routing and SM75 isolation: PASS')


def check_kernel():
    if not torch.cuda.is_available():
        print('16-bit sparse kernel: SKIP (no CUDA)')
        return
    from veda_bf16_test.kernels.sage import sparse_bf16
    device = torch.device('cuda')
    cc = torch.cuda.get_device_capability(device)
    dtype = torch.bfloat16 if cc >= (8, 0) else torch.float16
    _, _, _, mask, layout = base.selftest_problem(device, dtype)
    heads, slots = 7, layout.num_slots
    q, k, v = [torch.randn(slots, heads, 128, device=device, dtype=dtype) * .5 for _ in range(3)]
    for tensor in (q, k, v):
        tensor[layout.pad_slots] = 0
    mask = mask[:1].expand(heads, -1, -1).clone() & layout.kv_ok
    index, count = selection.tile_index_list(mask)
    expected = reference.block_sparse_attention(q, k, v, mask, layout)
    live = layout.slot_valid
    for strided in (False, True):
        if strided:
            q = q.repeat_interleave(2, dim=0)[::2]
            parent = torch.empty(slots, heads + 2, 128, device=device, dtype=dtype)
            parent[:, 1:-1] = k
            k = parent[:, 1:-1]
            v = v.transpose(0, 1).contiguous().transpose(0, 1)
            assert len({q.stride(), k.stride(), v.stride()}) == 3
        actual = sparse_bf16.attend(q, k, v, index, count, layout.valid_count)
        torch.testing.assert_close(actual[live], expected[live], atol=.008, rtol=.03)
        assert torch.isfinite(actual).all()
        print(f'SM{cc[0]}{cc[1]} {dtype} same-mask/padding/independent strides={strided}: PASS')
    empty = sparse_bf16.attend(q, k, v, index, torch.zeros_like(count), layout.valid_count)
    assert torch.equal(empty, torch.zeros_like(empty))
    print('Empty route produces finite zero output: PASS')
    scaled = sparse_bf16.attend(q, k, v * 4096, index, count, layout.valid_count)
    torch.testing.assert_close(scaled[live], expected[live] * 4096, atol=32, rtol=.03)
    assert torch.isfinite(scaled).all()
    print('Large V values with FP32 accumulation: PASS')
    if cc < (8, 0):
        print('BF16 SM80+ runtime: SKIP; local FP16 run checks kernel arithmetic only')


if __name__ == '__main__':
    check_routing()
    check_kernel()
