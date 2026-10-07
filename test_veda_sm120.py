"""SM120 VEDA numerical checks; no user models or diagnostic logs required."""
import pathlib
import sys
import types
from unittest.mock import patch
import torch

ROOT = pathlib.Path(__file__).resolve().parent
package = types.ModuleType('veda120_test')
package.__path__ = [str(ROOT / 'vendor' / 'veda')]
sys.modules[package.__name__] = package
from veda120_test import backends, hardware
from veda120_test.backends import base, triton_int8
from veda120_test.backends.sampled_reference import sampled_outputs, numerical_error
from veda120_test.backends.torch_reference import TorchReferenceBackend
from veda120_test.core import reference


def problem(device='cpu', heads=7):
    _, _, _, mask, layout = base.selftest_problem(torch.device(device), torch.bfloat16)
    q, k, v = [torch.randn(layout.num_slots, heads, 128, device=device,
                          dtype=torch.bfloat16) * 0.3 for _ in range(3)]
    for value in (q, k, v): value.index_fill_(0, layout.pad_slots, 0)
    mask = mask[:1].expand(heads, -1, -1).clone()
    return q, k, v, mask, layout


def check_cpu():
    q, k, v, mask, layout = problem()
    expected = reference.block_sparse_attention(q, k, v, mask, layout)
    actual = TorchReferenceBackend().attend(q, k, v, mask, layout)
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
    got, want = sampled_outputs(q, k, v, mask, layout.valid_count, actual)
    relative, _, cosine = numerical_error(got, want)
    assert relative < 0.01 and cosine > 0.999
    assert numerical_error(torch.zeros_like(got), want)[0] > 0.9
    calls = []
    broken = False
    def attend(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs['fp32_pv'] is True
        if broken or kwargs['num_stages'] == 3:
            return torch.zeros_like(expected)
        return expected.clone()
    backend = triton_int8.TritonInt8Backend('SM120', fp32_pv=True)
    with patch.object(triton_int8, '_kernel', return_value=types.SimpleNamespace(attend=attend)):
        result = backend.attend(q, k, v, mask, layout)
        assert backend.stages == 1 and torch.equal(result, expected)
        assert len(calls) == 2
        broken = True
        backend.begin_attention()
        try:
            backend.attend(q, k, v, mask, layout)
        except base.BackendUnavailable as error:
            assert 'numerical check failed' in str(error)
        else:
            raise AssertionError('Changed finite-but-wrong activations must be rejected')
    info = hardware.DeviceInfo('cuda', 0, 'test', (12, 0), 'sm120',
        'test', 'windows', 'x86_64', None, False, '13.0')
    with patch.dict('os.environ', {'STAR7_VEDA_REFERENCE': '1'}):
        assert backends.candidates(info) == ['torch-reference']
        info = types.SimpleNamespace(kind='cuda', family='sm75', cc=(7, 5))
        assert backends.candidates(info) == ['star7-cuda-int8']
    bounded = triton_int8.TritonInt8Backend('SM86', audit=True)
    def normal(*args, **kwargs):
        assert kwargs['fp32_pv'] is False and kwargs['num_stages'] == 3
        return expected.clone()
    with patch.object(triton_int8, '_kernel', return_value=types.SimpleNamespace(attend=normal)):
        for _ in range(20):
            bounded.begin_attention()
            bounded.attend(q, k, v, mask, layout)
        assert len(bounded.checks) == 1 and bounded.checks[0]['sampled_heads'] == 2
        bounded.begin_run()
        assert not bounded.checks
        bounded.attend(q, k, v, mask, layout)
        assert len(bounded.checks) == 1 and not bounded.fp32_pv
        for layer in (1, 49, 0, 1, 49, 0):
            bounded.begin_attention(layer)
            bounded.attend(q, k, v, mask, layout)
        assert len(bounded.checks) == 2
    print('Default INT8 arithmetic, bounded per-run audit and consecutive-run reset: PASS')
    print('Same-mask FP32 reference, padding/7 heads, finite-wrong rejection and SM75 isolation: PASS')


def check_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0):
        print('SM120 native numerical tests: SKIP (no SM120 device)')
        return
    backend = triton_int8.TritonInt8Backend('SM120', fp32_pv=True)
    q, k, v, mask, layout = problem('cuda')
    base.self_test(backend, torch.device('cuda'))
    for factor in (1.0, 4096.0):
        backend.begin_attention()
        out = backend.attend(q, k, v * factor, mask, layout)
        got, want = sampled_outputs(q, k, v * factor, mask, layout.valid_count, out)
        error, maximum, cosine = numerical_error(got, want)
        assert error <= 0.12 and cosine >= 0.98
        print(f'SM120 PV scale={factor:g}: relative-L2={error:.5f}, max-abs={maximum:.5g}, cosine={cosine:.5f}')
    # A long real-sized NHD stride with seven heads, padding and ragged routes.
    from veda120_test.kernels.sage import sparse_int8
    slots, heads, tiles = 32768, 7, 256
    q, k, v = [torch.randn(slots, heads, 128, device='cuda', dtype=torch.bfloat16) * 0.3 for _ in range(3)]
    valid = torch.full((tiles,), 128, device='cuda', dtype=torch.int32)
    valid[3::11] = 37
    offsets = torch.arange(128, device='cuda')
    live = (offsets[None, :] < valid[:, None]).flatten()
    for value in (q, k, v): value[~live] = 0
    mask = torch.eye(tiles, device='cuda', dtype=torch.bool)[None].expand(heads, -1, -1).clone()
    mask[:, :, ::13] = True
    from veda120_test.core import selection
    index, count = selection.tile_index_list(mask)
    for stages in (3, 1):
        out = sparse_int8.attend(q, k, v, index, count, valid, fp32_pv=True, num_stages=stages)
        got, want = sampled_outputs(q, k, v, mask, valid, out)
        error, maximum, cosine = numerical_error(got, want)
        assert error <= 0.12 and cosine >= 0.98, (stages, error, maximum, cosine)
        print(f'SM120 32768 slots / 7 heads / stages={stages}: relative-L2={error:.5f}, cosine={cosine:.5f}')


if __name__ == '__main__':
    check_cpu()
    check_gpu()
