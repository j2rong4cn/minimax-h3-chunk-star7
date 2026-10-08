"""Snapshot ownership, memory guards and bounded native-row projection checks."""
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parents[1])]
import nodes
import sm80_diagnostics
import comfy.ops as ops
import comfy.quant_ops as quant_ops
import comfy.model_management as mm


def check_guard():
    x = SimpleNamespace(device=torch.device('cuda'), dtype=torch.bfloat16,
                        shape=(16257, 5376), element_size=lambda: 2)
    weight = SimpleNamespace(shape=(21504, 5376), numel=lambda: 21504 * 5376)
    linear = SimpleNamespace(weight=weight, out_features=21504)
    with patch.object(nodes, '_linear_can_reuse_weights', return_value=True), \
         patch.object(nodes, '_linear_quantization_mode', return_value='weight-only'), \
         patch.object(torch.cuda, 'get_device_capability', return_value=(12, 0)), \
         patch.object(torch.cuda, 'memory_reserved', return_value=0), \
         patch.object(torch.cuda, 'memory_allocated', return_value=0), \
         patch.object(torch.cuda, 'mem_get_info', return_value=(3 * 2**30, 16 * 2**30)):
        assert nodes._sm80_qkv_reuse_path(linear, x, 8192)
        assert not nodes._sm80_qkv_reuse_path(linear, x, 0)
        assert not nodes._sm80_qkv_reuse_path(linear, x, 16257)
        with patch.object(torch.cuda, 'mem_get_info', return_value=(256 * 2**20, 16 * 2**30)):
            assert not nodes._sm80_qkv_reuse_path(linear, x, 8192)
        with patch.object(nodes, '_linear_quantization_mode', return_value='input-and-weight'):
            assert not nodes._sm80_qkv_reuse_path(linear, x, 8192)
        linear._forward_hooks = {1: object()}
        assert not nodes._sm80_qkv_reuse_path(linear, x, 8192)
    print('SM80+ multi-chunk, memory reserve, input-quantization and forward-hook guards: PASS')


def check_probes():
    x = torch.randn(519, 16)
    projector = torch.nn.Linear(16, 384, bias=False)
    d = sm80_diagnostics.RunDiagnostics('test')
    capture = d.projection_capture(projector, x)
    for start in (0, 256, 512):
        end = min(519, start + 256)
        q, k, v = projector(x[start:end]).reshape(end-start, 3, 1, 128).unbind(1)
        capture(start, end, q, k, v)
    assert len(d.projection_checks) == 2 and d.projection_probed
    assert all(check['metrics'][0] < 1e-5 for check in d.projection_checks)
    assert d.projection_capture(projector, x) is None
    d._reset()
    capture = d.projection_capture(projector, x)
    capture(0, 256, *(torch.zeros(256, 1, 128) for _ in range(3)))
    assert d.projection_checks[0]['metrics'][0] > .99
    d._reset()
    assert callable(d.projection_capture(projector, x, input_quantized=True))
    assert d.projection_checks == [{'status': 'skipped-dynamic-input-scale'}]
    d._reset()
    for format_name in ('first', 'second'):
        projector.layout_type = format_name
        capture = d.projection_capture(projector, x)
        for start, end in ((0, 256), (512, 519)):
            parts = projector(x[start:end]).reshape(end-start, 3, 1, 128).unbind(1)
            capture(start, end, *parts)
    assert len(d.projection_checks) == 4
    projector.layout_type = 'third'
    assert d.projection_capture(projector, x) is None
    print('First/tail row probes, wrong finite projection detection, task reset and dynamic-scale exclusion: PASS')


def check_norm_probe():
    if not torch.cuda.is_available():
        return
    dtype = torch.bfloat16
    linear = ops.mixed_precision_ops(compute_dtype=dtype).Linear(512, 1152, bias=False, device='cuda', dtype=dtype)
    linear.weight = torch.nn.Parameter(torch.randn(1152, 512, device='cuda', dtype=dtype) * .02, requires_grad=False)
    weights = torch.rand(128, device='cuda', dtype=dtype) + .5
    attn = SimpleNamespace(heads=3, head_dim=128, qkv_proj=linear,
        q_norm=SimpleNamespace(weight=weights, eps=1e-6),
        k_norm=SimpleNamespace(weight=weights.clone(), eps=1e-6))
    x = torch.randn(519, 512, device='cuda', dtype=dtype)
    angles = torch.randn(519, 48, device='cuda')
    c, s = angles.cos(), angles.sin()
    freqs = torch.stack((c, -s, s, c), dim=-1).reshape(1, 519, 1, 48, 2, 2)
    nodes._CONFIG.update(verbose=False, reuse_mlp_weights=True, auto_halve_on_oom=False,
                         effective_qkv_chunk_tokens=256, node_id=None)
    d = sm80_diagnostics.RunDiagnostics('test')
    with patch.object(nodes, '_sm80_qkv_reuse_path', return_value=True):
        nodes._prepare_h3_qkv_chunked(attn, x, freqs, mm, quant_ops,
                                     raw_capture=d.projection_capture(linear, x))
    assert len(d.rope_checks) == 2
    for check in d.rope_checks:
        assert check['Q/K_relative_l2_max_abs'][:, 0].max() < .01
    # A faulty in-place norm/RoPE must be visible even if all output values are finite.
    d._reset()
    with patch.object(nodes, '_ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE', lambda q, k, *args, **kwargs: (q.zero_(), k.zero_())):
        nodes._prepare_h3_qkv_chunked(attn, x, freqs, mm, quant_ops,
                                     raw_capture=d.projection_capture(linear, x))
    assert d.rope_checks[0]['Q/K_relative_l2_max_abs'][:, 0].min() > .99
    print('Fused BF16 norm/partial RoPE vs independent FP32 sample reference, finite-wrong detection: PASS')


def check_snapshot(device):
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    operations = ops.mixed_precision_ops(compute_dtype=dtype)
    linear = operations.Linear(512, 1152, bias=False, device=device, dtype=dtype)
    linear.weight = torch.nn.Parameter(torch.randn(1152, 512, device=device, dtype=dtype) * .02, requires_grad=False)
    delta = torch.randn_like(linear.weight) * .001
    linear.weight_function = [lambda weight: weight + delta]
    x = torch.randn(519, 512, device=device, dtype=dtype)
    nodes._CONFIG.update(verbose=False, reuse_mlp_weights=True, auto_halve_on_oom=False,
                         effective_qkv_chunk_tokens=256, node_id=None)
    attn = SimpleNamespace(heads=3, head_dim=128, qkv_proj=linear,
                           q_norm=lambda value: value, k_norm=lambda value: value)
    with patch.object(nodes, '_sm80_qkv_reuse_path', return_value=False):
        expected = nodes._prepare_h3_qkv_chunked(attn, x, None, mm, quant_ops)
    with patch.object(nodes, '_sm80_qkv_reuse_path', return_value=True):
        actual = nodes._prepare_h3_qkv_chunked(attn, x, None, mm, quant_ops)
    assert attn._star7_qkv_weight_mode == 'resident-sm80-dense'
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    # Reusing a prepared snapshot must not retain a mutable cast-buffer scale.
    params = quant_ops.TensorWiseINT8Layout.Params(scale=torch.ones(1), orig_dtype=torch.float32,
                                                   orig_shape=(3, 16), convrot=False)
    source = quant_ops.QuantizedTensor(torch.ones(3, 16, dtype=torch.int8), 'TensorWiseINT8Layout', params)
    private = source.clone()
    source._qdata.zero_()
    source._params.scale.fill_(99)
    assert private._qdata.count_nonzero() == 48 and private._params.scale.item() == 1
    print(f'{device}: native prepared-weight LoRA parity and independent Q data/scale snapshot: PASS')


def check_mlp_snapshot():
    if not torch.cuda.is_available():
        return
    dtype = torch.bfloat16
    operations = ops.mixed_precision_ops(compute_dtype=dtype)
    x = torch.randn(519, 512, device='cuda', dtype=dtype)
    for quantized in (False, True):
        layers = []
        for width_in, width_out in ((512, 1024), (512, 512)):
            layer = operations.Linear(width_in, width_out, bias=False, device='cuda', dtype=dtype)
            weight = torch.randn(width_out, width_in, device='cuda', dtype=dtype) * .02
            if quantized:
                layer.weight = torch.nn.Parameter(quant_ops.QuantizedTensor.from_float(
                    weight, 'TensorWiseINT8Layout'), requires_grad=False)
                layer.layout_type = 'TensorWiseINT8Layout'
                layer.quant_format = 'int8_tensorwise'
            else:
                layer.weight = torch.nn.Parameter(weight, requires_grad=False)
                delta = torch.randn_like(weight) * .001
                layer.weight_function = [lambda value, delta=delta: value + delta]
            layers.append(layer)
        mlp = SimpleNamespace(fc1=layers[0], fc2=layers[1])
        # Override selection only while constructing snapshots. Native GPU
        # dispatch during arithmetic continues to see this machine's real GPU.
        with patch.object(torch.cuda, 'get_device_capability', return_value=(8, 9)):
            callers = nodes._sm80_resident_mlp_callers(mlp, x, 256)
            assert callers is not None
            assert nodes._sm80_resident_mlp_callers(mlp, x, 519) is None
            with patch.object(torch.cuda, 'mem_get_info', return_value=(0, 16 * 2**30)), \
                 patch.object(torch.cuda, 'memory_reserved', return_value=0), \
                 patch.object(torch.cuda, 'memory_allocated', return_value=0):
                assert nodes._sm80_resident_mlp_callers(mlp, x, 256) is None
            mlp.fc2._forward_hooks[1] = lambda *args: None
            assert nodes._sm80_resident_mlp_callers(mlp, x, 256) is None
            mlp.fc2._forward_hooks.clear()
        expected = torch.cat([ops.linear_input_act(mlp.fc2, mlp.fc1(x[start:start+256]), 'swiglu')
                              for start in range(0, 519, 256)])
        actual = torch.cat([callers[1](callers[0](x[start:start+256]), input_act='swiglu')
                            for start in range(0, 519, 256)])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        nodes._CONFIG.update(effective_mlp_chunk_tokens=256, verbose=False,
                             reuse_mlp_weights=True, auto_halve_on_oom=False)
        with patch.object(nodes, '_sm80_resident_mlp_callers', return_value=callers):
            assembled = nodes._run_chunked_h3_mlp(mlp, x)
        torch.testing.assert_close(assembled, expected, rtol=0, atol=0)
        print(f'SM80+ private MLP: quantized={quantized}, native SwiGLU/LoRA and tail assembly exact: PASS')


if __name__ == '__main__':
    check_guard()
    check_probes()
    check_norm_probe()
    check_mlp_snapshot()
    check_snapshot(torch.device('cpu'))
    if torch.cuda.is_available():
        check_snapshot(torch.device('cuda'))
