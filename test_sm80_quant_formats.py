"""Whole-input quantization scope and registered-format guards for SM80+ slabs."""
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parents[1])]
import nodes
import comfy.quant_ops as quant_ops


torch.manual_seed(73)
x = torch.randn(519, 64, dtype=torch.bfloat16)
x[-1, 0] = 100
for layout in ('TensorCoreFP8Layout', 'TensorCoreFP8E4M3Layout', 'TensorCoreFP8E5M2Layout'):
    full = quant_ops.QuantizedTensor.from_float(x, layout)
    chunks = [quant_ops.QuantizedTensor.from_float(x[start:start+256], layout)
              for start in range(0, 519, 256)]
    assert full._params.scale.item() == 1
    assert all(chunk._params.scale.item() == 1 for chunk in chunks)
    assert torch.equal(full._qdata.float(), torch.cat([chunk._qdata.float() for chunk in chunks]))
    linear = SimpleNamespace(layout_type=layout)
    assert nodes._fixed_fp8_input_scale(linear)
    with patch.object(quant_ops, 'get_layout_class', return_value=object()):
        assert not nodes._fixed_fp8_input_scale(linear)
print('Comfy FP8 E4M3/E5M2/default: scale=1, exact whole/slab quantization, registry override guard: PASS')


class CPUInputWithCudaMetadata:
    device = torch.device('cuda')
    dtype = torch.bfloat16
    shape = x.shape
    def __getitem__(self, index):
        return x[index]


linear = SimpleNamespace(layout_type='TensorCoreNVFP4Layout', input_scale=None)
with patch.object(nodes, '_linear_quantization_mode', return_value='input-and-weight'), \
     patch.object(nodes, '_linear_can_reuse_weights', return_value=True), \
     patch.object(torch.cuda, 'get_device_capability', return_value=(12, 0)):
    scale = nodes._sm80_qkv_global_input_scale(linear, CPUInputWithCudaMetadata(), 256)
    assert scale is not None
    with patch.object(quant_ops, 'get_layout_class', return_value=object()):
        assert nodes._sm80_qkv_global_input_scale(linear, CPUInputWithCudaMetadata(), 256) is None
full = quant_ops.QuantizedTensor.from_float(x, 'TensorCoreNVFP4Layout')
assert torch.equal(scale, full._params.scale)
old_first = quant_ops.QuantizedTensor.from_float(x[:256], 'TensorCoreNVFP4Layout')
assert not torch.equal(old_first._params.scale, scale)
parts = [quant_ops.QuantizedTensor.from_float(x[start:start+256], 'TensorCoreNVFP4Layout', scale=scale).dequantize()
         for start in range(0, 519, 256)]
assert torch.equal(full.dequantize(), torch.cat(parts))
print('NVFP4: old slab scope differs; bounded global calibration restores exact whole-input dequantized values: PASS')
