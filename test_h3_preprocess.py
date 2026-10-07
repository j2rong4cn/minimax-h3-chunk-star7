"""GPU parity and peak-memory checks for streaming query and fused FP16 SwiGLU."""
import torch
import comfy_kitchen
import h3_preprocess as pre

assert pre.available(torch.device('cuda'))
assert pre.compact_available(torch.device('cuda'))
torch.manual_seed(7)
for length in (1025, 1305, 4096):
    q, k, v = [torch.randn(1, 4, length, 128, device='cuda', dtype=torch.float16) for _ in range(3)]
    reference = comfy_kitchen.prequantize_int8_attention(q, k, v)
    compact = pre.CompactQK(q.shape, q.device)
    for start in range(0, length, 256):
        compact.write(start, q[:, :, start:start + 256])
    assert torch.equal(compact.data, reference.q), f'Q values differ at {length}'
    assert torch.equal(compact.scale, reference.q_scale), f'Q scales differ at {length}'
    packed = compact.prequantize_value(k, v)
    assert torch.equal(packed.k, reference.k)
    assert torch.equal(packed.k_scale, reference.k_scale)
    actual = comfy_kitchen.int8_attention_from_prequantized(packed)
    expected = comfy_kitchen.int8_attention_from_prequantized(reference)
    assert torch.equal(actual, expected), f'Attention output differs at {length}'
    positions = torch.arange(9, device=k.device) * (length - 1) // 8
    compact.prepare_key(k.index_select(2, positions))
    for start in range(0, length, 256):
        compact.write_key(start, k[:, :, start:start + 256])
    packed = compact.prequantize_value(None, v)
    assert torch.equal(packed.k, reference.k), f'Global anchored K differs at {length}'
    assert torch.equal(packed.k_scale, reference.k_scale)
    assert torch.equal(comfy_kitchen.int8_attention_from_prequantized(packed), expected)
    assert compact.data.nbytes + compact.scale.nbytes < q.nbytes * .57
print('Compact Q / global K / CK attention bitwise parity: PASS')

for rows, width in ((13, 256), (256, 14336), (2048, 14336)):
    x = torch.randn(rows, width * 2, device='cuda', dtype=torch.float16) * 5
    gate, up = x.chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate.float()) * up.float() / 256).half()
    actual = pre.swiglu_scaled(x)
    assert torch.allclose(actual, expected, rtol=.001, atol=1e-5)
    print(f'Fused SwiGLU {rows}x{width}: max error {(actual-expected).abs().max().item():.6g}')
print('Fused FP16 SwiGLU numerical parity: PASS')

# Execute the production QKV projection loop, including the nine global
# anchor samples, rather than testing only prepared random Q/K tensors.
import nodes
import comfy.model_management as mm
import comfy.quant_ops
from types import SimpleNamespace
nodes._CONFIG.update(effective_qkv_chunk_tokens=256, verbose=False, reuse_mlp_weights=False)
attn = SimpleNamespace(heads=4, head_dim=128,
    qkv_proj=torch.nn.Linear(512, 1536, device='cuda', dtype=torch.float16),
    q_norm=torch.nn.RMSNorm(128, eps=1e-6, device='cuda', dtype=torch.float16),
    k_norm=torch.nn.RMSNorm(128, eps=1e-6, device='cuda', dtype=torch.float16))
x = torch.randn(1305, 512, device='cuda', dtype=torch.float16)
q, k, v = nodes._prepare_h3_qkv_chunked(attn, x, None, mm, comfy.quant_ops)
reference = comfy_kitchen.prequantize_int8_attention(q, k, v)
compact, absent_key, actual_v = nodes._prepare_h3_qkv_chunked(
    attn, x, None, mm, comfy.quant_ops, compact_query=True)
assert absent_key is None
actual = compact.prequantize_value(absent_key, actual_v)
assert torch.equal(actual.q, reference.q)
assert torch.equal(actual.k, reference.k)
assert torch.equal(comfy_kitchen.int8_attention_from_prequantized(actual),
                   comfy_kitchen.int8_attention_from_prequantized(reference))
print('Production QKV loop / sampled global K anchor: PASS')

# Force the pressure decision in a small test; compare the complete CK forward
# and ensure an attention override still receives the floating tensors.
from unittest.mock import patch
attn.out_proj = torch.nn.Linear(512,512,device='cuda',dtype=torch.float16)
with patch.object(pre, 'memory_pressure', return_value=False):
    expected = nodes._minimax_ck_int8_attention_forward(attn,x.clone())
with patch.object(pre, 'memory_pressure', return_value=True):
    actual = nodes._minimax_ck_int8_attention_forward(attn,x.clone())
assert torch.equal(actual,expected)
with patch.object(pre, 'memory_pressure', side_effect=AssertionError('override must bypass compact policy')):
    def override(fn,q,k,v,heads,**kwargs):
        assert q.dtype==torch.float16 and k.dtype==torch.float16
        return fn(q,k,v,heads,**kwargs)
    nodes._minimax_ck_int8_attention_forward(attn,x.clone(),transformer_options={'optimized_attention_override':override})
print('Complete CK forward / override isolation: PASS')

# The H3 path uses fused RMSNorm + split-half RoPE. Verify sampled-anchor
# preparation at the real rotary boundary and on a nondefault CUDA stream.
freqs = torch.randn(1,1305,1,64,2,2,device='cuda',dtype=torch.float32)
rotary_attn = SimpleNamespace(heads=attn.heads,head_dim=attn.head_dim,
    qkv_proj=lambda value:torch.nn.functional.linear(value,attn.qkv_proj.weight.detach(),attn.qkv_proj.bias.detach()),
    q_norm=SimpleNamespace(weight=attn.q_norm.weight.detach(),eps=attn.q_norm.eps),
    k_norm=SimpleNamespace(weight=attn.k_norm.weight.detach(),eps=attn.k_norm.eps))
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    q,k,v = nodes._prepare_h3_qkv_chunked(rotary_attn,x,freqs,mm,comfy.quant_ops)
    compact,_,actual_v = nodes._prepare_h3_qkv_chunked(rotary_attn,x,freqs,mm,comfy.quant_ops,compact_query=True)
    actual = compact.prequantize_value(None,actual_v)
    reference = comfy_kitchen.prequantize_int8_attention(q,k,v)
    assert torch.equal(actual.q,reference.q) and torch.equal(actual.k,reference.k)
stream.synchronize()
print('Nondefault CUDA stream: PASS')

if '--model' in __import__('sys').argv:
    import json
    import sys
    import star7_w4a8
    from safetensors import safe_open
    from comfy_kitchen.tensor import QuantizedTensor, AsymW4A8Int8Layout
    path = sys.argv[sys.argv.index('--model') + 1]
    prefix = 'blocks.10.mlp.fc2'
    with safe_open(path, framework='pt', device='cpu') as file:
        tensors = {name:file.get_tensor(prefix + '.' + name).cuda() for name in
                   ('weight', 'weight_s_rel', 'weight_s_channel', 'weight_codebook')}
        config = json.loads(file.metadata()['_quantization_metadata'])['layers'][prefix]
    rows, width = tensors['weight'].shape
    width *= 2
    params = AsymW4A8Int8Layout.Params(scale=tensors['weight_s_rel'],
        s_channel=tensors['weight_s_channel'], codebook=tensors['weight_codebook'],
        group_size=config['group_size'], convrot_groupsize=256,
        orig_dtype=torch.float16, orig_shape=(rows,width))
    weight = QuantizedTensor(tensors['weight'], 'AsymW4A8Int8Layout', params)
    linear = SimpleNamespace(quant_format='asym_w4a8_int8')
    x = torch.randn(256, width * 2, device='cuda', dtype=torch.float16) * 3
    actual = star7_w4a8.try_forward(linear,x,weight,None,input_act='swiglu')
    from unittest.mock import patch
    with patch.object(star7_w4a8.h3_preprocess, 'swiglu_scaled', return_value=None):
        expected = star7_w4a8.try_forward(linear,x,weight,None,input_act='swiglu')
    error = (actual - expected).float().norm() / expected.float().norm()
    assert float(error) < .001, float(error)
    print(f'Actual W4A8 group-{config["group_size"]} fc2: relative L2 {float(error):.6g}, native path PASS')
