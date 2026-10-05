import ast, importlib.util, pathlib, sys, types
import numpy as np
import torch
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
pkg = types.ModuleType('star7_chunk_test'); pkg.__path__ = [str(ROOT)]; sys.modules[pkg.__name__] = pkg
from star7_chunk_test.vendor.veda.backends import base
from star7_chunk_test.vendor.veda.backends.torch_gather import TorchGatherBackend
from star7_chunk_test.vendor.veda.comfy_patch import VedaPatch
from star7_chunk_test.vendor.veda.settings import VedaSettings
from star7_chunk_test.vendor.veda.core.selection import Budget
from star7_chunk_test.veda_star7 import NODE_CLASS_MAPPINGS
from star7_chunk_test.vendor.veda import backends, hardware, nodes as veda_nodes

node_cls = NODE_CLASS_MAPPINGS['Star7VedaSparseAttention']
schema = node_cls.INPUT_TYPES()
assert schema['optional']['enabled'][1]['default'] is True
assert schema['required']['generated_sparsity'][1]['display_name'] == '生成区域稀疏度'
model = object()
with patch.object(veda_nodes, '_predictor_path', side_effect=AssertionError('disabled predictor load')), \
     patch.object(veda_nodes.comfy_patch, 'apply', side_effect=AssertionError('disabled patch')):
    assert node_cls.execute(model, 'missing.safetensors', enabled=False).result[0] is model
for cc in [(7,5), (8,0), (8,6), (8,9), (9,0), (12,0)]:
    info = hardware.DeviceInfo('cuda', 0, 'test', cc, f'sm{cc[0]}{cc[1]}',
                               'test', 'windows', 'x86_64', None, False, '13.0')
    assert backends.candidates(info) == (['star7-cuda-int8'] if cc == (7,5) else ['triton-int8'])

base.self_test(TorchGatherBackend(), torch.device('cuda'), torch.float16)
assert 'Star7VedaSparseAttention' in NODE_CLASS_MAPPINGS
assert NODE_CLASS_MAPPINGS['Star7VedaSparseAttention'].INPUT_TYPES()['required']['model'][0] == 'MODEL'

# Invoke the real chunk CK forward through ComfyUI's attention decorator.
code = ast.parse((ROOT / 'nodes.py').read_text(encoding='utf-8'))
func = next(n for n in code.body if isinstance(n, ast.FunctionDef) and n.name == '_minimax_ck_int8_attention_forward')
q = torch.randn(1, 2, 384, 128, device='cuda', dtype=torch.float16)
ns = {'_prepare_h3_qkv_chunked': lambda *a, **kw: (q, q.clone(), q.clone()), '_log_h3_cuda_memory': lambda *a, **kw: None}
exec(compile(ast.Module(body=[func], type_ignores=[]), str(ROOT/'nodes.py'), 'exec'), ns)
veda = VedaPatch(types.SimpleNamespace(num_layers=50, num_heads=2, head_dim=128), VedaSettings(Budget(ratio=.1), Budget(ratio=.1)), None)
hit = []
def sparse(q, k, v, layout, options, skip_output_reshape, dense):
    hit.append(True)
    return q.transpose(1, 2).reshape(1, q.shape[2], -1)
veda._sparse = sparse
options = {'optimized_attention_override': veda.make_override(None), 'minimax_h3_layout': types.SimpleNamespace(seq_len=384), 'block_index': 0}
attn = types.SimpleNamespace(heads=2, out_proj=lambda x: x)
out = ns['_minimax_ck_int8_attention_forward'](attn, torch.randn(384,256,device='cuda',dtype=torch.float16), transformer_options=options)
assert hit and out.shape == (384,256)

# The unmodified H3 forward uses AttentionTensorContainer; verify that path too.
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention, ComfyAttention
source = ROOT.parents[1] / 'comfy/ldm/minimax/model.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Attention')
forward = next(n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
normal_ns = {'AttentionTensorContainer': AttentionTensorContainer, 'optimized_attention': optimized_attention}
exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), 'exec'), normal_ns)
normal_attn = types.SimpleNamespace(heads=2, head_dim=128,
    qkv_proj=lambda x: torch.cat([x,x,x],dim=-1), q_norm=lambda x:x, k_norm=lambda x:x,
    comfy_attention=ComfyAttention(), out_proj=lambda x:x)
hit.clear()
normal_out = normal_ns['forward'](normal_attn, torch.randn(384,256,device='cuda',dtype=torch.float16), transformer_options=options)
assert hit and normal_out.shape == (384,256)

spec = importlib.util.spec_from_file_location('star7_preview_quality_test', ROOT/'h3_live_preview.py')
preview = importlib.util.module_from_spec(spec); spec.loader.exec_module(preview)
assert '1024' in preview.MiniMaxH3LivePreviewStar7.INPUT_TYPES()['required']['preview_resolution'][0]
assert preview.preview_latent_size(48,84,1024)[1] * 16 == 1024
worker = preview.LatestPreviewWorker.__new__(preview.LatestPreviewWorker)
worker.fps=5;worker.node_id='test';worker.run_id='test';worker.quality=94;worker._sent_logged=True
captured=[]
class FakeImage:
    width=16; height=16
    def save(self, buf, **kw): captured.append(kw);buf.write(b'webp')
server=types.SimpleNamespace(instance=types.SimpleNamespace(send_sync=lambda *a: None, client_id=None))
with patch.object(preview.Image,'fromarray',return_value=FakeImage()), patch.object(preview,'PromptServer',server):
    worker._encode_and_send(np.zeros((2,16,16,3),dtype=np.uint8),1,8)
assert captured[0]['quality']==94 and captured[0]['format']=='WEBP'
assert captured[0]['duration']==200
print('Hardware dispatch / disabled identity / CUDA self-test / normal + CK -> VEDA / legacy registry / 1024 / quality transport: PASS')
