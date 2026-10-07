"""GPU attention grouping plus predictor/cache lifecycle regression checks."""
import pathlib
import sys
import types
from unittest.mock import patch
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
pkg = types.ModuleType('s7cache'); pkg.__path__ = [str(ROOT)]; sys.modules[pkg.__name__] = pkg
from s7cache.vendor.veda import nodes, comfy_patch, settings
from s7cache.vendor.veda.core import engine, plans, tiling, h3_layout, selection
from s7cache.vendor.veda.backends.torch_gather import TorchGatherBackend

nodes._BUNDLES.clear()
with patch.object(nodes.os, 'stat', return_value=types.SimpleNamespace(st_mtime_ns=1, st_size=12)) as stat, \
     patch.object(nodes.veda_bundle, 'load_bundle', side_effect=[object(), object()]) as load:
    first = nodes._bundle('predictor')
    assert nodes._bundle('predictor') is first and load.call_count == 1
    stat.return_value = types.SimpleNamespace(st_mtime_ns=2, st_size=12)
    assert nodes._bundle('predictor') is not first and load.call_count == 2

heads, dim, length = 12, 128, 512
bundle = types.SimpleNamespace(proj_q=torch.randn(1, heads, dim * 3, dim, dtype=torch.bfloat16),
                               proj_k=torch.randn(1, heads, dim * 3, dim, dtype=torch.bfloat16))
e = engine.VedaEngine(bundle, selection.Budget(ratio=.25), selection.Budget(ratio=.25),
                      TorchGatherBackend(), torch.device('cuda'))
spec = h3_layout.LayoutSpec(length, h3_layout.SpanSpec('video', 0, (4,8,16)), ())
plan = plans.TilePlan('test', (4,8,16), [tiling.TileShape(2,8,8)], [[0] * heads])
q,k,v = [torch.randn(length,heads,dim,device='cuda',dtype=torch.float16) for _ in range(3)]
normal = e.attention(q,k,v,0,spec,plan)
split = e.attention(q,k,v,0,spec,plan,head_chunks=12)
torch.testing.assert_close(normal,split,rtol=0,atol=0)
assert e.chunking['heads_per_chunk'] == 1 and e.chunking['workspace_bytes'] > 0
cached = next(iter(e._tile_layouts.values()))
e.attention(q,k,v,0,spec,plan)
assert next(iter(e._tile_layouts.values())) is cached

p = comfy_patch.VedaPatch(types.SimpleNamespace(num_layers=1,num_heads=heads,head_dim=dim),
                         settings.VedaSettings(selection.Budget(ratio=1),selection.Budget(ratio=1)),None)
opts={'minimax_head_chunks':5}
p.install(opts)
assert opts['minimax_head_chunks']==1 and opts['veda_held_head_chunks']==5
hits=[]
def dense(q,k,v,heads,**kw):
    hits.append(heads)
    return v if kw['skip_output_reshape'] else v.transpose(1,2).flatten(2)
for unshaped in (False,True):
    hits.clear()
    output=p.make_override(None)(dense,q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],
                                 heads,skip_reshape=True,skip_output_reshape=unshaped,transformer_options=opts)
    assert len(hits)==5 and sum(hits)==heads
    expected=v.transpose(0,1)[None] if unshaped else v.reshape(1,length,-1)
    assert torch.equal(output,expected)
p._engines['cuda']=e
with patch.object(p, '_summary', return_value=None):p.on_cleanup()
assert e._tile_layouts and e.stats.kept is None and not e.chunking
print('Predictor reload / workspace / head grouping / retained layout / dense fallback: PASS')
