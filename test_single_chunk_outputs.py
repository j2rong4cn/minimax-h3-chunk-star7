"""Single-chunk ownership, lazy assembly and backend layout regression tests."""
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import nodes
import comfy.model_management as mm
import comfy.quant_ops


def check(device):
    nodes._CONFIG.update(verbose=False,reuse_mlp_weights=False,auto_halve_on_oom=True,
                         out_proj_memory_protection=True,out_proj_policy={},node_id=None)
    x=torch.randn(519,512,device=device,dtype=torch.float16)

    for chunk in (0,519,1024):
        nodes._CONFIG['effective_mlp_chunk_tokens']=chunk
        result=torch.randn_like(x)
        mlp=SimpleNamespace(_star7_reuse_mlp_input=True)
        assert nodes._run_chunked_h3_mlp(mlp,x,upstream_forward=lambda value:result) is result
        promoted=result.float()
        assert nodes._run_chunked_h3_mlp(mlp,x,upstream_forward=lambda value:promoted) is promoted

    # Enter out_proj's chunk loop even though its policy covers the full input.
    result=torch.randn_like(x)
    with patch.object(nodes,'_select_out_proj_policy',return_value={'mode':'chunk','chunk':1024}), \
         patch.object(nodes,'_sm75_h3_out_proj_fused_candidate',return_value=False):
        nodes._CONFIG['out_proj_policy']={}
        assert nodes._run_chunked_h3_out_proj(SimpleNamespace(out_features=512),x,lambda value:result) is result

    class Projection:
        result=None
        def __call__(self,value):
            self.result=torch.cat((value,value+1,value+2),dim=1)
            return self.result

    for layout in ('BHLD','BTHD'):
        for chunk in (0,519,1024,256):
            projection=Projection()
            attn=SimpleNamespace(heads=4,head_dim=128,qkv_proj=projection,
                                 q_norm=lambda value:value,k_norm=lambda value:value)
            nodes._CONFIG['effective_qkv_chunk_tokens']=chunk
            shape=(1,4,519,128) if layout=='BHLD' else (1,519,4,128)
            with patch.object(torch,'empty',wraps=torch.empty) as alloc:
                output=nodes._prepare_h3_qkv_chunked(attn,x,None,mm,comfy.quant_ops,output_layout=layout)
            assemblies=[call for call in alloc.call_args_list if call.args and call.args[0]==shape]
            assert len(assemblies)==(3 if chunk==256 else 0),(layout,chunk,len(assemblies))
            for i,part in enumerate(output):
                expected=(x+i).reshape(1,519,4,128)
                if layout=='BHLD':expected=expected.transpose(1,2)
                assert torch.equal(part,expected)
                if chunk!=256:
                    assert part.untyped_storage().data_ptr()==projection.result.untyped_storage().data_ptr()
                    assert not part.is_contiguous()
                else:assert part.is_contiguous()

        nodes._CONFIG['effective_qkv_chunk_tokens']=0
        converted=nodes._prepare_h3_qkv_chunked(attn,x,None,mm,comfy.quant_ops,
                                                output_layout=layout,output_dtype=torch.bfloat16)
        for i,part in enumerate(converted):
            expected=(x+i).reshape(1,519,4,128).to(torch.bfloat16)
            if layout=='BHLD':expected=expected.transpose(1,2)
            assert part.dtype==torch.bfloat16 and torch.equal(part,expected)

    # A fixed assembly-buffer OOM must not silently lower projection chunks.
    larger=x.repeat(3,1)
    shape=(1,4,larger.shape[0],128)
    nodes._CONFIG['effective_qkv_chunk_tokens']=512
    original_empty=torch.empty
    def fail_assembly(*args,**kwargs):
        if args and args[0]==shape:raise RuntimeError('CUDA out of memory')
        return original_empty(*args,**kwargs)
    attn=SimpleNamespace(heads=4,head_dim=128,qkv_proj=Projection(),
                         q_norm=lambda value:value,k_norm=lambda value:value)
    with patch.object(torch,'empty',side_effect=fail_assembly), \
         patch.object(nodes,'_clear_cuda_after_oom'):
        try:nodes._prepare_h3_qkv_chunked(attn,larger,None,mm,comfy.quant_ops)
        except RuntimeError as error:assert 'out of memory' in str(error)
        else:raise AssertionError('Expected assembly allocation failure')
    assert nodes._CONFIG['effective_qkv_chunk_tokens']==512

    if device.type=='cuda':
        import comfy_kitchen
        nodes._CONFIG['effective_qkv_chunk_tokens']=0
        attn=SimpleNamespace(heads=4,head_dim=128,qkv_proj=Projection(),
                             q_norm=lambda value:value,k_norm=lambda value:value)
        q,k,v=nodes._prepare_h3_qkv_chunked(attn,x,None,mm,comfy.quant_ops)
        assert not q.is_contiguous() and not v.is_contiguous()
        actual=comfy_kitchen.int8_attention(q,k,v)
        expected=comfy_kitchen.int8_attention(q.contiguous(),k.contiguous(),v.contiguous())
        assert torch.equal(actual,expected)
        # Exercise fused RMSNorm/RoPE with the same dtype boundary as H3.
        weight=torch.randn(1536,512,device=device,dtype=torch.float16)*.01
        attn.qkv_proj=lambda value:torch.nn.functional.linear(value,weight)
        attn.q_norm=SimpleNamespace(weight=torch.ones(128,device=device,dtype=torch.float16),eps=1e-6)
        attn.k_norm=SimpleNamespace(weight=torch.ones(128,device=device,dtype=torch.float16),eps=1e-6)
        angles=torch.randn(519,64,device=device,dtype=torch.float32)
        c,s=angles.cos(),angles.sin()
        freqs=torch.stack((c,-s,s,c),dim=-1).reshape(1,519,1,64,2,2)
        nodes._CONFIG['effective_qkv_chunk_tokens']=0
        q,k,v=nodes._prepare_h3_qkv_chunked(attn,x,freqs,mm,comfy.quant_ops)
        assert not q.is_contiguous()
        direct=comfy_kitchen.int8_attention(q,k,v)
        contiguous=comfy_kitchen.int8_attention(q.contiguous(),k.contiguous(),v.contiguous())
        assert torch.equal(direct,contiguous)
        import sla_backend,sol_backend
        for consume,kwargs in ((sla_backend.sparse_attention_consume,{'all_int8':True}),
                               (sol_backend.run_custom_consume,{'all_int8':True})):
            materialized=[part.contiguous() for part in (q,k,v)]
            output=consume(materialized,**kwargs).output
            assert output.shape==q.shape and torch.isfinite(output).all()
        import types
        pkg=types.ModuleType('single_veda');pkg.__path__=[str(ROOT)];sys.modules[pkg.__name__]=pkg
        from single_veda.vendor.veda.core import engine,selection,h3_layout,plans,tiling
        from single_veda.vendor.veda.backends.torch_gather import TorchGatherBackend
        bundle=SimpleNamespace(proj_q=torch.randn(1,4,384,128,dtype=torch.bfloat16),
                               proj_k=torch.randn(1,4,384,128,dtype=torch.bfloat16))
        e=engine.VedaEngine(bundle,selection.Budget(ratio=.5),selection.Budget(ratio=.5),
                            TorchGatherBackend(),device)
        spec=h3_layout.LayoutSpec(519,h3_layout.SpanSpec('video',7,(4,8,16)),())
        plan=plans.TilePlan('stride test',(4,8,16),[tiling.TileShape(2,8,8)],[[0]*4])
        actual=e.attention(q[0].transpose(0,1),k[0].transpose(0,1),v[0].transpose(0,1),0,spec,plan)
        reference=e.attention(q[0].transpose(0,1).contiguous(),k[0].transpose(0,1).contiguous(),
                              v[0].transpose(0,1).contiguous(),0,spec,plan)
        assert torch.equal(actual,reference)
        print('CK/SLA/Sol native boundary and VEDA strided-input checks: PASS')
    print(f'Single MLP/out_proj identity, QKV storage, multi-chunk assembly and CK stride: {device} PASS')


check(torch.device('cpu'))
if torch.cuda.is_available():check(torch.device('cuda'))
