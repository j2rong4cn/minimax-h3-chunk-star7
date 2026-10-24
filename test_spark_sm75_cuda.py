"""Real SM75 CUDA validation; no checkpoints or queued workflows are used."""
import importlib
import json
from pathlib import Path
import sys
import types
import unittest

import torch

ROOT=Path(__file__).resolve().parent
pkg=types.ModuleType('star7_spark_cuda_tests'); pkg.__path__=[str(ROOT)]; sys.modules[pkg.__name__]=pkg
kernel=importlib.import_module(pkg.__name__+'.acceleration.sm75_kernel')
# Reuse the read-only native + Director idle check.
sys.path.insert(0,str(ROOT))
from bench_unified_acceleration import assert_idle


def summary_reference(k,v,anchor):
    aks=[];avs=[];lms=[]
    for kk,vv in zip(k.split(64,2),v.split(64,2)):
        logits=(anchor.unsqueeze(2)*kk.float()).sum(-1)*128**-.5
        probability=logits.softmax(-1)
        ak=(probability.unsqueeze(-1)*kk.float()).sum(2).half()
        av=(probability.unsqueeze(-1)*vv.float()).sum(2).half()
        lm=logits.logsumexp(-1)-(anchor*ak.float()).sum(-1)*128**-.5
        aks.append(ak);avs.append(av);lms.append(lm)
    return torch.stack(aks,2),torch.stack(avs,2),torch.stack(lms,2)


def dequantize(x,block):
    quantized,scale=kernel.quantize(x,block)
    return quantized.float()*scale.repeat_interleave(block,2)[:,:,:x.shape[2]].unsqueeze(-1)


def attention_reference(q,k,v,mask,ak,av,lm):
    qd=dequantize(q,16);kd=dequantize(k,64);ad=dequantize(ak,64)
    leaves=mask.shape[-1]; n=q.shape[2]
    exact=mask.bool().repeat_interleave(64,2)[:,:,:n]
    token_mask=exact.repeat_interleave(64,3)[:,:,:,:n]
    scores=(qd@kd.transpose(-1,-2))*128**-.5
    scores=scores.masked_fill(~token_mask,-torch.inf)
    approx=(qd@ad[:,:,:leaves].transpose(-1,-2))*128**-.5+lm[:,:,:leaves].unsqueeze(2)
    approx=approx.masked_fill(exact,-torch.inf)
    probabilities=torch.cat((scores,approx),-1).softmax(-1)
    values=torch.cat((v.float(),av[:,:,:leaves].float()),2)
    return probabilities@values


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability()==(7,5),'Actual SM75 required')
class SparkCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert_idle();torch.manual_seed(75)

    def test_summaries(self):
        for n in (1,31,63,64,65,129,512):
            for scale in (.1,1.,8.):
                with self.subTest(n=n,scale=scale):
                    k=torch.randn(1,2,n,128,device='cuda',dtype=torch.float16)*scale
                    v=torch.randn_like(k)
                    anchor=torch.randn(1,2,128,device='cuda')*scale
                    actual=kernel.summaries(k,v,anchor)
                    reference=summary_reference(k,v,anchor)
                    for result,expected in zip(actual,reference):
                        result=result[:,:,:expected.shape[2]]
                        self.assertTrue(torch.isfinite(result).all())
                        torch.testing.assert_close(result.float(),expected.float(),rtol=.003,atol=.003)
                    torch.cuda.synchronize()

    def test_consumer(self):
        for n in (1,63,64,65,129,512):
            for density in (0.,.3,1.):
                with self.subTest(n=n,density=density):
                    q=torch.randn(1,2,n,128,device='cuda',dtype=torch.float16)
                    k=torch.randn_like(q);v=torch.randn_like(q)
                    leaves=(n+63)//64
                    mask=(torch.rand(1,2,leaves,leaves,device='cuda')<density).to(torch.uint8)
                    ak,av,lm=kernel.summaries(k,v,q.float().mean(2).contiguous())
                    result=kernel.consume(q,k,v,mask,ak,av,lm)
                    expected=attention_reference(q,k,v,mask,ak,av,lm)
                    self.assertTrue(torch.isfinite(result).all())
                    torch.testing.assert_close(result.float(),expected,rtol=.01,atol=.003)
                    if density==1.:
                        dense=torch.nn.functional.scaled_dot_product_attention(q,k,v)
                        relative=(result.float()-dense.float()).norm()/dense.float().norm()
                        self.assertLess(float(relative),.02)  # INT8 QK approximation budget
                        self.assertLess(float((result.float()-dense.float()).abs().max()),.03)
                    torch.cuda.synchronize()

    def test_fast_int8_consumer_and_exact_suffix(self):
        import comfy_kitchen
        for n in (1,63,64,65,129,512):
            for density in (0.,.3,1.):
                with self.subTest(n=n,density=density):
                    q=torch.randn(1,2,n,128,device='cuda',dtype=torch.float16)
                    k=torch.randn_like(q);v=torch.randn_like(q);leaves=(n+63)//64
                    mask=(torch.rand(1,2,leaves,leaves,device='cuda')<density).to(torch.uint8)
                    prefix=min(n,64);mask[:,:,1:]=1
                    ak,av,lm=kernel.summaries(k,v,q.float().mean(2).contiguous())
                    actual=kernel.consume_fast(q,k,v,mask,ak,av,lm,query_tokens=prefix)
                    expected=kernel.consume(q,k,v,mask,ak,av,lm)
                    relative=(actual.float()-expected.float()).norm()/expected.float().norm()
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertLess(float(relative),.02)
                    if n>64:
                        tail=comfy_kitchen.int8_attention(q[:,:,64:].contiguous(),k,v)
                        torch.testing.assert_close(actual[:,:,64:],tail,rtol=0,atol=0)

    def test_reject_wrong_dtype(self):
        x=torch.zeros((1,1,64,128),device='cuda',dtype=torch.float32)
        with self.assertRaises(ValueError):kernel.validate(x,x,x)

    def test_global_anchor_mass_is_preserved(self):
        # At q == the global anchor, weighted summaries reproduce the exact
        # block partition function and weighted value, even with ZERO exact
        # video blocks. This distinguishes Reweight from ordinary Sol means.
        n=193
        anchor=torch.randn(1,2,128,device='cuda')
        q=anchor.unsqueeze(2).expand(-1,-1,n,-1).half().contiguous()
        k=torch.randn_like(q);v=torch.randn_like(q);leaves=(n+63)//64
        mask=torch.zeros((1,2,leaves,leaves),device='cuda',dtype=torch.uint8)
        ak,av,lm=kernel.summaries(k,v,q.float().mean(2).contiguous())
        actual=kernel.consume(q,k,v,mask,ak,av,lm)
        expected=torch.nn.functional.scaled_dot_product_attention(q,k,v)
        relative=(actual.float()-expected.float()).norm()/expected.float().norm()
        self.assertLess(float(relative),.02)

    def test_reblock_cached_plan_without_cuda_graph_keeps_exact_conditioning_rows(self):
        from sla_backend import _configure_triton_cache
        _configure_triton_cache()
        spark=importlib.import_module(pkg.__name__+'.acceleration.sm75_spark')
        q=torch.randn(1,2,1101,128,device='cuda',dtype=torch.float16)
        k=torch.randn_like(q);v=torch.randn_like(q);cache={}
        # Full Top-K must match dense INT8-QK within its known error budget.
        for _ in range(3):
            actual=spark.attention(q,k,v,1024,(4,16,16),1.,cache=cache)
        expected=torch.nn.functional.scaled_dot_product_attention(q,k,v)
        self.assertLess(float((actual-expected).float().norm()/expected.float().norm()),.02)
        plans=[value for value in cache.values() if hasattr(value,'graph_active')]
        self.assertFalse(plans[0].graph_active)
        # Change all Q/K values after cached planning. Output must use new data and
        # conditioning queries remain fully exact at a sparse video budget.
        q=torch.randn_like(q);k=torch.randn_like(k)
        actual=spark.attention(q,k,v,1024,(4,16,16),.2,cache=cache)
        expected=torch.nn.functional.scaled_dot_product_attention(q,k,v)
        self.assertTrue(torch.isfinite(actual).all())
        exact=actual[:,:,1024:];dense=expected[:,:,1024:]
        self.assertLess(float((exact-dense).float().norm()/dense.float().norm()),.02)

    def test_native_fp16_qkv_projection_and_rope(self):
        from comfy.ldm.minimax.model import Attention,rope_rotation_table
        import comfy.quant_ops
        spark=importlib.import_module(pkg.__name__+'.acceleration.sm75_spark')
        with torch.inference_mode():
            attn=Attention(256,2,128,1e-5,operations=torch.nn,dtype=torch.float16,device='cuda')
            attn.requires_grad_(False).eval()
            x=torch.randn(211,256,device='cuda',dtype=torch.float16)
            rope=rope_rotation_table(torch.randn(211,64,device='cuda'),torch.float16)
            actual=spark.compatibility.prepare_qkv(attn,x,rope)
            q,k,v=attn.qkv_proj(x).split(256,-1)
            q=q.view(1,211,2,128);k=k.view_as(q)
            comfy.quant_ops.ck.rms_rope_split_half_(q,k,rope,attn.q_norm.weight,
                attn.k_norm.weight,epsilon=attn.q_norm.eps,rot_dim=rope.shape[-3]*2)
            for result,reference in zip(actual,(q,k,v.reshape_as(q))):
                torch.testing.assert_close(result,reference.permute(0,2,1,3),atol=.002,rtol=.002)


if __name__=='__main__':
    unittest.main()
