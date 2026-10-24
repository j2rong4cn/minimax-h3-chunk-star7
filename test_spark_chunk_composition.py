"""Actual SM75 tiny H3: FP16 loader + CK Chunk + Spark + inherited contexts.

Uses random tiny weights; no user media, checkpoints or generation queue jobs.
"""
import contextlib
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parents[1]))
package=types.ModuleType('star7_composition_tests');package.__path__=[str(ROOT)];sys.modules[package.__name__]=package
unified=importlib.import_module(package.__name__+'.unified_acceleration')
chunk=importlib.import_module(package.__name__+'.nodes')
adaptive=importlib.import_module(package.__name__+'.adaptive_loader')
sm75=importlib.import_module(package.__name__+'.acceleration.sm75_spark')
from bench_unified_acceleration import assert_idle
from comfy.model_patcher import ModelPatcher
from comfy.ldm.minimax.model import MiniMaxH3Model,PackedLayout
from comfy.patcher_extension import CallbacksMP


class CarrierLinear(torch.nn.Linear):
    def forward(self,x):
        return super().forward(x.to(self.weight.dtype))


@contextlib.contextmanager
def apply_object_patches(base,model):
    # Tiny private fixture only. Production uses ModelPatcher's own lifecycle.
    backup=[]
    try:
        for path,value in model.object_patches.items():
            owner=base
            parts=path.split('.')
            for part in parts[:-1]:owner=getattr(owner,part)
            backup.append((owner,parts[-1],getattr(owner,parts[-1])))
            setattr(owner,parts[-1],value)
        yield
    finally:
        for owner,name,value in reversed(backup):setattr(owner,name,value)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability()==(7,5),'Actual SM75 required')
class CompositionTests(unittest.TestCase):
    def test_fp16_chunk_spark_inheritance_and_schedule(self):
        assert_idle();torch.manual_seed(754);torch.set_num_threads(4)
        device=torch.device('cuda');dtype=torch.float16
        ops=types.SimpleNamespace(Linear=CarrierLinear,RMSNorm=torch.nn.RMSNorm,LayerNorm=torch.nn.LayerNorm)
        diffusion=MiniMaxH3Model(hidden_size=128,num_layers=2,token_refiner_num_layers=0,
            num_attention_heads=1,attention_head_dim=128,ffn_hidden_size=128,text_dim=128,
            time_embed_hidden_size=128,time_embed_dim=128,operations=ops,dtype=dtype,device=device)
        diffusion.rope.inv_freq.copy_(torch.arange(1,17,device=device)/100)
        diffusion.requires_grad_(False).eval()
        base=torch.nn.Module();base.diffusion_model=diffusion
        # Match a real, not-yet-loaded quantized checkpoint: FP16 is a pending
        # ModelPatcher override, not a mutation of the shared BaseModel.
        base.manual_cast_dtype=torch.bfloat16
        base.model_sampling=types.SimpleNamespace(percent_to_sigma=lambda value:1.-value)
        base.get_dtype_inference=lambda:base.manual_cast_dtype
        original=ModelPatcher(base,device,device)
        protected=adaptive._patch_h3_model(original,loader_native=True)
        model=chunk.install_model_patch(protected,256,True,False,128,True,True,
            'comfy_kitchen_int8',qkv_chunk_tokens=256,out_proj_chunk_tokens=128)
        positive=[[torch.randn(1,4,128,device=device,dtype=dtype),{}]]
        context={'model':model,'positive':positive,'video_vae':object(),'first_frame':object()}
        events=[]
        with patch.object(unified,'MIN_TOKENS',256), patch.object(unified.diagnostics,'emit',side_effect=lambda node,report:events.append(report)):
            prepared=unified.MiniMaxH3UnifiedAccelerationStar7().patch(model,positive,mode='spark',
                warmup_ratio=.2,h3_context=context)
            enhanced,conditions,_,refine=prepared['result']
            self.assertEqual(prepared['ui']['star7_unified'][0]['effective_mode'],'spark')
            self.assertTrue(prepared['ui']['star7_unified'][0]['composition']['star7_chunk'])
            self.assertIs(refine['model'],enhanced);self.assertIs(context['model'],model)
            for path in model.object_patches:
                if '.mlp.forward' in path or '.out_proj.forward' in path:
                    self.assertIs(enhanced.object_patches[path],model.object_patches[path])
            inputs=[torch.randn(1,24,7,16,16,device=device,dtype=dtype),
                    torch.randn(1,32,2,12,device=device,dtype=dtype)]
            payload={'refs':[],'cond_video_latents':[],'layout':PackedLayout(4,7,16,16,12,refs=[])}
            schedule=torch.tensor([1.,.75,.5,.25,0.],device=device)
            options={**enhanced.model_options['transformer_options'],'sample_sigmas':schedule}
            def apply(value,timestep,**c):
                return diffusion._forward(value,timestep,conditions[0][0],
                    transformer_options=options,minimax_payload=payload)
            def run(patched):
                options.clear();options.update(patched.model_options['transformer_options'])
                options['sample_sigmas']=schedule
                for callback in patched.get_all_callbacks(CallbacksMP.ON_PRE_RUN):callback(patched)
                with apply_object_patches(base,patched),torch.inference_mode():
                    for sigma in schedule[:-1]:
                        options['sigmas']=sigma.reshape(1)
                        wrapper=patched.model_options.get('model_function_wrapper')
                        arguments={'input':inputs,'timestep':sigma.reshape(1)*1000,'c':{}}
                        result=wrapper(apply,arguments) if wrapper else apply(inputs,arguments['timestep'])
                        for value in result:self.assertTrue(torch.isfinite(value).all())
                torch.cuda.synchronize()
                return [value.detach().clone() for value in result]
            baseline=run(model)
            with (patch.object(chunk,'_prepare_h3_qkv_chunked',wraps=chunk._prepare_h3_qkv_chunked) as producer,
                  patch.object(sm75,'attention',wraps=sm75.attention) as sparse):
                spark_output=run(enhanced)
                self.assertEqual(sparse.call_count,6)  # 4 steps, 1 warmup, 2 blocks
                self.assertEqual(producer.call_count,8)  # chunk producer also owns warmup
            self.assertEqual(events[-1]['sampling_steps'],4)
            # Spark state/plans must not leak to the untouched CK branch.
            restored=run(model)
            for before,after in zip(baseline,restored):
                torch.testing.assert_close(before,after,rtol=0,atol=0)
            repeated=run(enhanced)
            for before,after in zip(spark_output,repeated):
                torch.testing.assert_close(before,after,rtol=0,atol=0)
            off=unified.MiniMaxH3UnifiedAccelerationStar7().patch(model,positive,mode='spark',enabled=False,h3_context=context)['result']
            self.assertIs(off[0],model);self.assertIs(off[1],positive);self.assertIs(off[3],context)
            # HD/face choosing a later explicit CK backend must not silently
            # retain Spark. VPack wrappers/conditioning can remain attached.
            override=chunk.install_model_patch(enhanced,256,True,False,128,True,True,
                'comfy_kitchen_int8',qkv_chunk_tokens=256,out_proj_chunk_tokens=128)
            self.assertIsNot(override.object_patches['diffusion_model.blocks.0.attn.forward'],
                             enhanced.object_patches['diffusion_model.blocks.0.attn.forward'])
            options.clear();options.update(override.model_options['transformer_options']);options['sample_sigmas']=schedule
            with patch.object(sm75,'attention',side_effect=AssertionError('Spark survived explicit CK selection')):
                run(override)
            options.clear();options.update(enhanced.model_options['transformer_options']);options['sample_sigmas']=schedule
            for error in (KeyboardInterrupt('cancel fixture'),torch.cuda.OutOfMemoryError('synthetic OOM')):
                with patch.object(sm75,'attention',side_effect=error),self.assertRaises(type(error)):
                    run(enhanced)
                self.assertEqual(events[-1]['phase'],'interrupted-or-error')


if __name__=='__main__':unittest.main()
