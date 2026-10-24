"""CPU/native-layout checks; no model weights or live ComfyUI jobs are used."""
import copy
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
pkg = types.ModuleType('star7_unified_tests'); pkg.__path__ = [str(ROOT)]; sys.modules[pkg.__name__] = pkg
node = importlib.import_module(pkg.__name__ + '.unified_acceleration')
vpack = node.conditioning
from comfy.ldm.minimax.model import PackedLayout


class MiniMaxH3Model:
    dtype = torch.bfloat16


class Model:
    def __init__(self):
        self.model_options = {}; self.object_patches = {}; self.callbacks = {}; self.diffusion = MiniMaxH3Model()
    def clone(self):
        result = copy.copy(self); result.model_options = copy.deepcopy(self.model_options)
        result.object_patches = self.object_patches.copy(); result.callbacks = copy.deepcopy(self.callbacks)
        return result
    def get_model_object(self, name):
        return self.diffusion if name == 'diffusion_model' else types.SimpleNamespace(percent_to_sigma=lambda percent: 1. - percent)
    def set_model_unet_function_wrapper(self, wrapper): self.model_options['model_function_wrapper'] = wrapper
    def add_callback_with_key(self, kind, key, callback): self.callbacks[(kind, key)] = callback
    def remove_callbacks_with_key(self,kind,key):self.callbacks.pop((kind,key),None)
    def set_model_patch_replace(self, callback, name, block, index):
        self.model_options.setdefault('transformer_options', {}).setdefault('patches_replace', {}).setdefault(name, {})[(block, index)] = callback


def video(t=12, kind='video', audio=0):
    return {'kind': kind, 'latent': torch.arange(24*t*4*4, dtype=torch.float32).reshape(1,24,t,4,4),
            'latent_t': t, 'latent_h': 4, 'latent_w': 4, 'ref_audio_t': audio,
            'audio_latent': torch.zeros(1,32,2,audio) if audio else None}


def conditioning(refs): return [[torch.zeros(1,3,8), {'minimax_refs': refs, 'other': 'keep'}]]


class Tests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.hardware = patch.object(node.diagnostics, 'hardware', return_value={'gpu':'test SM75','capability':(7,5),'cuda':'13.0'})
        self.hardware.start(); self.addCleanup(self.hardware.stop)
        quiet = patch.object(node.diagnostics, 'emit'); quiet.start(); self.addCleanup(quiet.stop)

    def test_off_is_true_identity_without_backend_probe(self):
        positive = conditioning([video()])
        with patch.object(node.diagnostics, 'hardware', side_effect=AssertionError('off probed CUDA')):
            result = node.MiniMaxH3UnifiedAccelerationStar7().patch(self.model, positive, mode='off')['result']
        self.assertIs(result[0], self.model); self.assertIs(result[1], positive)

    def test_compute_precision_reads_pending_branch_patch(self):
        from comfy.model_patcher import ModelPatcher
        base = torch.nn.Module()
        base.diffusion_model = torch.nn.Module(); base.diffusion_model.dtype = torch.bfloat16
        base.manual_cast_dtype = None
        base.get_dtype_inference = lambda: base.manual_cast_dtype or base.diffusion_model.dtype
        original = ModelPatcher(base, torch.device('cpu'), torch.device('cpu'))
        protected = original.clone(); protected.set_model_compute_dtype(torch.float16)
        self.assertEqual(node.spark._dtype(original, base.diffusion_model), torch.bfloat16)
        self.assertEqual(node.spark._dtype(protected, base.diffusion_model), torch.float16)
        self.assertIsNone(base.manual_cast_dtype)
        self.assertNotIn('manual_cast_dtype', original.object_patches)

    def test_switch_disabled_preserves_all_inputs_without_backend_probe(self):
        positive = conditioning([video()]); context = {'keep': object()}
        with patch.object(node.diagnostics, 'hardware', side_effect=AssertionError('disabled probed CUDA')):
            for mode in ('spark', 'spark+vpack', 'vpack-only'):
                result = node.MiniMaxH3UnifiedAccelerationStar7().patch(
                    self.model, positive, mode=mode, enabled=False, h3_context=context)['result']
                self.assertIs(result[0], self.model); self.assertIs(result[1], positive)
                self.assertIs(result[3], context)

    def test_schedule_all_short_tails_and_tensor_values(self):
        expected = {'0/5': [0,1,2,3,4,5,6], '1/5': [0,1,2,3,5,6], '2/5': [0,1,3,5,6], '3/5': [0,3,5]}
        for ratio in expected:
            self.assertEqual(vpack._compression_dense_indices(7, ratio), expected[ratio])
        for t in range(1,32):
            for ratio in ('1/5','2/5','3/5'):
                ref = video(t); original = ref['latent'].clone(); positive = conditioning([ref])
                out_model, packed, stats = vpack.pack(self.model, positive, ratio)
                indices = vpack._compression_dense_indices(t, ratio)
                self.assertTrue(torch.equal(packed[0][1]['minimax_refs'][0]['latent'], original[:,:,indices]))
                self.assertTrue(torch.equal(ref['latent'], original)); self.assertEqual(ref['latent_t'], t)
                self.assertEqual(stats['saved_tokens'], (t-len(indices))*4)
                if stats['saved_tokens']: self.assertIsNot(out_model, self.model)

    def test_images_audio_metadata_and_multiple_conditioning_entries_preserved(self):
        image = {'kind':'image','latent':torch.randn(1,24,1,4,4),'latent_h':4,'latent_w':4}
        audio = {'kind':'audio','audio_latent':torch.randn(1,32,2,4),'ref_audio_t':4}
        ref = video(kind='video_audio',audio=5)
        positive = conditioning([image, ref, audio]); positive.append([torch.ones(1,2,8), {'another':'entry'}])
        _, packed, _ = vpack.pack(self.model, positive, '2/5')
        self.assertIs(packed[0][0], positive[0][0]); self.assertIs(packed[1][1], positive[1][1])
        result_refs = packed[0][1]['minimax_refs']
        self.assertIs(result_refs[0], image); self.assertIs(result_refs[2], audio)
        self.assertIs(result_refs[1]['audio_latent'], ref['audio_latent'])
        self.assertEqual(packed[0][1]['other'], 'keep')

    def test_native_positions_preserve_full_timeline_including_guides_and_audio(self):
        for audio_length in (0,3,100):
            refs = [video(12, audio=audio_length), {'kind':'image','latent_h':4,'latent_w':4}, video(7)]
            guide = {'resolved_frame_index':3, 'latent':torch.zeros(1,24,1,4,4), 'audio_latent':torch.zeros(1,32,2,3)}
            _, packed, _ = vpack.pack(self.model, conditioning(refs), '2/5'); new_refs = packed[0][1]['minimax_refs']
            original = PackedLayout(3,7,4,4,6,keyframes=[guide],refs=refs)
            compact = PackedLayout(3,7,4,4,6,keyframes=[guide],refs=new_refs)
            before = compact.position_ids.clone(); payload={'layout':compact,'refs':new_refs,'keyframes':[guide]}
            corrected = vpack.correct_payload(payload)['layout']
            self.assertIsNot(corrected, compact); self.assertTrue(torch.equal(compact.position_ids,before))
            self.assertTrue(torch.equal(corrected.position_ids[:,1:],before[:,1:]))
            ref_index=0
            for segment, dense_segment in zip(corrected.segments, original.segments):
                start,stop,kind=segment; ds,de,_=dense_segment
                expected = original.position_ids[ds:de,0]
                if kind=='ref_img':
                    ref=new_refs[ref_index]; ref_index+=1
                    if ref['kind'] in ('video','video_audio'):
                        ids=ref[vpack.ROPE_DENSE_INDICES_KEY]
                        expected=expected.reshape(ref[vpack.ROPE_DENSE_LATENT_T_KEY],4)[ids].flatten()
                torch.testing.assert_close(corrected.position_ids[start:stop,0],expected,rtol=0,atol=0)

    def test_wrapper_chains_existing_wrapper_and_does_not_modify_shared_payload(self):
        calls=[]
        def previous(apply,args):
            calls.append('previous'); return apply(args['input'],args['timestep'],**args['c'])
        self.model.model_options['model_function_wrapper']=previous
        positive=conditioning([video()]); patched, packed, _=vpack.pack(self.model,positive,'2/5')
        refs=packed[0][1]['minimax_refs']; layout=PackedLayout(3,2,4,4,4,refs=refs)
        payload={'layout':layout,'refs':refs}; before=layout.position_ids.clone()
        def apply(inputs,timestep,**c):
            calls.append(c['minimax_payload']['layout']);return inputs
        args={'input':object(),'timestep':torch.ones(1),'c':{'minimax_payload':payload}}
        result=patched.model_options['model_function_wrapper'](apply,args)
        self.assertIs(result,args['input']);self.assertEqual(calls[0],'previous');self.assertIsNot(calls[1],layout)
        self.assertIs(self.model.model_options['model_function_wrapper'],previous)
        self.assertIs(payload['layout'],layout);self.assertTrue(torch.equal(layout.position_ids,before))
        patched.model_options['model_function_wrapper'](apply,args)
        self.assertIs(calls[1],calls[3])

    def test_idempotence_and_reject_changing_already_packed_inputs(self):
        model,positive,_=vpack.pack(self.model,conditioning([video()]),'2/5')
        again,packed,_=vpack.pack(model,positive,'2/5')
        self.assertIs(again,model);self.assertIs(packed[0][1]['minimax_refs'][0],positive[0][1]['minimax_refs'][0])
        with self.assertRaises(ValueError):vpack.pack(model,positive,'1/5')
        with self.assertRaises(ValueError):vpack.pack(self.model,conditioning([{**video(),vpack.ROPE_MODE_KEY:'affine'}]),'2/5')

    def test_spark_requires_chunk_and_vpack_is_independent(self):
        operation=node.MiniMaxH3UnifiedAccelerationStar7()
        positive=conditioning([video()])
        for mode in ('auto','vpack-only'):
            result=operation.patch(self.model,positive,mode=mode)
            self.assertEqual(result['ui']['star7_unified'][0]['effective_mode'],'vpack-only')
            self.assertEqual(result['ui']['star7_unified'][0]['sparse_kernel_calls'],0)
        for mode in ('spark','spark+vpack'):
            with self.assertRaisesRegex(ValueError,'upstream Star7 Chunk'):
                operation.patch(self.model,positive,mode=mode)
        no_refs=conditioning([])
        result=operation.patch(self.model,no_refs,mode='vpack-only')
        self.assertIs(result['result'][0],self.model)
        self.assertEqual(result['ui']['star7_unified'][0]['effective_mode'],'off')

    def test_lean_parameters_and_ignored_legacy_chunk_controls(self):
        operation=node.MiniMaxH3UnifiedAccelerationStar7();positive=conditioning([video()])
        for kwargs in ({'topk_ratio':float('nan')},{'warmup_ratio':float('inf')},{'vpack_ratio':'8/5'}):
            with self.assertRaises(ValueError):operation.patch(self.model,positive,mode='vpack-only',**kwargs)
        required=operation.INPUT_TYPES()['required']
        self.assertEqual(list(required),['model','positive','mode','topk_ratio','warmup_ratio','vpack_ratio'])
        result=operation.patch(self.model,positive,mode='vpack-only',producer_chunk=65,
            quality_preset='fidelity',steps=0)['ui']['star7_unified'][0]
        self.assertEqual(result['vpack']['ratio'],'2/5')
        self.assertEqual(result['config']['topk_ratio'],.2)
        self.assertNotIn('producer_chunk',result['config'])

    def test_malformed_reference_and_temporal_indices_fail_before_mutation(self):
        ref=video();ref['latent_t']=0
        with self.assertRaises(ValueError):vpack.pack(self.model,conditioning([ref]),'2/5')
        _,positive,_=vpack.pack(self.model,conditioning([video()]),'2/5')
        ref=positive[0][1]['minimax_refs'][0];ref[vpack.ROPE_DENSE_INDICES_KEY]=[0]*ref['latent_t']
        with self.assertRaises(ValueError):vpack.pack(self.model,positive,'2/5')

    def test_optional_spark_adapter_imports_without_cuda_extension(self):
        upstream=importlib.import_module(pkg.__name__+'.vendor.spark_h3.comfyui_nodes')
        state=upstream._RunState.create(4,.25,.2)
        for sigma in (.9,.6):state.begin_evaluation({'sigmas':torch.tensor([sigma])})
        self.assertEqual(state.controller.evaluation_index,1)
        state.begin_evaluation({'sigmas':torch.tensor([.95])})
        self.assertEqual(state.controller.evaluation_index,0)
        for capability in ((7,5),(8,0),(9,0),(10,0),(12,1)):
            available,reason=node.spark.availability(self.model,{'capability':capability})
            self.assertFalse(available);self.assertIn('Spark',reason)

    def test_sm75_requires_explicit_backend_and_preserves_conflicting_patches(self):
        kernel=importlib.import_module(pkg.__name__+'.acceleration.sm75_kernel')
        self.model.diffusion.dtype=torch.float16
        self.model.diffusion.blocks=[types.SimpleNamespace(attn=types.SimpleNamespace(head_dim=128))]
        info={'capability':(7,5)}
        with patch.object(kernel,'load',return_value=object()) as loader:
            self.assertFalse(node.spark.availability(self.model,info,'auto')[0])
            loader.assert_not_called()
            self.assertTrue(node.spark.availability(self.model,info,'sm75_custom')[0])
            self.model.object_patches['blocks.0.attn.forward']=object()
            self.assertFalse(node.spark.availability(self.model,info,'sm75_custom')[0])
            self.assertFalse(node.spark.availability(self.model,{'capability':(8,6)},'sm75_custom')[0])

    def test_refine_context_receives_same_model_and_packed_refs_without_full_frame_guides(self):
        positive=conditioning([video()]);positive[0][1]['minimax_keyframes']=[object()]
        context={'model':self.model,'positive':positive,'video_vae':object(),'first_frame':object()}
        operation=node.MiniMaxH3UnifiedAccelerationStar7()
        untouched=operation.patch(self.model,positive,mode='off',h3_context=context)['result']
        self.assertIs(untouched[3],context)
        model,packed,_,result=operation.patch(self.model,positive,mode='vpack-only',h3_context=context)['result']
        self.assertIs(result['model'],model)
        self.assertIs(result['positive'][0][1]['minimax_refs'][0],packed[0][1]['minimax_refs'][0])
        self.assertNotIn('minimax_keyframes',result['positive'][0][1])
        self.assertIn('minimax_keyframes',positive[0][1])
        self.assertIs(result['first_frame'],context['first_frame'])
        self.assertIs(context['model'],self.model)

    def test_actual_sampler_steps_and_explicit_refine_attention_take_ownership(self):
        compatibility=node.spark.compatibility
        state=types.SimpleNamespace(steps=20)
        compatibility.sync_steps(state,{'sample_sigmas':torch.linspace(1,0,5)},{'auto_steps':True})
        self.assertEqual(state.steps,4)
        state.steps=20
        compatibility.sync_steps(state,{'sample_sigmas':torch.linspace(1,0,3)},{'auto_steps':False})
        self.assertEqual(state.steps,20)

    def test_veda_is_suspended_only_on_the_spark_clone(self):
        from comfy.patcher_extension import CallbacksMP
        def previous(*args,**kwargs):pass
        def veda(*args,**kwargs):pass
        veda._star7_attention_kind='veda';veda._star7_previous_attention=previous
        self.model.model_options={'transformer_options':{'star7_veda_active':True,
            'optimized_attention_override':veda,'veda_held_head_chunks':4,'minimax_head_chunks':1}}
        self.model.add_callback_with_key(CallbacksMP.ON_PREPARE_STATE,'veda_sparse_attention',object())
        output,report=node.spark.compatibility.prepare(self.model)
        self.assertIs(output.model_options['transformer_options']['optimized_attention_override'],previous)
        self.assertEqual(output.model_options['transformer_options']['minimax_head_chunks'],4)
        self.assertNotIn((CallbacksMP.ON_PREPARE_STATE,'veda_sparse_attention'),output.callbacks)
        self.assertIs(self.model.model_options['transformer_options']['optimized_attention_override'],veda)
        self.assertIn((CallbacksMP.ON_PREPARE_STATE,'veda_sparse_attention'),self.model.callbacks)

    def test_real_veda_override_is_unwrapped_without_editing_veda(self):
        veda=importlib.import_module(pkg.__name__+'.vendor.veda.comfy_patch')
        def previous(*args,**kwargs):pass
        override=veda.VedaPatch.__new__(veda.VedaPatch).make_override(previous)
        self.model.model_options={'transformer_options':{'star7_veda_active':True,
            'optimized_attention_override':override,'veda_held_head_chunks':8}}
        output,report=node.spark.compatibility.prepare(self.model)
        self.assertIs(output.model_options['transformer_options']['optimized_attention_override'],previous)
        self.assertEqual(output.model_options['transformer_options']['minimax_head_chunks'],8)
        self.assertIs(self.model.model_options['transformer_options']['optimized_attention_override'],override)

    def test_real_native_h3_forward_accepts_packed_conditioning(self):
        from comfy.ldm.minimax.model import MiniMaxH3Model as NativeH3
        torch.manual_seed(44)
        diffusion=NativeH3(hidden_size=128,num_layers=1,token_refiner_num_layers=0,
            num_attention_heads=1,attention_head_dim=128,ffn_hidden_size=128,text_dim=128,
            time_embed_hidden_size=128,time_embed_dim=128,operations=torch.nn,dtype=torch.float32,device='cpu')
        diffusion.rope.inv_freq.copy_(torch.arange(1,17,dtype=torch.float32)/100)
        diffusion.requires_grad_(False).eval()
        _,positive,_=vpack.pack(self.model,conditioning([video(7)]),'2/5')
        refs=positive[0][1]['minimax_refs']
        payload=vpack.correct_payload({'layout':PackedLayout(3,2,4,4,4,refs=refs),'refs':refs,
            'cond_video_latents':[refs[0]['latent']]})
        x=[torch.randn(1,24,2,4,4),torch.randn(1,32,2,4)]
        with torch.no_grad():
            output=diffusion._forward(x,torch.tensor([500.]),torch.randn(1,3,128),transformer_options={},minimax_payload=payload)
        self.assertEqual([tuple(value.shape) for value in output],[tuple(value.shape) for value in x])
        self.assertTrue(all(torch.isfinite(value).all() for value in output))

    def test_spark_precision_architecture_and_competing_patches_are_not_overridden(self):
        self.model.diffusion.blocks=[types.SimpleNamespace(attn=types.SimpleNamespace(head_dim=128))]
        self.model.diffusion.dtype=torch.float16
        self.assertIn('BF16',node.spark.availability(self.model,{'capability':(8,6)})[1])
        self.model.diffusion.dtype=torch.bfloat16
        with patch.object(node.spark.metadata,'version',return_value='0.2.39+spark.h3.sm89.cu130.1'):
            self.assertIn('matched',node.spark.availability(self.model,{'capability':(8,6)})[1])
        block=object()
        self.model.model_options={'transformer_options':{'patches_replace':{'dit':{('double_block',0):block}}}}
        self.assertIn('conflict',node.spark.availability(self.model,{'capability':(8,6)})[1])
        self.assertIs(self.model.model_options['transformer_options']['patches_replace']['dit'][('double_block',0)],block)

    def test_real_model_patcher_wrapper_rebuilds_a_resized_runtime_layout(self):
        from comfy.model_patcher import ModelPatcher
        base=torch.nn.Module();base.diffusion_model=MiniMaxH3Model()
        model=ModelPatcher(base,torch.device('cpu'),torch.device('cpu'))
        original=conditioning([video()]);patched,positive,_=vpack.pack(model,original,'2/5')
        refs=positive[0][1]['minimax_refs'];layout=PackedLayout(3,7,4,4,6,refs=refs)
        conditions={'minimax_payload':{'layout':layout,'refs':refs},'c_crossattn':torch.zeros(1,3,128),
                    'latent_shapes':[(1,24,7,8,8),(1,32,2,6)]}
        def apply(inputs,timestep,**c):return c['minimax_payload']['layout']
        result=patched.model_options['model_function_wrapper'](apply,{'input':object(),'timestep':torch.ones(1),'c':conditions})
        self.assertEqual(result.signature,(3,7,8,8,6));self.assertEqual(layout.signature,(3,7,4,4,6))
        self.assertNotIn(vpack.WRAPPER_KEY,model.model_options)
        from comfy.patcher_extension import CallbacksMP
        for callback in patched.get_all_callbacks(CallbacksMP.ON_PRE_RUN):callback(patched)

    def test_spark_dispatches_only_to_shared_chunk_implementation(self):
        shared=importlib.import_module(pkg.__name__+'.acceleration.sm75_spark')
        prepared=self.model.clone();report={};config={'steps':1}
        with patch.object(node.spark.compatibility,'prepare',return_value=(prepared,{'star7_chunk':True})), \
             patch.object(shared,'install',return_value=prepared) as install:
            output=node.spark.install(self.model,config,{'capability':(7,5)},report,'123')
        self.assertIs(output,prepared)
        install.assert_called_once_with(prepared,config,{'capability':(7,5)},report,'123')
        self.assertEqual(report['composition'],{'star7_chunk':True})


if __name__=='__main__':unittest.main()
