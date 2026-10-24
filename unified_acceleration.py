"""Spark/VPack acceleration. The historical class ID preserves saved links."""
import math

def __getattr__(name):
    # Compatibility for diagnostic clients; registration itself loads no backend.
    if name in ('conditioning','diagnostics','spark'):
        from importlib import import_module
        return import_module('.acceleration.'+name,__package__)
    raise AttributeError(name)


MODES=['off','spark','spark+vpack','vpack-only']
MARKER='_star7_unified_acceleration'
MIN_TOKENS=12288

class MiniMaxH3UnifiedAccelerationStar7:
    SEARCH_ALIASES=['Spark','VPack','H3 Spark 加速']
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'model':('MODEL',),'positive':('CONDITIONING',),
            'mode':(MODES,{'default':'spark'}),
            'topk_ratio':('FLOAT',{'default':.2,'min':.01,'max':1.,'step':.01}),
            'warmup_ratio':('FLOAT',{'default':.2,'min':0.,'max':1.,'step':.01}),
            'vpack_ratio':(['0/5','1/5','2/5','3/5'],{'default':'2/5'}),
        },'optional':{'enabled':('BOOLEAN',{'default':False}), 'h3_context':('STAR7_H3_REFINE_CONTEXT',)},'hidden':{'unique_id':'UNIQUE_ID'}}
    RETURN_TYPES=('MODEL','CONDITIONING','STRING','STAR7_H3_REFINE_CONTEXT')
    RETURN_NAMES=('model','positive','diagnostics','h3_context')
    FUNCTION='patch'
    CATEGORY='Star7/MiniMax H3'
    DESCRIPTION='Spark / VPack acceleration; Star7 Chunk owns projection chunk settings.'

    def patch(self,model,positive,mode='off',topk_ratio=.2,warmup_ratio=.2,
              vpack_ratio='2/5',h3_context=None,unique_id=None,enabled=None,**legacy):
        if mode=='auto':mode='vpack-only'
        if mode not in ('off',*MODES):raise ValueError('Unknown Spark/VPack mode')
        report={'requested_mode':mode,'effective_mode':'off','phase':'prepared',
            'sparse_kernel_calls':0,'evaluations':0,'reasons':[]}
        if enabled is not None and not isinstance(enabled,bool):raise TypeError('enabled must be boolean')
        if enabled is False or mode=='off':
            return {'ui':{'star7_unified':[report]},'result':(model,positive,'Star7 Spark/VPack | off',h3_context)}
        # Keep startup, bypass and off independent of all Spark/VPack dependencies.
        from .acceleration import conditioning, diagnostics, spark
        def finish(out_model,out_positive):
            context=h3_context
            if context is not None:
                if not isinstance(context,dict):raise TypeError('Invalid Star7 H3 context')
                if out_model is not model or out_positive is not positive:
                    context={**context,'model':out_model,'positive':[
                        [embedding,{key:value for key,value in metadata.items() if key!='minimax_keyframes'}]
                        for embedding,metadata in out_positive]}
            diagnostics.emit(unique_id,report)
            return {'ui':{'star7_unified':[report]},'result':(out_model,out_positive,diagnostics.text(report),context)}
        if model.model_options.get(MARKER):raise ValueError('Use one Spark/VPack acceleration node per MODEL branch')
        diffusion=model.get_model_object('diffusion_model')
        if not any(cls.__name__=='MiniMaxH3Model' for cls in type(diffusion).__mro__):
            raise TypeError('Spark/VPack requires native MiniMax H3')
        for name,value,low in [('topk_ratio',topk_ratio,.01),('warmup_ratio',warmup_ratio,0.)]:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=1.:
                raise ValueError('Invalid '+name)
        if vpack_ratio not in ('0/5','1/5','2/5','3/5'):raise ValueError('Invalid VPack ratio')
        refs=list(conditioning.video_references(positive))
        for ref in refs:conditioning._validate(ref)
        info=diagnostics.hardware();report['hardware']=info
        config=dict(steps=1,auto_steps=True,warmup_ratio=warmup_ratio,topk_ratio=topk_ratio,
            dense_layers=0,min_tokens=MIN_TOKENS,strict=True,
            sparse_scope='joint-video' if refs else 'target-video')
        report['config']=config
        want_spark=mode in ('spark','spark+vpack')
        if want_spark:
            if not spark.compatibility.has_chunk(model):
                raise ValueError('Spark requires upstream Star7 Chunk: connect its MODEL output')
            backend='sm75_custom' if info['capability']==(7,5) else 'native_spark'
            available,reason=spark.availability(model,info,backend)
            report['reasons'].append(reason)
            if not available:raise RuntimeError('Spark unavailable: '+reason)
        out_model,out_positive=model,positive
        reference_t=sum(ref['latent_t'] for ref in refs)
        stats=dict(videos=len(refs),dense_t=reference_t,packed_t=reference_t,saved_tokens=0,ratio=vpack_ratio)
        if mode in ('spark+vpack','vpack-only'):
            if refs:out_model,out_positive,stats=conditioning.pack(model,positive,vpack_ratio)
            else:report['reasons'].append('VPack skipped: no encoded reference videos')
        report['vpack']=stats
        packed=stats['saved_tokens']>0
        report['effective_mode']=('spark+vpack' if packed else 'spark') if want_spark else ('vpack-only' if packed else 'off')
        if want_spark:out_model=spark.install(out_model,config,info,report,unique_id)
        if out_model is not model:out_model.model_options[MARKER]=report['effective_mode']
        return finish(out_model,out_positive)

NODE_CLASS_MAPPINGS={'MiniMaxH3UnifiedAccelerationStar7':MiniMaxH3UnifiedAccelerationStar7}
NODE_DISPLAY_NAME_MAPPINGS={'MiniMaxH3UnifiedAccelerationStar7':'MiniMax H3 Spark / VPack 加速 - Star7'}
