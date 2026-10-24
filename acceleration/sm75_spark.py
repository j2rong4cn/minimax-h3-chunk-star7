"""FP16 SM75 Spark port: upstream LMv2 + Top-K + global-anchor Reweight.

Reblock is the pinned upstream FP16-capable planner, compiled by Triton.
The fused exact/approximate attention and weighted summaries use our NVCC DLL.
This is a separate port, not the upstream BF16 comfy-kitchen binary.
"""
import math

import torch

from . import sm75_kernel as kernel
from . import compatibility


def reblock(q, k, video_tokens, grid,cache=None):
    from ..vendor.spark_h3.h3_sparse_attention.mahalanobis_kmeans import (
        hilbert_midpoint_sample_indices, batched_noncentered_metric_factor)
    from ..vendor.spark_h3.h3_sparse_attention.landmark_tree_v2 import PreparedLandmarkTreeV2Permutation
    b,h,n,d=q.shape
    if video_tokens < 64 or math.prod(grid) != video_tokens:
        raise ValueError('Invalid sparse-video geometry for SM75 Spark Reblock')
    # Upstream default q_from_k: cluster K in the opposite Q second-moment
    # metric and share that layout with Q. RoPE was already applied in natural
    # packed order, so permutation never changes physical positions.
    queries=q[:,:,:video_tokens].reshape(b*h,video_tokens,d)
    keys=k[:,:,:video_tokens].reshape(b*h,video_tokens,d)
    cache={} if cache is None else cache
    key=(b*h,video_tokens,grid,str(q.device))
    index_key=('metric_indices',video_tokens,grid,str(q.device))
    if index_key not in cache:
        cache[index_key]=hilbert_midpoint_sample_indices(grid,video_tokens//64,device=q.device)
    indices=cache[index_key]
    factor=batched_noncentered_metric_factor(queries,indices,ridge_epsilon=.001)
    factor=factor.to(keys.dtype)
    if key not in cache:
        cache[key]=PreparedLandmarkTreeV2Permutation(batch=b*h,tokens=video_tokens,dim=d,
            grid_shape=grid,device=q.device,dtype=q.dtype,return_inverse=False,
            compact_direct_route=True)
    plan=cache[key]
    features=torch.empty_like(keys)
    # Bound the projection workspace independently of the video length.
    for start in range(0,video_tokens,4096):
        features[:,start:start+4096]=torch.bmm(keys[:,start:start+4096],factor)
    # The user's cudaMallocAsync run fatally trapped while freeing a failed
    # CUDA Graph capture. This adapter never captures/replays allocator graphs.
    permutation,_=plan.run(features,allow_cuda_graph=False)
    video=permutation.reshape(b,h,video_tokens).long()
    suffix=torch.arange(video_tokens,n,device=q.device).expand(b,h,-1)
    permutation=torch.cat((video,suffix),2)
    inverse=torch.empty_like(permutation)
    inverse.scatter_(2,permutation,torch.arange(n,device=q.device).expand(b,h,-1))
    return permutation,inverse


def routes(q,k,video_tokens,ratio):
    """Top-K over video leaves; all conditioning/tail keys and queries exact."""
    b,h,n,d=q.shape;leaves=(n+63)//64;candidates=video_tokens//64
    if not candidates or not 0 < ratio <= 1:
        raise ValueError('Invalid Spark Top-K budget')
    mask=torch.zeros((b,h,leaves,leaves),device=q.device,dtype=torch.uint8)
    mask[:,:,:,candidates:]=1
    mask[:,:,candidates:,:]=1
    # Only complete sparse leaves enter approximate routing. The high-norm
    # remainder protected by the upstream planner stays in the exact suffix.
    qm=q[:,:,:candidates*64].reshape(b,h,candidates,64,d).mean(3,dtype=torch.float32)
    km=k[:,:,:candidates*64].reshape(b,h,candidates,64,d).mean(3,dtype=torch.float32)
    quota=max(1,min(candidates,round(ratio*candidates)))
    for start in range(0,candidates,64):
        scores=qm[:,:,start:start+64]@km.transpose(-1,-2)
        selected=scores.topk(quota,dim=-1,sorted=False).indices
        mask[:,:,start:start+64,:candidates].scatter_(-1,selected,1)
    return mask


def attention(q,k,v,video_tokens,grid,ratio,*,reorder=True,cache=None):
    kernel.validate(q,k,v)
    if reorder:
        permutation,inverse=reblock(q,k,video_tokens,grid,cache)
        expanded=permutation.unsqueeze(-1).expand(-1,-1,-1,128)
        q=q.gather(2,expanded);k=k.gather(2,expanded);v=v.gather(2,expanded)
    mask=routes(q,k,video_tokens,ratio)
    anchor=q[:,:,:video_tokens].mean(2,dtype=torch.float32).contiguous()
    ak,av,lm=kernel.summaries(k,v,anchor)
    # The SM75 consumer uses INT8 Tensor Core PV for selected exact blocks;
    # weighted Reweight summaries retain FP16 PV. Conditioning-only queries
    # use the original native CK kernel instead of a serial dense loop.
    result=kernel.consume_fast(q,k,v,mask,ak,av,lm,query_tokens=video_tokens)
    if reorder:
        result=result.gather(2,inverse.unsqueeze(-1).expand_as(result))
    return result


def install(model,config,info,report,node_id):
    from ..vendor.spark_h3 import comfyui_nodes as upstream
    from ..sla_backend import _configure_triton_cache
    from comfy.patcher_extension import CallbacksMP
    from .diagnostics import emit
    _configure_triton_cache()
    diffusion=model.get_model_object('diffusion_model')
    state=upstream._RunState.create(config['steps'],config['warmup_ratio'],config['topk_ratio'],
        sparse_scope=config['sparse_scope'],target_capability=info['capability'])
    state.controller.direct_output=False
    is_sm75=info['capability']==(7,5)
    required_dtype=torch.float16 if is_sm75 else torch.bfloat16
    if not compatibility.has_chunk(model):raise ValueError('Connect Star7 Chunk before Spark')
    counters={'evaluations':0,'sparse_kernel_calls':0,'dense_blocks':0}
    gates=set();plans={};patched=model.clone()
    for layer,block in enumerate(diffusion.blocks):
        expected_attention=model.object_patches.get(f'diffusion_model.blocks.{layer}.attn.forward')
        def replacement(args,extra,layer=layer,block=block,expected_attention=expected_attention):
            options=args['transformer_options'];x=args['img']
            compatibility.sync_steps(state,options,config)
            if layer==0:state.begin_evaluation(options)
            layout=options.get('minimax_h3_layout')
            reason=None
            if expected_attention is not None and block.attn.forward is not expected_attention:reason='later explicit attention selection'
            elif state.is_warmup:reason='dense warmup'
            elif layer<config['dense_layers']:reason='fixed dense layer'
            elif x.shape[0]<config['min_tokens']:reason='short sequence'
            elif layout is None:reason='missing native H3 layout'
            elif layout.signature[1]<=1:reason='single-frame target'
            elif not x.is_cuda:reason='Spark requires CUDA activations'
            if reason is not None:
                gates.add(reason);counters['dense_blocks']+=1
                if config['strict'] and reason in ('missing native H3 layout','Spark requires CUDA activations'):
                    raise upstream._Incompatible(reason)
                return extra['original_block'](args)
            try:
                packed=state.spark_layout(layout,x.shape[0],x.device)
            except upstream._Incompatible as error:
                if config['strict']:raise
                gates.add(str(error));counters['dense_blocks']+=1
                return extra['original_block'](args)
            if packed.video_tokens<64:
                gates.add('short sparse-video prefix');counters['dense_blocks']+=1
                return extra['original_block'](args)
            def accelerated(h,rope_freqs=None,transformer_options=None):
                # FP16 Exact keeps residual x in FP32 but feeds FP16 h into
                # attention. Validate at this compute boundary, not at x.
                if h.dtype!=required_dtype:
                    raise RuntimeError(f'Spark attention compute requires {required_dtype}, got {h.dtype}')
                q,k,v=compatibility.prepare_qkv(block.attn,h,rope_freqs)
                q,k,v=[value.index_select(2,packed.permutation) for value in (q,k,v)]
                if is_sm75:
                    result=attention(q,k,v,packed.video_tokens,packed.grid,config['topk_ratio'],cache=plans)
                else:
                    from ..vendor.spark_h3.comfyui_backend import comfy_kitchen_spark_attention
                    result=comfy_kitchen_spark_attention(state.controller,
                        *[value.permute(0,2,1,3).contiguous() for value in (q,k,v)],packed,layer).permute(0,2,1,3)
                result=result.index_select(2,packed.inverse_permutation)
                counters['sparse_kernel_calls']+=1
                return block.attn.out_proj(result.permute(0,2,1,3).reshape(h.shape[0],-1))
            # Runtime CUDA errors, OOM and cancellation propagate. Never retry
            # them as dense computation or report a false successful kernel.
            return extra['original_block']({**args,'attention':accelerated})
        patched.set_model_patch_replace(replacement,'dit','double_block',layer)
    previous=patched.model_options.get('model_function_wrapper')
    def wrapper(apply_model,args):
        try:
            result=previous(apply_model,args) if previous else apply_model(args['input'],args['timestep'],**args['c'])
        except BaseException:
            emit(node_id,{**report,**counters,'phase':'interrupted-or-error'});raise
        counters['evaluations']+=1
        actual=('spark+vpack' if report['vpack']['saved_tokens'] else 'spark') if counters['sparse_kernel_calls'] else ('vpack-only' if report['vpack']['saved_tokens'] else 'off')
        emit(node_id,{**report,**counters,'phase':'runtime','effective_mode':actual,
            'sampling_steps':state.steps,'auto_steps':config.get('auto_steps',True),
            'dense_gate_reasons':sorted(gates),'attention_executed':('spark-sm75-int8-pv-fp16-summary' if is_sm75 else 'matched-native-spark') if counters['sparse_kernel_calls'] else 'input-dense/gated'})
        return result
    patched.set_model_unet_function_wrapper(wrapper)
    def reset(*args):
        state.controller.reset();state.layout_cache.clear();state.previous_sigma=None;gates.clear()
        plans.clear()
        for key in counters:counters[key]=0
    patched.add_callback_with_key(CallbacksMP.ON_PRE_RUN,'star7_unified_sm75_spark',reset)
    patched.add_callback_with_key(CallbacksMP.ON_CLEANUP,'star7_unified_sm75_spark',reset)
    return patched
