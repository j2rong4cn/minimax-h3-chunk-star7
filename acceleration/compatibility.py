"""Explicit composition with Star7 Chunk; unrelated attention patches stay guarded."""
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
_ATTENTION_NAMES={
    '_minimax_ck_int8_attention_forward','_minimax_sla_forward',
    '_minimax_sol_forward','_minimax_hybrid_attention_forward',
}


def has_chunk(model):
    return bool(model.model_options.get('transformer_options',{}).get('star7_h3_chunk'))


def known_chunk_attention(value):
    function=getattr(value,'__func__',value)
    code=getattr(function,'__code__',None)
    return bool(code is not None and function.__name__ in _ATTENTION_NAMES
        and Path(code.co_filename).resolve()==(ROOT/'nodes.py').resolve())


def unknown_attention_paths(model):
    return [key for key,value in getattr(model,'object_patches',{}).items()
        if str(key).endswith('.attn.forward') and not (has_chunk(model) and known_chunk_attention(value))]


def prepare(model):
    """Give Spark attention ownership on a clone, preserving all chunk/FP16 work.

    Native CK is the dense/warmup path; selecting Spark does not stack SLA/Sol
    or VEDA approximation on the same attention operation. The input branch is
    untouched, so off/VPack-only still use its original attention selection.
    """
    patched=model.clone()
    composition={'star7_chunk':has_chunk(model),'attention_owner':'spark','preserved':[]}
    if has_chunk(model):
        from .. import nodes
        diffusion=model.get_model_object('diffusion_model')
        for index,block in enumerate(diffusion.blocks):
            path=f'diffusion_model.blocks.{index}.attn.forward'
            value=getattr(model,'object_patches',{}).get(path)
            if value is not None and known_chunk_attention(value):
                composition.setdefault('previous_attention',set()).add(getattr(getattr(value,'__func__',value),'__name__','native'))
                patched.add_object_patch(path,nodes._weak_method(block.attn,nodes._minimax_ck_int8_attention_forward))
        composition['preserved']=['QKV chunk/resident weights','MLP chunk','output projection chunk','FP16 guards','decode chunk']
        if 'previous_attention' in composition:composition['previous_attention']=sorted(composition['previous_attention'])
    options=patched.model_options.setdefault('transformer_options',{})
    if options.get('star7_veda_active'):
        current=options.get('optimized_attention_override')
        if getattr(current,'_star7_attention_kind',None)=='veda':
            previous=current._star7_previous_attention
        else:
            code=getattr(current,'__code__',None)
            if code is None or code.co_name!='override' or Path(code.co_filename).resolve()!=(ROOT/'vendor/veda/comfy_patch.py').resolve():
                raise RuntimeError('Cannot safely compose an unknown VEDA attention override')
            closure=dict(zip(code.co_freevars,(cell.cell_contents for cell in current.__closure__ or ())))
            previous=closure['previous']
        options['optimized_attention_override']=previous
        if options['optimized_attention_override'] is None:options.pop('optimized_attention_override')
        options.pop('star7_veda_active',None)
        if 'veda_held_head_chunks' in options:
            options['minimax_head_chunks']=options.pop('veda_held_head_chunks')
        from comfy.patcher_extension import CallbacksMP
        for phase in (CallbacksMP.ON_PREPARE_STATE,CallbacksMP.ON_CLEANUP):
            patched.remove_callbacks_with_key(phase,'veda_sparse_attention')
        composition['veda']='suspended on Spark branch; preserved on input branch'
    return patched,composition


def sync_steps(state,options,config):
    schedule=options.get('sample_sigmas')
    if config.get('auto_steps',True) and torch.is_tensor(schedule) and schedule.numel()>1:
        state.steps=schedule.numel()-1


def prepare_qkv(attn,x,rope):
    import comfy.model_management as mm
    import comfy.quant_ops
    from ..nodes import _prepare_h3_qkv_chunked
    return tuple(value.contiguous() for value in _prepare_h3_qkv_chunked(
        attn,x,rope,mm,comfy.quant_ops,output_dtype=x.dtype,output_layout='BHLD'))
