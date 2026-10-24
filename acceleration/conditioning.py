"""VPack schedules with branch-safe native H3 physical-time reconstruction."""
import copy
import logging
import weakref

import torch

from ..vendor.vpack.comfyui_ref2va import (
    _compression_dense_indices, ROPE_SCALE_KEY, ROPE_MODE_KEY,
    ROPE_DENSE_LATENT_T_KEY, ROPE_DENSE_INDICES_KEY,
)

LOG = logging.getLogger('Star7UnifiedAcceleration')
RATIO_KEY = '_star7_vpack_ratio'
WRAPPER_KEY = '_star7_vpack_physical_time'


def video_references(positive):
    if not isinstance(positive, (list, tuple)):
        raise ValueError('H3 positive must be CONDITIONING')
    for item in positive:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], dict):
            raise ValueError('Malformed CONDITIONING entry')
        refs = item[1].get('minimax_refs', [])
        if not isinstance(refs, list):
            raise ValueError('H3 minimax_refs must be a list')
        for ref in refs:
            if not isinstance(ref, dict):
                raise ValueError('Malformed H3 reference')
            if ref.get('kind') in ('video', 'video_audio'):
                yield ref


def _validate(ref):
    value = ref.get('latent')
    if not torch.is_tensor(value) or value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 24:
        raise ValueError('Reference video must have latent shape [1,24,T,H,W]')
    for name, size in (('latent_t', value.shape[2]), ('latent_h', value.shape[3]), ('latent_w', value.shape[4])):
        if type(ref.get(name)) is not int or ref[name] != size or size < 1:
            raise ValueError('Reference video latent metadata mismatch: ' + name)
    audio_t = ref.get('ref_audio_t', 0)
    if type(audio_t) is not int or audio_t < 0:
        raise ValueError('Invalid reference-video audio length')
    audio = ref.get('audio_latent')
    if audio_t and (not torch.is_tensor(audio) or tuple(audio.shape) != (1, 32, 2, audio_t)):
        raise ValueError('Reference-video audio metadata mismatch')


def _selection(ref):
    _validate(ref)
    indices, dense_t = ref.get(ROPE_DENSE_INDICES_KEY), ref.get(ROPE_DENSE_LATENT_T_KEY)
    if (not isinstance(indices, (list, tuple)) or len(indices) != ref['latent_t']
            or type(dense_t) is not int or dense_t < ref['latent_t'] or dense_t <= 0
            or any(type(index) is not int for index in indices)
            or not indices or indices[0] < 0 or indices[-1] >= dense_t
            or any(a >= b for a, b in zip(indices, indices[1:]))):
        raise ValueError('Invalid compressed reference-video temporal schedule')
    return dense_t, tuple(indices)


def correct_payload(payload, cache=None, signature=None):
    """Use native dense geometry to recover exact original retained time rows.

    Unlike mutating a shared PackedLayout, this also preserves guide anchors
    before the reference rows and supports separate positive/negative branches.
    No reference audio, spatial coordinate, mask or target latent is modified.
    """
    if not isinstance(payload, dict):
        return payload
    refs = payload.get('refs', [])
    if not isinstance(refs, list):
        raise ValueError('Invalid runtime H3 references')
    schedules = {index: _selection(ref) for index, ref in enumerate(refs)
                 if isinstance(ref, dict) and RATIO_KEY in ref}
    if not schedules:
        return payload
    from comfy.ldm.minimax.model import PackedLayout
    layout = payload.get('layout')
    if signature is None:
        signature = getattr(layout, 'signature', None)
    if not isinstance(signature, tuple) or len(signature) != 5:
        raise RuntimeError('VPack needs a native H3 PackedLayout or latent_shapes/context')
    source_layout = layout
    ref_geometry = tuple((ref.get('kind'), ref.get('latent_t'), ref.get('latent_h'), ref.get('latent_w'), ref.get('ref_audio_t', 0)) for ref in refs)
    guide_geometry = tuple((guide.get('resolved_frame_index'),
        tuple(guide['latent'].shape) if guide.get('latent') is not None else None,
        tuple(guide['audio_latent'].shape) if guide.get('audio_latent') is not None else None)
        for guide in (payload.get('keyframes') or ()))
    fingerprint = (signature, ref_geometry, guide_geometry, tuple((index, *schedule) for index, schedule in schedules.items()))
    cached = cache.get(source_layout) if cache is not None and source_layout is not None else None
    if cached is not None and cached[0] == fingerprint:
        return {**payload, 'layout': cached[1]}
    if layout is None or getattr(layout, 'signature', None) != signature:
        layout = PackedLayout(*signature, keyframes=payload.get('keyframes'), refs=refs)
    dense_refs = [dict(ref, latent_t=schedules[index][0]) if index in schedules else ref
                  for index, ref in enumerate(refs)]
    dense = PackedLayout(*signature, keyframes=payload.get('keyframes'), refs=dense_refs)
    if len(layout.segments) != len(dense.segments):
        raise RuntimeError('VPack PackedLayout segment contract changed')
    # Locate each video segment using the same order as native H3, including
    # guides and each video's optional audio segment.
    cursor = 1
    while cursor < len(layout.segments) and layout.segments[cursor][2] in ('cond', 'cond_audio'):
        cursor += 1
    video_segments = {}
    for index, ref in enumerate(refs):
        kind = ref['kind']
        if kind == 'image': cursor += 1
        elif kind == 'audio': cursor += int(ref.get('ref_audio_t', 0) > 0)
        elif kind in ('video', 'video_audio'):
            cursor += int(ref.get('ref_audio_t', 0) > 0)
            if index in schedules: video_segments[cursor] = schedules[index]
            cursor += 1
        else: raise ValueError('Unsupported H3 reference kind: ' + str(kind))
    corrected = layout.position_ids.clone()
    for index, (compact_segment, dense_segment) in enumerate(zip(layout.segments, dense.segments)):
        start, stop, kind = compact_segment; ds, de, dense_kind = dense_segment
        if kind != dense_kind:
            raise RuntimeError('VPack PackedLayout segment mismatch')
        times = dense.position_ids[ds:de, 0]
        if index in video_segments:
            dense_t, indices = video_segments[index]
            rows, remainder = divmod(de - ds, dense_t)
            if remainder or (stop - start) != rows * len(indices):
                raise RuntimeError('VPack reference-video rows are not frame-aligned')
            times = times.reshape(dense_t, rows).index_select(0, torch.tensor(indices, device=times.device)).flatten()
        elif stop - start != de - ds:
            raise RuntimeError('VPack changed non-video reference rows')
        corrected[start:stop, 0] = times.to(corrected.device)
    private_layout = copy.copy(layout)
    private_layout.position_ids = corrected
    if cache is not None: cache[source_layout if source_layout is not None else layout] = (fingerprint, private_layout)
    LOG.info('Star7 unified | VPack RoPE corrected | packed_rows=%d dense_rows=%d guides_preserved=true',
             corrected.shape[0], dense.position_ids.shape[0])
    return {**payload, 'layout': private_layout}


def install_rope_wrapper(model):
    if model.model_options.get(WRAPPER_KEY):
        return model
    previous = model.model_options.get('model_function_wrapper')
    cache = weakref.WeakKeyDictionary()
    def wrapper(apply_model, args):
        def apply_corrected(inputs, timestep, **conditions):
            conditions = dict(conditions)
            payload = conditions.get('minimax_payload')
            shapes, context = conditions.get('latent_shapes'), conditions.get('c_crossattn')
            signature = None
            if shapes is not None and len(shapes) >= 2 and torch.is_tensor(context):
                vs = shapes[0]
                signature = (context.shape[1], vs[2], (vs[3] + 1)//2*2, (vs[4] + 1)//2*2, shapes[1][-1])
            conditions['minimax_payload'] = correct_payload(payload, cache, signature)
            return apply_model(inputs, timestep, **conditions)
        if previous is not None:
            return previous(apply_corrected, args)
        return apply_corrected(args['input'], args['timestep'], **args['c'])
    patched = model.clone()
    patched.set_model_unet_function_wrapper(wrapper)
    patched.model_options[WRAPPER_KEY] = True
    from comfy.patcher_extension import CallbacksMP
    def clear(*args): cache.clear()
    patched.add_callback_with_key(CallbacksMP.ON_PRE_RUN, WRAPPER_KEY, clear)
    patched.add_callback_with_key(CallbacksMP.ON_CLEANUP, WRAPPER_KEY, clear)
    return patched


def pack(model, positive, ratio):
    refs = list(video_references(positive))
    stats = {'videos': len(refs), 'dense_t': 0, 'packed_t': 0, 'saved_tokens': 0, 'ratio': ratio}
    if ratio not in ('0/5', '1/5', '2/5', '3/5'):
        raise ValueError('Invalid VPack compression ratio')
    if ratio == '0/5':
        for ref in refs:
            _validate(ref)
            stats['dense_t'] += ref['latent_t']; stats['packed_t'] += ref['latent_t']
        return model, positive, stats
    if not refs:
        return model, positive, stats
    replacements = {}
    for ref in refs:
        _validate(ref)
        if RATIO_KEY in ref:
            if ref[RATIO_KEY] != ratio:
                raise ValueError('VPack is already applied; connect the original conditioning before changing its ratio')
            dense_t, indices = _selection(ref)
            packed = ref
        else:
            if any(key in ref for key in (ROPE_MODE_KEY, ROPE_DENSE_INDICES_KEY, ROPE_DENSE_LATENT_T_KEY)) or ref.get(ROPE_SCALE_KEY, 1.) != 1.:
                raise ValueError('Reference already has external RoPE/VPack metadata; do not compress twice')
            dense_t = ref['latent_t']
            indices = _compression_dense_indices(dense_t, ratio)
            packed = {**ref, 'latent': ref['latent'].index_select(2, torch.tensor(indices, device=ref['latent'].device)),
                      'latent_t': len(indices), ROPE_SCALE_KEY: 1., ROPE_MODE_KEY: 'chunk_phase',
                      ROPE_DENSE_LATENT_T_KEY: dense_t, ROPE_DENSE_INDICES_KEY: indices, RATIO_KEY: ratio}
        replacements[id(ref)] = packed
        stats['dense_t'] += dense_t; stats['packed_t'] += len(indices)
        stats['saved_tokens'] += (dense_t - len(indices)) * ((ref['latent_h'] + 1)//2) * ((ref['latent_w'] + 1)//2)
    # Short video tails can retain all tokens; preserve exact identity in that case.
    if stats['saved_tokens'] == 0:
        return model, positive, stats
    packed_positive = [[embedding, {**metadata, 'minimax_refs': [replacements.get(id(ref), ref) for ref in metadata['minimax_refs']]}]
                       if isinstance(metadata.get('minimax_refs'), list) else [embedding, metadata]
                       for embedding, metadata in positive]
    return install_rope_wrapper(model), packed_positive, stats
