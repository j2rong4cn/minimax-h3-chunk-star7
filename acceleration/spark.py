"""Lazy Spark adapter. No architecture spoofing or environment modifications."""
from importlib import metadata
import re

import torch
from . import compatibility

SUPPORTED = {(8, 6): 'SM86', (8, 9): 'SM89', (12, 0): 'SM120'}


def _dtype(model, diffusion):
    # ModelPatcher applies this override only when loading the model. Checking
    # the shared BaseModel here sees the checkpoint's old dtype, even when the
    # upstream loader has already requested FP16 on this particular branch.
    getter = getattr(model, 'get_model_object', None)
    if callable(getter):
        try:
            manual = getter('manual_cast_dtype')
        except AttributeError:
            manual = None
        if isinstance(manual, torch.dtype):
            return manual
    base = getattr(model, 'model', None)
    if callable(getattr(base, 'get_dtype_inference', None)):
        return base.get_dtype_inference()
    return getattr(diffusion, 'dtype', None)


def availability(model, info, backend='auto'):
    capability = info['capability']
    if backend == 'sm75_custom' and capability != (7,5):
        return False, 'The compiled Spark SM75 port requires compute capability 7.5'
    if capability == (7,5) and backend != 'sm75_custom':
        return False, 'SM75 requires the FP16 Spark adapter; backend selection is automatic in the Spark/VPack companion'
    if capability != (7,5) and capability not in SUPPORTED:
        label = 'CPU' if capability is None else f'SM{capability[0]}{capability[1]}'
        return False, f'No validated Spark adapter for {label}; supported adapters target SM75/86/89/120'
    diffusion = model.get_model_object('diffusion_model')
    required = torch.float16 if capability == (7,5) else torch.bfloat16
    if _dtype(model, diffusion) != required:
        if capability == (7,5):
            return False, 'Spark SM75 requires FP16 compute; input precision is preserved'
        return False, 'Upstream Spark requires BF16 compute; input precision is preserved, not silently converted'
    blocks = getattr(diffusion, 'blocks', [])
    if not blocks or any(getattr(block.attn, 'head_dim', None) != 128 for block in blocks):
        return False, 'Spark requires native H3 attention blocks with head_dim=128'
    replacements = model.model_options.get('transformer_options', {}).get('patches_replace', {}).get('dit', {})
    if replacements:
        return False, 'Existing DiT block replacements conflict with Spark (e.g. SLA/Sol/VSA/VEDA/TeaCache); connect a clean MODEL branch'
    if capability == (7,5):
        if compatibility.unknown_attention_paths(model) or model.model_options.get('transformer_options',{}).get('sol_take_forward'):
            return False, 'Spark SM75 attention-forward composition has not been validated; connect a clean attention branch'
        try:
            from . import sm75_kernel
            sm75_kernel.load()
            import triton  # FP16 LMv2 planner compiles its own SM75 kernels
        except (ImportError,OSError,RuntimeError,ValueError) as error:
            return False, 'Spark SM75 compiled backend unavailable: '+str(error)
        return True, 'Experimental Spark SM75 FP16 port: compiled CUDA global Reweight + LMv2/Top-K; full-video quality still requires testing'
    try:
        version = metadata.version('comfy-kitchen')
        architecture = SUPPORTED[capability].lower()
        if 'spark' not in version.lower() or not re.search(r'(?:^|[._-])' + architecture + r'(?:[._-]|$)', version.lower()):
            return False, f'comfy-kitchen {version} is not an explicitly matched {architecture} Spark build'
        from comfy_kitchen.backends import cuda as backend
        extension = getattr(backend, '_C', None)
        if extension is None or not all(hasattr(extension, name) for name in ('spark_attn', 'rms_rope')) or not callable(getattr(backend, 'spark_attn', None)):
            return False, 'Matched Spark CUDA extension/entry points are unavailable'
        from ..vendor.spark_h3 import comfyui_nodes as upstream
        # A known KJ low-VRAM hook composes in the documented order. Other
        # attention-forward replacements must not silently disappear.
        try:
            kj = upstream._kj_low_vram_forward(model)
        except upstream._Unsupported as error:
            return False, 'Incompatible Spark composition: ' + str(error)
        if kj is None and compatibility.unknown_attention_paths(model):
            return False, 'Existing attention-forward override is not a verified Spark composition'
    except (ImportError, OSError, metadata.PackageNotFoundError) as error:
        return False, 'Spark optional backend unavailable: ' + str(error)
    return True, f'comfy-kitchen {version}, {SUPPORTED[capability]}; kernel execution still requires runtime validation'


def install(model, config, info, report, node_id):
    model,composition=compatibility.prepare(model)
    report['composition']=composition
    from . import sm75_spark
    return sm75_spark.install(model,config,info,report,node_id)
