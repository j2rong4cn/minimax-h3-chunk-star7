"""The ComfyUI node: "Veda Sparse Attention (MiniMax H3)".

One node, MODEL in -> MODEL out: an attention override, placed on the MODEL
wire after the model loader and any LoRA loaders and last before the sampler
or guider. Visible inputs are just the model and the predictor file; every
tuning knob is an advanced input (hidden until "show advanced" is on) with
the trained defaults. Runtime status and diagnostics are reported in the console log.
"""

from __future__ import annotations

import os

import comfy.model_management
import comfy.patcher_extension
import folder_paths
from comfy_api.latest import ComfyExtension, io

from . import backends
from . import comfy_patch
from . import hardware
from . import predictors
from . import settings as veda_settings
from . import status as veda_status
from .core import bundle as veda_bundle

FOLDER = 'veda'


def register_model_folder() -> str:
    """Registers models/veda (only .safetensors, so stray files and
    stray files never show up in the list)."""
    path = os.path.join(folder_paths.models_dir, FOLDER)
    entry = folder_paths.folder_names_and_paths.get(FOLDER)
    if entry is None:
        folder_paths.folder_names_and_paths[FOLDER] = ([path], {'.safetensors'})
    else:
        if path not in entry[0]:
            entry[0].append(path)
        if entry[1]:
            entry[1].add('.safetensors')
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:  # read-only installs: the user can still add paths
        pass
    return path


def _predictor_options() -> list[str]:
    local = folder_paths.get_filename_list(FOLDER)
    return list(local) + [n for n in predictors.KNOWN_PREDICTORS
                          if n not in local]


def _predictor_path(name: str) -> str:
    """Local path of a predictor.

    Raises:
        ValueError: If the file is not there, with how to get it. The node
            does not fetch it: ComfyUI's missing-model dialog does that,
            driven by `properties.models` in the example workflows.
    """
    path = folder_paths.get_full_path(FOLDER, name)
    if path is not None:
        return path
    folder = folder_paths.get_folder_paths(FOLDER)[0]
    known = predictors.KNOWN_PREDICTORS.get(name)
    if known is None:
        raise ValueError(f'Predictor {name!r} is not in {folder}. '
                         'Pick another file or put it there.')
    raise ValueError(predictors.how_to_get(known, folder))


def _bundle(path: str) -> veda_bundle.PredictorBundle:
    return veda_bundle.load_bundle(path)


def _check_model(model, bundle) -> tuple[int, int, int]:
    diffusion = model.get_model_object('diffusion_model')
    if type(diffusion).__name__ != 'MiniMaxH3Model':
        raise ValueError(
            'Veda accelerates MiniMax-H3 only. Connect the MODEL of a '
            f'MiniMax-H3 checkpoint (got {type(diffusion).__name__}).')
    attn = diffusion.blocks[0].attn
    shape = (len(diffusion.blocks), attn.heads, attn.head_dim)
    if shape != (bundle.num_layers, bundle.num_heads, bundle.head_dim):
        raise ValueError(
            f'This model has {shape[0]} blocks x {shape[1]} heads x '
            f'{shape[2]}, but the predictor was trained for '
            f'{bundle.num_layers} x {bundle.num_heads} x {bundle.head_dim}.')
    return shape


def _other_sparse_node(model) -> bool:
    callbacks = getattr(model, 'callbacks', {}) or {}
    prepare = callbacks.get(
        comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, {})
    return 'block_sparse_attention' in prepare


class VedaSparseAttention(io.ComfyNode):
    """Veda learned block-sparse attention for MiniMax-H3."""

    @classmethod
    def define_schema(cls):
        default = predictors.DEFAULT_PREDICTOR
        return io.Schema(
            node_id='Star7VedaSparseAttention',
            display_name='MiniMax H3 VEDA 稀疏注意力 - Star7',
            category='Star7/MiniMax H3',
            search_aliases=['veda', 'sparse attention', 'minimax h3 speed',
                            'accelerate', 'faster video'],
            description='MiniMax H3 稀疏注意力。SM75 使用 Star7 CUDA 路线，SM80 及以上使用原版 Triton 路线；可连接普通采样或 Star7 分块节点。关闭时直接透传模型。',
            inputs=[
                io.Model.Input('model', display_name='模型', tooltip='连接模型或 LoRA 加载器的模型输出。'),
                io.Combo.Input('predictor', options=_predictor_options(), default=default,
                               display_name='VEDA 预测模型', tooltip='选择 models/veda 中的本地预测模型。'),
                io.String.Input('generated_sparsity', default='90%', advanced=True,
                                display_name='生成区域稀疏度', tooltip='90% 表示跳过 90% 的候选注意力块。降低数值更接近完整注意力，但速度更慢；整数如 24 表示保留 24 个块。'),
                io.String.Input('reference_sparsity', default='90%', advanced=True,
                                display_name='参考区域稀疏度', tooltip='首尾帧、参考图像及视频的稀疏度。0% 表示参考区域使用完整注意力。'),
                io.String.Input('full_attention_layers', default='', advanced=True,
                                display_name='完整注意力层', tooltip='从 0 开始的层编号，如 0, 1, 47-49；留空表示所有层使用稀疏注意力。'),
                io.String.Input('full_attention_steps', default='', advanced=True,
                                display_name='完整注意力步', tooltip='从 0 开始的采样步编号，如 0 表示第一步；留空表示所有步使用稀疏注意力。'),
                io.Boolean.Input('verbose', default=False, advanced=True,
                                 display_name='详细日志', label_on='开启', label_off='关闭', tooltip='在控制台输出耗时和诊断信息。'),
                # Append for old positional workflows; the frontend displays this first.
                io.Boolean.Input('enabled', default=True, optional=True,
                                 display_name='启用 VEDA', label_on='开启', label_off='关闭', tooltip='关闭时直接透传输入模型，不加载预测模型或安装注意力补丁。'),
            ],
            outputs=[io.Model.Output(display_name='模型', tooltip='启用时输出 VEDA 模型，关闭时原样输出输入模型。')],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, predictor, generated_sparsity='90%',
                reference_sparsity='90%', full_attention_layers='',
                full_attention_steps='', verbose=False, enabled=True) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model)
        hidden = getattr(cls, 'hidden', None)
        node_id = getattr(hidden, 'unique_id', None)
        status = veda_status.NodeStatus(node_id)
        try:
            bundle = _bundle(_predictor_path(predictor))
        except veda_bundle.BundleError as error:
            raise ValueError(str(error)) from error
        num_layers, _, _ = _check_model(model, bundle)
        settings = veda_settings.VedaSettings(
            generated=veda_settings.parse_sparsity(generated_sparsity,
                                                   'generated_sparsity'),
            reference=veda_settings.parse_sparsity(reference_sparsity,
                                                   'reference_sparsity'),
            dense_layers=veda_settings.parse_index_list(
                full_attention_layers, 'full_attention_layers'),
            dense_steps=veda_settings.parse_index_list(
                full_attention_steps, 'full_attention_steps'),
            verbose=verbose)
        missing = sorted(i for i in settings.dense_layers if i >= num_layers)
        if missing:
            raise ValueError(f'full_attention_layers: this model has blocks '
                             f'0-{num_layers - 1}; '
                             f'{veda_settings.format_index_list(missing)} '
                             'do not exist.')
        patched, _ = comfy_patch.apply(model, bundle, settings, node_id)
        device = comfy.model_management.get_torch_device()
        info = hardware.describe(device)
        probe = backends.probe(device)
        usable = [display for _, display, error in probe if error is None]
        lines = [f'Star7 VEDA ready · {usable[0] if usable else "full attention"}'
                 f' · {info.short_name}',
                 f'Sparsity: {settings.describe()}']
        full = settings.describe_full_attention()
        if full:
            lines.append(f'Full attention: {full}')
        if info.kind == 'cuda' and any(error for _, _, error in probe):
            lines.append('Tip: pip install triton (triton-windows on '
                         'Windows) for the sparse kernel')
        if verbose:
            lines.append(f'Predictor: {bundle.describe()}')
            lines += [f'  {name}: {error or "available"}'
                      for name, _, error in probe]
        if _other_sparse_node(model):
            status.warn('ComfyUI\'s "Model Sparse Attention" node is also '
                        'applied; on H3 it replaces the attention blocks, so '
                        'Veda would not run. Remove one of the two.')
        else:
            status.show('\n'.join(lines))
        return io.NodeOutput(patched)


class VedaExtension(ComfyExtension):
    async def get_node_list(self):
        return [VedaSparseAttention]


async def comfy_entrypoint() -> VedaExtension:
    register_model_folder()
    return VedaExtension()
