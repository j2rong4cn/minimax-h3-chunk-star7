"""Star7's compact VDN-H3 integration node.

The VDN execution core is an adapted Apache-2.0 port under
``vendor.star7_vdn``.  This module deliberately keeps the public node small:
the trained model specification owns the attention geometry and Star7 owns the
safe memory policy.  Expert ablation controls do not belong in the one-click
node because changing them no longer reproduces the released VDN model.
"""

import logging
import os
import torch

from .vendor.star7_vdn import spec
from .vendor.star7_vdn.nodes import _apply_vdn


_LOG = logging.getLogger("MiniMaxH3VDNStar7")

MODE_DMD8 = "8步加速 / 8-step"
MODE_BASE50 = "50步原始 / 50-step"
_EMPTY_MODEL = "请将完整 VDN stage 文件夹放入 ComfyUI/models/vdn"


def _checkpoint_choices():
    names = spec.list_vdn_checkpoints()
    return names or [_EMPTY_MODEL]


def _has_turbo_adapter(checkpoint):
    """Detect DMD8 by its trained adapter, not by the user's folder name."""
    path = spec.resolve_vdn_checkpoint(checkpoint)
    adapter = os.path.join(path, "adapters", "turbo")
    config_present = any(
        os.path.isfile(os.path.join(adapter, name))
        for name in ("adapter_spec.json", "adapter_config.json")
    )
    return config_present and os.path.isfile(
        os.path.join(adapter, "adapter_model.safetensors")
    )


class MiniMaxH3VDNStar7:
    """Apply the complete trained VDN-H3 stage to a loaded H3 MODEL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "连接未预先融合 Turbo/其他 LoRA 的 MiniMax H3 基础模型；本节点放在模型载入与 Star7 分块节点之间。",
                }),
                "vdn_checkpoint": (_checkpoint_choices(), {
                    "tooltip": "选择完整 VDN stage 目录；不能只放一个 LoRA 或 safetensors 文件。",
                }),
                "inference_mode": ([MODE_DMD8, MODE_BASE50], {
                    "default": MODE_DMD8,
                    "tooltip": "8步加速会加载 default + turbo 两组配套 Adapter；50步原始只加载 default。",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("模型",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/Star7"
    DESCRIPTION = (
        "加载完整 VDN-H3 训练阶段并安装混合注意力。下游 Star7 分块节点会自动保留 VDN 注意力。"
    )

    def apply(self, model, vdn_checkpoint, inference_mode):
        if vdn_checkpoint == _EMPTY_MODEL:
            raise FileNotFoundError(
                "No VDN model found. Install a complete stage in ComfyUI/models/vdn; "
                "keep model_spec.json, linear_branch, and adapters."
            )

        dmd8 = inference_mode == MODE_DMD8
        if dmd8 and not _has_turbo_adapter(vdn_checkpoint):
            raise ValueError(
                f"8-step VDN mode requires a complete turbo adapter, but  {vdn_checkpoint!r} is missing it. "
                "Install a complete DMD8 stage or select the original 50-step mode."
            )

        # At node execution the base patcher may still be CPU-resident. A free-
        # VRAM probe would then incorrectly cache the whole VDN branch on a
        # 2080 Ti / 24 GB card before the H3 weights and activations arrive.
        small_gpu = bool(
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            <= 24 * (1 << 30)
        )
        branch_policy = "stream" if small_gpu else "auto"
        buffer_policy = "off" if small_gpu else "auto"

        patched, = _apply_vdn(
            model=model,
            vdn_checkpoint=vdn_checkpoint,
            strength=1.0,
            lora_mode="merge",
            branch_weights=branch_policy,
            attention_backend="grouped",
            verbose=False,
            apply_turbo_adapter=dmd8,
            fast_kernels=False,
            retain_buffers=buffer_policy,
        )
        options = patched.model_options.setdefault("transformer_options", {})
        options["star7_vdn_h3"] = {
            "checkpoint": str(vdn_checkpoint),
            "mode": "dmd8" if dmd8 else "base50",
            "expected_steps": 8 if dmd8 else 50,
            "attention": "vdn_hybrid_grouped",
        }
        _LOG.info(
            "[Star7 VDN] Ready | stage=%s | mode=%s | recommended-steps=%d | "
            "adapter=merge | branch=%s | attention=VDN hybrid grouped",
            vdn_checkpoint,
            "DMD8" if dmd8 else "Base50",
            8 if dmd8 else 50,
            branch_policy,
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3VDNStar7": MiniMaxH3VDNStar7,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3VDNStar7": "MiniMax H3 VDN 加速 - Star7",
}
