from types import SimpleNamespace

import torch

from . import h3_vdn_star7 as vdn_module
from . import nodes as star7_nodes
from .nodes import _model_has_vdn_attention, _prepare_h3_qkv_chunked, _step_backend_label
from .vendor.star7_vdn import apply as vdn_apply
from .vendor.star7_vdn import branch as vdn_branch


def test_tiled_temporal_shift_activation_matches_original(monkeypatch):
    monkeypatch.setattr(vdn_branch, "_TEMPORAL_SHIFT_SCRATCH_BYTES", 128)
    for frames, height, width, l2norm in ((1, 2, 3, False), (2, 3, 3, True),
                                           (11, 4, 3, False), (11, 4, 3, True)):
        torch.manual_seed(frames + height)
        heads, head_dim = 2, 8
        x = torch.randn(frames, height * width, heads * head_dim)
        w = torch.randn(heads * head_dim, 5)
        expected = vdn_branch._activate(
            vdn_branch._temporal_shift(x, w, 5).reshape(-1, heads, head_dim),
            l2norm,
        )
        actual = vdn_branch._temporal_shift_activated(
            x, w, 5, heads, head_dim, l2norm,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_turbo_adapter_detection_uses_files_not_folder_name(tmp_path, monkeypatch):
    stage = tmp_path / "renamed-by-user"
    adapter = stage / "adapters" / "turbo"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"test")
    monkeypatch.setattr(vdn_module.spec, "resolve_vdn_checkpoint", lambda _: str(stage))
    assert vdn_module._has_turbo_adapter("anything")


def test_incomplete_turbo_adapter_is_not_accepted(tmp_path, monkeypatch):
    stage = tmp_path / "stage-dmd-but-incomplete"
    adapter = stage / "adapters" / "turbo"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(vdn_module.spec, "resolve_vdn_checkpoint", lambda _: str(stage))
    assert not vdn_module._has_turbo_adapter("anything")


def test_chunk_detects_vdn_from_marker_or_attention_patch():
    marked = SimpleNamespace(
        model_options={"transformer_options": {"star7_vdn_h3": {"mode": "dmd8"}}},
        object_patches={},
    )
    assert _model_has_vdn_attention(marked)

    def vdn_forward():
        pass

    vdn_forward._vdn_forward = True
    patched = SimpleNamespace(
        model_options={},
        object_patches={"diffusion_model.blocks.0.attn.forward": vdn_forward},
    )
    assert _model_has_vdn_attention(patched)
    assert not _model_has_vdn_attention(
        SimpleNamespace(model_options={}, object_patches={})
    )


def test_step_backend_log_names_vdn_instead_of_existing(monkeypatch):
    monkeypatch.setitem(star7_nodes._CONFIG, "attention_backend", "existing")
    assert _step_backend_label({"star7_vdn_h3": {"mode": "dmd8"}}) == (
        "VDN hybrid", "VDN"
    )


def test_chunked_qkv_can_capture_raw_rows_for_vdn():
    attention = SimpleNamespace(
        heads=2,
        head_dim=2,
        qkv_proj=torch.nn.Linear(4, 12, bias=False),
        q_norm=torch.nn.RMSNorm(2),
        k_norm=torch.nn.RMSNorm(2),
    )
    captured = []
    quant_ops = SimpleNamespace(
        ck=SimpleNamespace(rms_rope_split_half_=None)
    )
    q, k, v = _prepare_h3_qkv_chunked(
        attention,
        torch.randn(7, 4),
        None,
        None,
        quant_ops,
        output_layout="BTHD",
        raw_capture=lambda start, end, qr, kr, vr: captured.append(
            (start, end, qr.shape, kr.shape, vr.shape)
        ),
    )
    assert q.shape == k.shape == v.shape == (1, 7, 2, 2)
    assert captured == [
        (0, 7, torch.Size([7, 2, 2]), torch.Size([7, 2, 2]), torch.Size([7, 2, 2]))
    ]


def test_pruned_base_reinjects_both_adapters_even_in_merge_mode(monkeypatch):
    path = "blocks.0.adaln_proj.linear"
    converted = {
        name: {path: (torch.ones(2, 4), torch.ones(6, 2), scale)}
        for name, scale in (("default", 1.0), ("turbo", 0.5))
    }
    captured = []

    class Patcher:
        model = SimpleNamespace(state_dict=lambda: {})

        def get_model_object(self, _):
            return SimpleNamespace()

        def add_patches(self, loaded, strength):
            return loaded.keys()

    monkeypatch.setattr(vdn_apply, "_is_pruned_base", lambda _: True)
    monkeypatch.setattr(vdn_apply.comfy.lora, "load_lora", lambda *a, **kw: {
        "diffusion_model." + path + ".weight": object()
    })
    monkeypatch.setattr(vdn_apply, "_inject_adaln_egrid", lambda patcher, dm, parts: (
        captured.append(parts)
    ))
    report = vdn_apply.apply_adapters(Patcher(), converted, 1.0, "merge")
    assert len(captured) == 1
    assert len(captured[0][path]) == 2
    assert all("curve adapters" in description for description in report.values())


def test_chunked_qkv_capture_matches_direct_projection_across_boundaries():
    attention = SimpleNamespace(
        heads=2,
        head_dim=2,
        qkv_proj=torch.nn.Linear(4, 12, bias=False),
        q_norm=torch.nn.RMSNorm(2),
        k_norm=torch.nn.RMSNorm(2),
    )
    x = torch.randn(513, 4)
    projected = attention.qkv_proj(x).reshape(513, 3, 2, 2)
    captured = []
    old_chunk = star7_nodes._CONFIG["effective_qkv_chunk_tokens"]
    star7_nodes._CONFIG["effective_qkv_chunk_tokens"] = 256
    try:
        _prepare_h3_qkv_chunked(
            attention,
            x,
            None,
            None,
            SimpleNamespace(ck=SimpleNamespace(rms_rope_split_half_=None)),
            output_layout="BTHD",
            raw_capture=lambda start, end, q, k, v: captured.append(
                (start, end, q.clone(), k.clone(), v.clone())
            ),
        )
    finally:
        star7_nodes._CONFIG["effective_qkv_chunk_tokens"] = old_chunk
    assert [(start, end) for start, end, *_ in captured] == [
        (0, 256), (256, 512), (512, 513)
    ]
    for start, end, q, k, v in captured:
        torch.testing.assert_close(q, projected[start:end, 0])
        torch.testing.assert_close(k, projected[start:end, 1])
        torch.testing.assert_close(v, projected[start:end, 2])


def test_vdn_24gb_card_streams_branch_instead_of_trusting_unloaded_vram(monkeypatch):
    calls = []
    monkeypatch.setattr(vdn_module, "_has_turbo_adapter", lambda _: True)
    monkeypatch.setattr(vdn_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vdn_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        vdn_module.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=22 * (1 << 30)),
    )
    monkeypatch.setattr(vdn_module, "_apply_vdn", lambda **kwargs: (
        calls.append(kwargs) or SimpleNamespace(model_options={}),
    ))
    vdn_module.MiniMaxH3VDNStar7().apply(
        object(), "stage", vdn_module.MODE_DMD8
    )
    assert calls[0]["branch_weights"] == "stream"
    assert calls[0]["retain_buffers"] == "off"
