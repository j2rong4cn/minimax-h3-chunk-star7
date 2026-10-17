import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest import TestCase

import torch


ROOT = Path(__file__).resolve().parent


def test_internal_quantization_formats_survive_renaming(tmp_path):
    import json
    from safetensors.torch import save_file

    module = _load_module()
    prefix = "blocks.0.attn.qkv_proj."
    packed = {
        "video_patch_proj.weight": torch.zeros(128, 96),
        prefix + "qweight": torch.zeros(128, 64, dtype=torch.int8),
        prefix + "wscales": torch.ones(2, 128, dtype=torch.bfloat16),
        prefix + "smooth_factor": torch.ones(128, dtype=torch.bfloat16),
    }
    qf = {
        **packed,
        prefix + "proj_down": torch.zeros(128, 128, dtype=torch.int8),
        prefix + "proj_up": torch.zeros(128, 128, dtype=torch.int8),
        prefix + "resq_vr": torch.zeros(128, 128, dtype=torch.int8),
        prefix + "resq_vr_scale": torch.ones(128),
        prefix + "wgsums": torch.zeros(2, 128, dtype=torch.bfloat16),
    }
    path = tmp_path / "plain.safetensors"
    save_file(qf, str(path), metadata={"quantfunc_metadata_kv": "1"})
    before = module.inspect_h3_checkpoint_format(path)
    renamed = tmp_path / "unrelated-name.bin"
    path.rename(renamed)
    assert module.inspect_h3_checkpoint_format(renamed) == before
    assert before["family"] == "quantfunc_h3_int4"
    with TestCase().assertRaisesRegex(ValueError, "native forward validation"):
        module._check_h3_checkpoint_format(renamed)
    save_file(qf, str(path))
    assert module.inspect_h3_checkpoint_format(path)["family"] == "quantfunc_int4_layout"
    with TestCase().assertRaisesRegex(ValueError, "quantfunc_int4_layout"):
        module._check_h3_checkpoint_format(path)

    public = {
        **packed,
        prefix + "proj_down": torch.zeros(128, 128, dtype=torch.float16),
        prefix + "proj_up": torch.zeros(128, 128, dtype=torch.float16),
    }
    save_file(public, str(path))
    assert module.inspect_h3_checkpoint_format(path)["family"] == "public_svdquant_int4"

    normal = tmp_path / "quantfunc-int4.safetensors"
    metadata = {"_quantization_metadata": json.dumps({"layers": {
        "blocks.0.mlp.fc1": {"format": "asym_w4a8_int8"},
        "blocks.0.mlp.fc2": {"format": "int8_tensorwise"},
    }})}
    save_file({"blocks.0.mlp.fc1.weight": torch.zeros(128, 64, dtype=torch.int8)},
              str(normal), metadata=metadata)
    detected = module._check_h3_checkpoint_format(normal)
    assert detected["family"] == "comfy_quantized"
    assert detected["formats"] == ("asym_w4a8_int8", "int8_tensorwise")

    descriptor = json.dumps({"format": "asym_w4a8_int8", "convrot": False}).encode()
    save_file({"blocks.0.mlp.fc1.weight": torch.zeros(128, 64, dtype=torch.int8),
               "blocks.0.mlp.fc1.comfy_quant": torch.tensor(list(descriptor), dtype=torch.uint8)},
              str(normal))
    assert module._check_h3_checkpoint_format(normal)["formats"] == ("asym_w4a8_int8",)
    save_file({"blocks.0.mlp.fc1.comfy_quant": torch.tensor(list(descriptor), dtype=torch.uint8)},
              str(normal), metadata={"_quantization_metadata": json.dumps({"layers": {
                  "blocks.0.mlp.fc1": {"format": "int8_tensorwise"}}})})
    with TestCase().assertRaisesRegex(ValueError, "Conflicting quantization metadata"):
        module._check_h3_checkpoint_format(normal)


def test_unmarked_and_conflicting_packed_formats_are_rejected(tmp_path):
    import json
    from safetensors.torch import save_file

    module = _load_module()
    path = tmp_path / "ordinary-int4.safetensors"
    save_file({"blocks.0.attn.qkv_proj.qweight": torch.zeros(128, 64, dtype=torch.int8)}, str(path))
    assert module.inspect_h3_checkpoint_format(path)["family"] == "unknown_packed"
    with TestCase().assertRaisesRegex(ValueError, "unknown_packed"):
        module._check_h3_checkpoint_format(path)
    save_file({"blocks.0.mlp.fc1.weight": torch.zeros(128, 64, dtype=torch.int8)}, str(path),
              metadata={"_quantization_metadata": json.dumps({"layers": {
                  "blocks.0.mlp.fc1": {"format": "unregistered_int4"}
              }})})
    with TestCase().assertRaisesRegex(ValueError, "unregistered_int4"):
        module._check_h3_checkpoint_format(path)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "star7_adaptive_loader_tests", ROOT / "adaptive_loader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adaptive_loader_uses_the_fp16_loader_interface():
    module = _load_module()
    with mock.patch(
        "folder_paths.get_filename_list", return_value=["h3.safetensors"]
    ):
        required = module.MiniMaxH3FP16LoaderStar7.INPUT_TYPES()["required"]
    assert set(required) == {"unet_name"}
    assert module.NODE_VERSION == "2.0.14"


def test_chunk_project_registers_an_independent_enhanced_loader_id():
    init_source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    web_source = (ROOT / "web" / "adaptive_loader.js").read_text(encoding="utf-8")

    assert 'NODE_CLASS_MAPPINGS["MiniMaxH3ChunkEnhancedLoaderStar7"]' in init_source
    assert 'NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3ChunkEnhancedLoaderStar7"]' in init_source
    assert 'NODE_CLASS_MAPPINGS["MiniMaxH3FP16LoaderStar7"]' not in init_source
    assert 'NODE_CLASS_MAPPINGS["MiniMaxH3EnhancedLoaderStar7"]' not in init_source
    assert '"MiniMax H3 增强载入"' in init_source
    assert '"MiniMax H3 Enhanced Loader"' in init_source
    assert 'const NODE = "MiniMaxH3ChunkEnhancedLoaderStar7";' in web_source
    assert '"MiniMax H3 增强载入 - Star7"' in web_source
    assert '"MiniMax H3 Enhanced Loader - Star7"' in web_source
    assert "beforeRegisterNodeDef(_nodeType, nodeData)" in web_source
    assert "nodeData.display_name = title();" in web_source


def test_adaptive_loader_preserves_attention_override_under_fp16():
    module = _load_module()
    seen = {}

    class Identity(torch.nn.Module):
        def forward(self, value):
            return value

    class Block:
        norm1 = Identity()
        norm2 = Identity()

        def adaln_proj(self, _t_emb):
            return (torch.zeros(1),) * 6

        def attn(self, _value, **_kwargs):
            raise AssertionError("the supplied attention override must be used")

        def mlp(self, value):
            seen["mlp"] = value.dtype
            return value

    minimax = SimpleNamespace(
        _mod_scale_shift=lambda value, *_args: value,
        _mod_gate=lambda residual, _gate, update, _segments: residual + update,
    )
    forward = module._block_forward(
        lambda *_args, **_kwargs: None, minimax
    )

    def attention(value, **_kwargs):
        seen["attention"] = value.dtype
        return value

    output = forward(
        Block(), torch.ones(1, 2), torch.zeros(1), [], None, {},
        attention=attention,
    )
    assert output.dtype is torch.float32
    assert seen == {"attention": torch.float16, "mlp": torch.float16}


def test_fp32_tile_reuse_preserves_fp16_mlp_math_and_foreign_contract():
    module = _load_module()
    torch.manual_seed(817)
    mlp = SimpleNamespace(
        fc1=torch.nn.Linear(16, 48, bias=False, dtype=torch.float16),
        fc2=torch.nn.Linear(24, 16, bias=False, dtype=torch.float16),
        _star7_fp16_chunk_input=True,
    )
    value = torch.randn(519, 16)
    marker = object()
    forward = module._mlp_forward(lambda _: marker)
    with torch.inference_mode():
        expected = forward(mlp, value.half())
        actual = forward(mlp, value)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    mlp._star7_fp16_chunk_input = False
    assert forward(mlp, value) is marker


def test_fp32_mlp_carrier_requires_current_chunk_option_and_owned_wrapper():
    module = _load_module()
    seen = []

    class MLP:
        _star7_fp16_chunk_input = True

        def __call__(self, value):
            seen.append(value.dtype)
            return value.float()

    block = SimpleNamespace(
        norm1=lambda value: value,
        norm2=lambda value: value,
        adaln_proj=lambda _: (torch.zeros(1),) * 6,
        attn=lambda value, **kw: value,
        mlp=MLP(),
    )
    minimax = SimpleNamespace(
        _mod_scale_shift=lambda value, *a: value,
        _mod_gate=lambda value, gate, update, segments: value + update,
    )
    forward = module._block_forward(None, minimax)
    outputs = []
    for options, owned in (({}, True), ({"star7_fp16_mlp_input_reuse": True}, True),
                           ({"star7_fp16_mlp_input_reuse": True}, False)):
        block.mlp._star7_fp16_chunk_input = owned
        outputs.append(forward(block, torch.ones(3, 4), None, [], None, options))
    assert seen == [torch.float16, torch.float32, torch.float16]
    assert all(torch.equal(outputs[0], value) for value in outputs[1:])


def test_real_loader_chunk_install_and_block_preserve_output():
    spec = importlib.util.spec_from_file_location(
        "star7_reuse_matrix", ROOT / "test_compatibility_matrix.py"
    )
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    adaptive = _load_module()
    patcher, diffusion = matrix._make_patcher()
    patched = adaptive._patch_h3_model(patcher)
    final = matrix.CHUNK.install_model_patch(
        patched, chunk_tokens=256, mlp_chunk_tokens=256,
        auto_halve_on_oom=True, verbose=False, disable_dynamic_prefetch=True,
        reuse_mlp_weights=True, attention_backend="existing",
    )
    options = final.model_options["transformer_options"]
    assert options["star7_fp16_mlp_input_reuse"]
    assert all(block.mlp._star7_fp16_chunk_input for block in diffusion.blocks)
    for path, value in final.object_patches.items():
        if not path.startswith("diffusion_model."):
            continue
        owner = diffusion
        parts = path.split(".")[1:]
        for part in parts[:-1]:
            owner = owner[int(part)] if part.isdigit() else getattr(owner, part)
        setattr(owner, parts[-1], value)
    torch.manual_seed(987)
    source = torch.randn(519, 16)
    t_emb = torch.randn(1, 8, dtype=torch.float16)
    with torch.inference_mode():
        baseline_options = dict(options, star7_fp16_mlp_input_reuse=False)
        expected = diffusion.blocks[0](source.clone(), t_emb, [(0, 519, 0)], None,
                                        transformer_options=baseline_options)
        actual = diffusion.blocks[0](source.clone(), t_emb, [(0, 519, 0)], None,
                                      transformer_options=options)
    assert torch.equal(actual, expected)
    disabled = matrix.CHUNK.install_model_patch(
        final, chunk_tokens=256, mlp_chunk_tokens=0,
        auto_halve_on_oom=True, verbose=False, disable_dynamic_prefetch=True,
        reuse_mlp_weights=True, attention_backend="existing",
    )
    assert not disabled.model_options["transformer_options"]["star7_fp16_mlp_input_reuse"]
