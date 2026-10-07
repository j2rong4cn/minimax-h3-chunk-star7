"""SM80+ layout, producer, audio protection and optional native GPU checks."""
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parents[1])]
import nodes
import sol_backend as backend
import comfy.model_management as mm
import comfy.quant_ops


def rope(query, key, freqs, qw, kw, epsilon, rot_dim):
    nodes._rms_rope_one_chunk_inplace(query, freqs, qw, epsilon, rot_dim)
    nodes._rms_rope_one_chunk_inplace(key, freqs, kw, epsilon, rot_dim)


def frequencies(length, device="cpu"):
    angles = torch.randn(1, length, 1, 64, device=device)
    c, s = angles.cos(), angles.sin()
    return torch.stack((c, -s, s, c), -1).reshape(1, length, 1, 64, 2, 2)


def check_layouts():
    packed = torch.randn(1, 129, 3, 2, 128, dtype=torch.bfloat16)
    parts = [packed[:, :, i] for i in range(3)]
    captured = []
    def native(q, k, v, **kwargs):
        captured.append((q, k, v, kwargs))
        return v.clone()
    with patch.object(backend, "_comfy_kitchen_sol_attn", return_value=native), \
         patch.object(backend, "_kitchen_sol_supports_strides", return_value=True):
        backend.run_official(*parts, sink_start=63, sink_tokens=2)
        assert all(a is b for a, b in zip(parts, captured[-1][:3]))
        assert captured[-1][3]["sink_blocks"] == [0, 2]
        offset = torch.randn(1 + 129 * 2 * 128, dtype=torch.bfloat16)[1:].view(1, 129, 2, 128)
        backend.run_official(offset, *parts[1:])
        assert captured[-1][0].data_ptr() % 16 == 0
        assert captured[-1][0] is not offset and torch.equal(captured[-1][0], offset)
    with patch.object(backend, "_comfy_kitchen_sol_attn", return_value=native), \
         patch.object(backend, "_kitchen_sol_supports_strides", return_value=False):
        backend.run_official(*parts)
        assert all(part.is_contiguous() for part in captured[-1][:3])
    vendor = SimpleNamespace(sol_attn=native, get_sol_attn_backend=lambda device: "triton")
    with patch.object(backend, "_comfy_kitchen_sol_attn", return_value=None), \
         patch.object(backend, "_official_module", return_value=vendor):
        backend.run_official(*parts)
        assert all(part.is_contiguous() for part in captured[-1][:3])
    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), \
         patch.object(backend, "_comfy_kitchen_sol_attn", return_value=native), \
         patch.object(backend, "_official_module", side_effect=AssertionError("unneeded vendor import")):
        assert backend.check_runtime_support(backend.SOL_SM86PLUS_BACKEND_NAME) == (8, 9)
    print("Kitchen stride/alignment contracts, legacy layouts and compiled-backend preflight: PASS")


def check_producer():
    original = dict(nodes._CONFIG)
    nodes._CONFIG.update(verbose=False, node_id=None, effective_qkv_chunk_tokens=256, auto_halve_on_oom=True)
    calls = []
    def project(value):
        calls.append(len(value))
        # Model tensorwise activation scaling can depend on projection rows.
        return torch.cat((value + len(value) * 0.001, value * 0.5, value * 0.25), -1)
    attention = SimpleNamespace(heads=2, head_dim=128, qkv_proj=project,
        q_norm=SimpleNamespace(weight=torch.ones(128, dtype=torch.bfloat16), eps=1e-6),
        k_norm=SimpleNamespace(weight=torch.ones(128, dtype=torch.bfloat16), eps=1e-6))
    observed = []
    def producer(factory, length, heads, freqs, weights, **kwargs):
        assert kwargs["kmean"] is None and kwargs["vscale"] is None
        assert kwargs["sink_q"] == [0, 0] and kwargs["token_aug"] == 0 and kwargs["tail"]
        for _ in range(2):
            sizes = [len(part) for part in factory()]
            assert sum(sizes) == length and all(size % 64 == 0 for size in sizes[:-1])
            observed.append(sizes)
        return torch.zeros(1, length, heads, 128, dtype=torch.bfloat16), None, None
    try:
        with patch.object(nodes, "_ORIGINAL_RMS_ROPE_SPLIT_HALF_INPLACE", rope):
            for length in (519, 1025):
                x = torch.randn(length, 256, dtype=torch.bfloat16) * 0.1
                freqs = frequencies(length)
                ranges = [(17, 90), (245, 300), (length - 40, length)]
                calls.clear()
                result, overrides = nodes._run_h3_sol_producer(attention, x, freqs, backend, producer,
                    ranges, 245, 55, mm, comfy.quant_ops)
                assert calls == observed[-1] * 2
                assert result.implementation.endswith("current-stats")
                calls.clear()
                q, k, v = nodes._prepare_h3_qkv_chunked(attention, x, freqs, mm, comfy.quant_ops,
                    output_dtype=torch.bfloat16, output_layout="BTHD")
                expected = nodes._dense_audio_query_overrides(q, k, v, ranges, layout="BTHD")
                for (a, b, actual), (ea, eb, reference) in zip(overrides, expected):
                    assert (a, b) == (ea, eb)
                    torch.testing.assert_close(actual, reference, atol=1e-4, rtol=0.02)
                protected = result.output.clone()
                nodes._apply_dense_audio_query_overrides(protected, overrides, layout="BTHD")
                assert torch.count_nonzero(protected[:, 90:245]) == 0
                # A projection OOM during the second pass restarts both passes.
                nodes._CONFIG["effective_qkv_chunk_tokens"] = 512
                count = 0
                def failing(value):
                    nonlocal count
                    count += 1
                    if count == 5:
                        raise RuntimeError("CUDA out of memory")
                    return project(value)
                if length == 1025:
                    with patch.object(attention, "qkv_proj", failing), \
                         patch.object(nodes, "_clear_cuda_after_oom"):
                        _, recovered = nodes._run_h3_sol_producer(attention, x, freqs, backend, producer,
                            ranges, 245, 55, mm, comfy.quant_ops)
                    assert nodes._CONFIG["effective_qkv_chunk_tokens"] == 256
                    q, k, v = nodes._prepare_h3_qkv_chunked(attention, x, freqs, mm, comfy.quant_ops,
                        output_dtype=torch.bfloat16, output_layout="BTHD")
                    expected = nodes._dense_audio_query_overrides(q, k, v, ranges, layout="BTHD")
                    for (_, _, actual), (_, _, reference) in zip(recovered, expected):
                        torch.testing.assert_close(actual, reference, atol=1e-4, rtol=0.02)
            nodes._CONFIG["effective_qkv_chunk_tokens"] = 512
            def workspace_oom(*args, **kwargs):
                raise RuntimeError("CUDA out of memory")
            try:
                nodes._run_h3_sol_producer(attention, x, freqs, backend, workspace_oom,
                    [], None, 0, mm, comfy.quant_ops)
            except RuntimeError:
                assert nodes._CONFIG["effective_qkv_chunk_tokens"] == 512
            else:
                raise AssertionError("Fixed producer workspace OOM must propagate")
        print("Two-pass producer, post-projection audio capture, dense full-KV merge and OOM restart: PASS")
    finally:
        nodes._CONFIG.clear()
        nodes._CONFIG.update(original)


def check_selection():
    original = dict(nodes._CONFIG)
    nodes._CONFIG.update(verbose=False, effective_qkv_chunk_tokens=256,
                         attention_backend=nodes.SOL_SM86PLUS_BACKEND_NAME)
    weight = torch.ones(128, dtype=torch.bfloat16)
    attention = SimpleNamespace(heads=2, head_dim=128, out_proj=lambda value: value,
        q_norm=SimpleNamespace(weight=weight), k_norm=SimpleNamespace(weight=weight))
    x = SimpleNamespace(shape=(519, 256), dtype=torch.bfloat16, is_cuda=True, device=torch.device("cuda"))
    parts = [torch.zeros(1, 519, 2, 128, dtype=torch.bfloat16) for _ in range(3)]
    result = backend.SolResult(parts[2], 9, 9, -1, -1, float("nan"), "test", 1.0)
    try:
        with patch.object(nodes, "_load_sol_backend", return_value=backend), \
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)), \
             patch.object(mm, "in_training", False), \
             patch.object(backend, "chunked_producer", return_value=object()) as lookup, \
             patch.object(nodes, "_run_h3_sol_producer", return_value=(result, [])) as streamed, \
             patch.object(nodes, "_prepare_h3_qkv_chunked", return_value=parts) as full, \
             patch.object(backend, "run_official", return_value=result), \
             patch.object(nodes.h3_preprocess, "memory_pressure", return_value=False) as pressure:
            freqs = frequencies(519)
            nodes._minimax_sol_forward(attention, x, freqs)
            full.assert_called_once(); lookup.assert_not_called(); streamed.assert_not_called()
            pressure.return_value = True
            full.reset_mock()
            nodes._minimax_sol_forward(attention, x, freqs)
            streamed.assert_called_once(); full.assert_not_called()
            streamed.reset_mock(); lookup.reset_mock()
            streamed.side_effect = nodes._SolProducerUnsupported("test projection dtype")
            nodes._minimax_sol_forward(attention, x, freqs)
            full.assert_called_once()
            streamed.reset_mock(); lookup.reset_mock()
            nodes._CONFIG["effective_qkv_chunk_tokens"] = 0
            nodes._minimax_sol_forward(attention, x, freqs)
            streamed.assert_not_called(); lookup.assert_not_called()
        import comfy_kitchen
        from comfy_kitchen.backends import cuda
        symbols = SimpleNamespace(sol_attn_plan=object(), sol_producer_begin=object(),
                                  sol_producer_chunk=object(), sol_attn_core=object())
        def compatible(*args, kmean=None, vscale=None, rope_eps=None, sink_blocks=None,
                       sink_q=None, tail=None, tau=None, scale=None, topk_ratio=None, token_aug=None):
            pass
        with patch.object(backend, "_comfy_kitchen_sol_attn", return_value=object()), \
             patch.object(cuda, "_C", symbols), \
             patch.object(comfy_kitchen, "sol_attn_chunked", compatible):
            assert backend.chunked_producer(torch.device("cuda")) is compatible
            del symbols.sol_producer_chunk
            assert backend.chunked_producer(torch.device("cuda")) is None
        print("Memory-pressure selection, single-chunk bypass and producer capability detection: PASS")
    finally:
        nodes._CONFIG.clear()
        nodes._CONFIG.update(original)


def check_custom_quantization():
    packed = torch.randn(1, 256, 3, 2, 128, dtype=torch.float16)
    parts = [packed[:, :, i].transpose(1, 2) for i in range(3)]
    for part in parts:
        torch.testing.assert_close(backend._block_mean_fp32(part), backend._block_mean_fp32(part.contiguous()), rtol=0, atol=0)
    counts = torch.full((1, 2, 4), 4, dtype=torch.int32)
    lut = torch.arange(4, dtype=torch.int32).expand(1, 2, 4, 4).contiguous()
    captured = []
    def routing(q, k, **kwargs):
        assert q is parts[0] and k is parts[1]
        return counts, lut, 1.0, backend._block_mean_fp32(k).half(), torch.zeros_like(counts), lut[..., :0]
    def quantize(value, block, multiplier):
        assert value.stride(-1) == 1
        captured.append(value.shape)
        return torch.zeros(value.shape, dtype=torch.int8), torch.ones(1, 2, (value.shape[2]+block-1)//block)
    def pool(value, block, output_dtype):
        assert output_dtype == torch.float32
        return torch.stack([value[:, :, start:start+block].float().mean(-2)
                            for start in range(0, value.shape[2], block)], dim=2)
    class Kernel:
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                args[11].zero_()
            return call
    with patch.object(torch.Tensor, "is_cuda", property(lambda value: True)), \
         patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)), \
         patch.object(backend, "build_custom_routing", routing), \
         patch.object(backend, "_load_sla_backend", return_value=SimpleNamespace(_quantize=quantize, _mean_pool=pool)), \
         patch.object(backend, "triton", object()), \
         patch.object(backend, "_sol_qk_int8_pv_int8_kernel", Kernel(), create=True):
        result = backend.run_custom_consume(list(parts), all_int8=True)
        assert result.output.shape == parts[0].shape and len(captured) == 5
    print("SM80+ custom INT8 routing views and direct strided quantization: PASS")


def check_native_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        print("SM80+ native GPU checks: SKIP (no SM80+ device)")
        return
    if backend.chunked_producer(torch.device("cuda")) is None:
        raise RuntimeError("Install a Kitchen build with the compiled Sol producer to run the GPU check")
    original = dict(nodes._CONFIG)
    length, heads = 4097, 4
    x = torch.randn(length, heads * 128, device="cuda", dtype=torch.bfloat16) * 0.1
    weights = torch.randn(3 * heads * 128, heads * 128, device="cuda", dtype=torch.bfloat16) * 0.01
    attention = SimpleNamespace(heads=heads, head_dim=128,
        qkv_proj=lambda value: F.linear(value, weights), out_proj=lambda value: value,
        q_norm=SimpleNamespace(weight=torch.ones(128, device="cuda", dtype=torch.bfloat16), eps=1e-6),
        k_norm=SimpleNamespace(weight=torch.ones(128, device="cuda", dtype=torch.bfloat16), eps=1e-6),
        _star7_sla_mod_segments=[(0, 128, 2), (128, 256, 0), (256, 320, 2), (320, length, 1)])
    try:
        nodes._CONFIG.update(verbose=False, effective_qkv_chunk_tokens=512,
                             attention_backend=nodes.SOL_SM86PLUS_BACKEND_NAME)
        freqs = frequencies(length, "cuda")
        with patch.object(nodes, "_load_sol_backend", return_value=backend):
            with patch.object(nodes.h3_preprocess, "memory_pressure", return_value=False):
                reference = nodes._minimax_sol_forward(attention, x, freqs)
            with patch.object(nodes.h3_preprocess, "memory_pressure", return_value=True):
                actual = nodes._minimax_sol_forward(attention, x, freqs)
        torch.testing.assert_close(actual, reference, atol=0.003, rtol=0.03)
        for start, end in ((0, 128), (256, 320)):
            torch.testing.assert_close(actual[start:end], reference[start:end], atol=1e-4, rtol=0.02)
        assert torch.isfinite(actual).all()
        print("SM80+ compiled Sol producer and dense audio equivalence: PASS")
    finally:
        nodes._CONFIG.clear()
        nodes._CONFIG.update(original)


if __name__ == "__main__":
    check_layouts()
    check_producer()
    check_selection()
    check_custom_quantization()
    check_native_gpu()
