"""SM75 Sol fused preprocessing regression and optional paired benchmark."""
import argparse
import ctypes
import pathlib
import statistics
import sys
import time
from unittest.mock import patch
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parents[1])]
import sol_backend
import sol_sm75_backend as native


def configure_library(path):
    library = ctypes.CDLL(str(path.resolve()))
    original = native._load_library()
    library.star7_sla_sm75_quant_v_int8.argtypes = original.star7_sla_sm75_quant_v_int8.argtypes
    library.star7_sla_sm75_quant_v_int8.restype = original.star7_sla_sm75_quant_v_int8.restype
    loader = sys.modules["star7_sm75_backend"]
    with patch.object(loader, "_load", return_value=library):
        return native._load_library()


def check(reference, candidate):
    generator = torch.Generator(device="cuda").manual_seed(507)
    for batch, heads, length, tau, sink in (
        (1, 2, 1, 0.0, 0), (1, 2, 63, 1.0, 17),
        (1, 3, 519, 0.25, 128), (2, 2, 1025, 2.0, 129),
        (1, 4, 4096, 1.0, 0),
    ):
        q, k, v = [torch.randn(batch, heads, length, 128, device="cuda",
                              dtype=torch.float16, generator=generator) * 0.2 for _ in range(3)]
        if length == 63:
            q.zero_()
            k.zero_()
        with patch.object(native, "_load_library", return_value=reference):
            old = native.prepare(q, k, v, tau=tau, sink_tokens=sink)
            old_q, old_qs = native.quantize(q, 16)
            old_k, old_ks = native.quantize(k, 64)
        with patch.object(native, "_load_library", return_value=candidate):
            new = native.prepare(q, k, v, tau=tau, sink_tokens=sink, quantize_qk=True)
        for key in ("exact_mask", "row_count", "k_centroid", "v_centroid"):
            assert torch.equal(old[key], new[key]), (key, length)
        for key, expected in (("q_int8", old_q), ("q_scale", old_qs),
                              ("k_int8", old_k), ("k_scale", old_ks)):
            assert torch.equal(expected, new[key]), (key, length)
        assert new["minimum"] == old["minimum"] and new["maximum"] == old["maximum"]
        assert abs(new["density"] - old["density"]) < 1e-7
        valid = torch.arange(new["lut"].shape[-1], device="cuda") < new["row_count"][..., None]
        expected_mask = torch.zeros_like(new["exact_mask"], dtype=torch.bool)
        indices = new["lut"].masked_fill(~valid, 0).long()
        expected_mask.scatter_add_(-1, indices, valid)
        assert torch.equal(expected_mask, new["exact_mask"].bool())
        for all_int8 in (False, True):
            with patch.object(sol_backend, "_load_sm75_backend", return_value=native):
                with patch.object(native, "_load_library", return_value=reference):
                    expected = sol_backend.run_custom_consume([q, k, v], all_int8=all_int8,
                                                              tau=tau, sink_tokens=sink).output
                with patch.object(native, "_load_library", return_value=candidate):
                    actual = sol_backend.run_custom_consume([q, k, v], all_int8=all_int8,
                                                            tau=tau, sink_tokens=sink).output
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
            if getattr(reference, "star7_sol_sm75_prepare_quantized_routes", None) is not None:
                assert torch.equal(actual, expected), ("workspace-only change", length, all_int8)
        packed = torch.stack((q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)), dim=2)
        views = [packed[:, :, part].transpose(1, 2) for part in range(3)]
        with patch.object(native, "_load_library", return_value=candidate), \
             patch.object(sol_backend, "_load_sm75_backend", return_value=native):
            strided = native.prepare(*views, tau=tau, sink_tokens=sink, quantize_qk=True)
            for key in ("exact_mask", "row_count", "k_centroid", "v_centroid", "q_int8", "k_int8", "q_scale", "k_scale"):
                assert torch.equal(new[key], strided[key]), ("strided", key, length)
            actual = sol_backend.run_custom_consume(views, all_int8=True, tau=tau, sink_tokens=sink).output
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
    print("Fused Q/K quantization, centroids, routing, sinks, partial blocks and both PV modes: PASS")
    broadcast = [part[:, :, :1].expand_as(part) for part in (q, k, v)]
    with patch.object(native, "_load_library", return_value=candidate):
        expanded = native.prepare(*broadcast, tau=1.0, quantize_qk=True)
        dense = native.prepare(*(part.contiguous() for part in broadcast), tau=1.0, quantize_qk=True)
        for key in ("q_int8", "k_int8", "q_scale", "k_scale", "exact_mask", "row_count"):
            assert torch.equal(expanded[key], dense[key]), key
        invalid_counts = new["row_count"].clone().fill_(-1)
        try:
            native.run(old_q, old_k, v, old_qs, old_ks, invalid_counts, new["lut"])
        except ValueError as error:
            assert "row_count" in str(error)
        else:
            raise AssertionError("External invalid routing must still be rejected")
    print("Zero-stride inputs and external routing validation: PASS")


def check_node(reference, candidate):
    import nodes
    original = dict(nodes._CONFIG)
    x = torch.randn(519, 512, device="cuda", dtype=torch.float16) * 0.2
    attention = SimpleNamespace(heads=4, head_dim=128,
        qkv_proj=lambda value: torch.cat((value, value * 0.5, value * 0.25), dim=-1),
        q_norm=lambda value: value, k_norm=lambda value: value, out_proj=lambda value: value)
    try:
        nodes._CONFIG.update(verbose=False, effective_qkv_chunk_tokens=0,
                             attention_backend=nodes.SOL_SM75_ALL_INT8_BACKEND_NAME)
        with patch.object(nodes, "_load_sol_backend", return_value=sol_backend), \
             patch.object(sol_backend, "_load_sm75_backend", return_value=native):
            with patch.object(native, "_load_library", return_value=reference):
                expected = nodes._minimax_sol_forward(attention, x)
            with patch.object(native, "_load_library", return_value=candidate):
                actual = nodes._minimax_sol_forward(attention, x)
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
                nodes._CONFIG["attention_backend"] = nodes.HYBRID_SM75_CK_SOL_ALL_INT8_BACKEND_NAME
                schedule = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
                with patch.object(nodes, "_minimax_ck_int8_attention_forward", return_value="CK") as ck:
                    for step in range(4):
                        result = nodes._minimax_hybrid_attention_forward(attention, x,
                            transformer_options={"sample_sigmas": schedule, "sigmas": schedule[step:step+1]})
                        if step in (0, 3): assert result == "CK"
                        else: torch.testing.assert_close(result, actual, atol=2e-4, rtol=2e-3)
                    assert ck.call_count == 2
        print("Pure Sol node and CK/Sol/Sol/CK dispatch with native Sol execution: PASS")
    finally:
        nodes._CONFIG.clear()
        nodes._CONFIG.update(original)


def benchmark(reference, candidate):
    for length, heads in ((8192, 8), (32768, 8), (49152, 28)):
        generator = torch.Generator(device="cuda").manual_seed(507)
        tensors = [torch.randn(1, heads, length, 128, device="cuda", dtype=torch.float16,
                               generator=generator) for _ in range(3)]
        measurements = {}
        for label, library in (("reference", reference), ("fused", candidate)):
            with patch.object(native, "_load_library", return_value=library), \
                 patch.object(sol_backend, "_load_sm75_backend", return_value=native):
                for _ in range(2):
                    sol_backend.run_custom_consume(list(tensors), all_int8=True)
                times = []
                for _ in range(7):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    result = sol_backend.run_custom_consume(list(tensors), all_int8=True)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
                    peak = torch.cuda.max_memory_allocated() / 1024**2
                    del result
                measurements[label] = (round(statistics.median(times), 3), round(peak, 1))
        print(f"S={length} H={heads}: median ms / peak MiB {measurements}")
        del tensors
        packed = torch.randn(1, length, 3, heads, 128, device="cuda", dtype=torch.float16, generator=generator)
        views = [packed[:, :, part].transpose(1, 2) for part in range(3)]
        measurements = {}
        for label, library in (("reference-copy", reference), ("fused-view", candidate)):
            with patch.object(native, "_load_library", return_value=library), \
                 patch.object(sol_backend, "_load_sm75_backend", return_value=native):
                def run():
                    parts = [part.contiguous() for part in views] if label == "reference-copy" else list(views)
                    return sol_backend.run_custom_consume(parts, all_int8=True)
                for _ in range(2): run()
                times = []
                for _ in range(7):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    result = run()
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
                    peak = torch.cuda.max_memory_allocated() / 1024**2
                    del result
                measurements[label] = (round(statistics.median(times), 3), round(peak, 1))
        print(f"Strided S={length} H={heads}: median ms / peak MiB {measurements}")
        del views, packed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise SystemExit("This regression requires an SM75 CUDA GPU")
    reference = configure_library(args.reference)
    candidate = configure_library(args.candidate)
    check(reference, candidate)
    check_node(reference, candidate)
    if args.benchmark:
        benchmark(reference, candidate)
