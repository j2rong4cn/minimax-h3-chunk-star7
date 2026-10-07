"""Isolate projection chunks, tiled packing and the sparse kernel on one input."""
import pathlib
import sys
import time
import types
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parents[1])]
import nodes
import comfy.model_management as mm
import comfy.quant_ops as quant_ops

package = types.ModuleType('veda_chunk_test')
package.__path__ = [str(ROOT / 'vendor' / 'veda')]
sys.modules[package.__name__] = package
from veda_chunk_test.backends import torch_gather
from veda_chunk_test.backends.sampled_reference import sampled_outputs, numerical_error
from veda_chunk_test.core import tiling, selection


def check(device, dtype, long=False):
    torch.manual_seed(19)
    heads, dim = 7, 128
    grid, global_rows = ((9, 64, 70), 513) if long else ((3, 7, 37), 29)
    sequence = global_rows + grid[0] * grid[1] * grid[2]
    x = torch.randn(sequence, 512, device=device, dtype=dtype)
    weight = torch.randn(3 * heads * dim, 512, device=device, dtype=dtype) * .02
    attn = SimpleNamespace(heads=heads, head_dim=dim,
        qkv_proj=lambda value: torch.nn.functional.linear(value, weight),
        q_norm=SimpleNamespace(weight=torch.ones(dim, device=device, dtype=dtype), eps=1e-6),
        k_norm=SimpleNamespace(weight=torch.ones(dim, device=device, dtype=dtype), eps=1e-6))
    angles = torch.randn(sequence, 48, device=device)
    c, s = angles.cos(), angles.sin()
    freqs = torch.stack((c, -s, s, c), dim=-1).reshape(1, sequence, 1, 48, 2, 2)
    nodes._CONFIG.update(verbose=False, reuse_mlp_weights=False,
                         auto_halve_on_oom=False, node_id=None)
    layout = tiling.build_tile_layout(
        [tiling.TiledSpan(global_rows, grid, tiling.TileShape(2, 8, 8))], sequence, device)
    head_ids = torch.arange(heads, device=device)

    # Match the unpatched H3 projection and fused norm/RoPE, before any chunk assembly.
    raw = attn.qkv_proj(x)
    q, k, v = (part.view(1, sequence, heads, dim)
               for part in raw.split(heads * dim, dim=-1))
    quant_ops.ck.rms_rope_split_half_(q, k, freqs, attn.q_norm.weight,
                                     attn.k_norm.weight, epsilon=1e-6, rot_dim=96)
    reference = tuple(part.transpose(1, 2) for part in (q, k, v))
    tiled_reference = tuple(tiling.gather_tiles(part[0].transpose(0, 1), layout, head_ids)
                            for part in reference)
    mask = torch.eye(layout.n_tiles, device=device, dtype=torch.bool)
    mask[:, ::11 if long else 3] = True
    mask = mask[None].expand(heads, -1, -1).clone() & layout.kv_ok
    backend = torch_gather.TorchGatherBackend() if device.type == 'cuda' and torch.cuda.get_device_capability(device) == (7, 5) else None

    failures = []
    for chunk in ((0, 8192) if long else (0, 256, 512, 1024)):
        nodes._CONFIG['effective_qkv_chunk_tokens'] = chunk
        output = nodes._prepare_h3_qkv_chunked(attn, x, freqs, mm, quant_ops)
        errors = []
        for expected, actual in zip(reference, output):
            error = float((actual.float() - expected.float()).norm() / expected.float().norm())
            assert error < .003, (dtype, chunk, 'projection/norm/RoPE', error)
            errors.append(error)
        tiled = tuple(tiling.gather_tiles(part[0].transpose(0, 1), layout, head_ids)
                      for part in output)
        for expected, actual in zip(tiled_reference, tiled):
            assert actual.is_contiguous()
            torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
            assert not torch.count_nonzero(actual[layout.pad_slots])
        print(f'{device} {dtype} QKV chunk={chunk}: relative-L2={errors}; packing/padding PASS')

        if device.type != 'cuda':
            continue
        if backend is not None and dtype not in backend.dtypes:
            tiled = tuple(part.to(backend.dtypes[0]) for part in tiled)
        # Fix the mask so predictor decisions cannot conceal a kernel/layout error.
        variants = [('CUDA', {})] if backend is not None else [
            ('Triton original', dict(fp32_pv=False, num_stages=3)),
            ('Triton single-stage', dict(fp32_pv=False, num_stages=1)),
            ('Triton FP32 accumulation', dict(fp32_pv=True, num_stages=3))]
        for group in ((7,) if long else (1, 3, 7)):
            for name, options in variants:
                assembled = torch.empty_like(tiled[0])
                for start in range(0, heads, group):
                    stop = min(start + group, heads)
                    group_qkv = [part[:, start:stop].contiguous() for part in tiled]
                    if backend is not None:
                        call = lambda: backend.attend(*group_qkv, mask[start:stop], layout)
                    else:
                        from veda_chunk_test.kernels.sage import sparse_int8
                        index, count = selection.tile_index_list(mask[start:stop])
                        call = lambda: sparse_int8.attend(*group_qkv, index, count, layout.valid_count, **options)
                    call()  # Compile/warm up before measuring the kernel call.
                    torch.cuda.synchronize(device)
                    began = time.perf_counter()
                    result = call()
                    torch.cuda.synchronize(device)
                    milliseconds = (time.perf_counter() - began) * 1000
                    assembled[:, start:stop] = result
                got, want = sampled_outputs(*tiled, mask, layout.valid_count, assembled)
                error, maximum, cosine = numerical_error(got, want)
                passed = error < .08 and cosine > .98
                if not passed:
                    failures.append((chunk, group, name, error, maximum, cosine))
                print(f'  {name} heads/group={group}: relative-L2={error:.5f}, cosine={cosine:.5f}, last group={milliseconds:.2f} ms {"PASS" if passed else "FAIL"}')
    assert not failures, failures


if __name__ == '__main__':
    if torch.cuda.is_available():
        long = '--long' in sys.argv
        for dtype in ((torch.bfloat16,) if long else (torch.float16, torch.bfloat16)):
            check(torch.device('cuda'), dtype, long)
    else:
        check(torch.device('cpu'), torch.float32)
