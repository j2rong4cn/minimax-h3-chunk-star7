"""Keep per-frame AdaLN index gathers bounded without changing sample values."""
import ast
from pathlib import Path
from types import SimpleNamespace
import weakref

import torch

ROOT = Path(__file__).resolve().parent


def functions():
    ns = {"torch": torch}
    for path, names in (
        (ROOT.parents[1] / "comfy/ldm/minimax/model.py",
         {"_mod_row", "_mod_scale_shift", "_mod_gate"}),
        (ROOT / "adaptive_loader.py", {"_bounded_modulation", "_block_forward"}),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert len(selected) == len(names)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_indexed_modulation_matches_native_for_offsets_and_dtypes():
    ns = functions()
    torch.manual_seed(619)
    for dtype in (torch.float16, torch.float32):
        source = torch.randn(4131, 16, dtype=dtype)
        shift, scale, gate = (torch.randn(7, 16, dtype=dtype) for _ in range(3))
        other = torch.randn_like(source)
        segments = [(0, 13, 0), (13, 4114, torch.randint(0, 7, (4101,))),
                    (4114, 4131, torch.tensor(3))]
        for function, parameters in ((ns["_mod_scale_shift"], (shift, scale)),
                                     (ns["_mod_gate"], (gate, other))):
            calls = []

            def observe(value, *args):
                for a, b, row in args[-1]:
                    if isinstance(row, torch.Tensor) and row.ndim:
                        assert row.shape[0] <= 2048
                    calls.append((a, b))
                return function(value, *args)

            expected = function(source.clone(), *parameters, segments)
            carrier = source.clone()
            actual = ns["_bounded_modulation"](observe, carrier, parameters, segments)
            assert actual is carrier
            assert torch.equal(actual, expected)
            assert calls == [(0, 13), (13, 2061), (2061, 4109), (4109, 4114), (4114, 4131)]


def test_scalar_short_and_singleton_rows_keep_the_native_fast_path():
    ns = functions()
    rows = [(0, 10, 0), (10, 20, torch.tensor([1])), (20, 30, torch.arange(10))]
    calls = []
    value = torch.zeros(30, 4)

    def native(x, *args):
        calls.append(args[-1])
        return x

    assert ns["_bounded_modulation"](native, value, (), rows) is value
    assert calls == [rows]


def test_attention_temporaries_are_released_before_mlp_normalization():
    ns = functions()
    refs = []

    def attention(value, **kwargs):
        output = value.float()
        refs.extend([weakref.ref(value), weakref.ref(output)])
        return output

    def norm2(value):
        assert all(ref() is None for ref in refs), "Attention temporaries retained into MLP"
        return value.clone()

    block = SimpleNamespace(
        norm1=lambda value: value.clone(), norm2=norm2,
        adaln_proj=lambda _: (torch.zeros(1),) * 6,
        attn=attention, mlp=lambda value: value,
    )
    core = SimpleNamespace(_mod_scale_shift=lambda value, *args: value,
                           _mod_gate=lambda value, gate, other, segments: value + other)
    with torch.inference_mode():
        output = ns["_block_forward"](None, core)(block, torch.ones(3, 4), None, [], None)
    assert torch.equal(output, torch.full((3, 4), 4.0))
