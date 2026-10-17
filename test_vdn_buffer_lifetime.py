"""Exercise the VDN buffer ownership contract without loading VDN weights."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def test_vdn_dense_and_prepared_value_storage_and_list_input():
    source_path = Path(__file__).parent / "vendor/star7_vdn/hybrid.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_base_attention")
    torch.manual_seed(713)
    attn = SimpleNamespace(
        heads=2, head_dim=4,
        qkv_proj=torch.nn.Linear(8, 24, bias=False),
        out_proj=torch.nn.Linear(8, 8, bias=False),
        q_norm=torch.nn.RMSNorm(4), k_norm=torch.nn.RMSNorm(4),
    )
    source = torch.randn(37, 8)
    with torch.inference_mode():
        q, k, v = attn.qkv_proj(source).reshape(37, 3, 2, 4).unbind(1)
        q, k, v = attn.q_norm(q), attn.k_norm(k), v.clone()
        reference = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None]
        ).transpose(1, 2).reshape(37, 8)
        expected = attn.out_proj(reference)
        for prepared in (False, True):
            for use_list in (False, True):
                observed = []

                def attention(q_in, k_in, v_in, *args, **kwargs):
                    # Dense V must release the larger backing QKV allocation.
                    assert v_in.untyped_storage().nbytes() == v_in.numel() * v_in.element_size()
                    if prepared:
                        assert v_in.data_ptr() == v.data_ptr(), "Independent V was copied"
                    observed.append(True)
                    return F.scaled_dot_product_attention(q_in, k_in, v_in).transpose(1, 2).reshape(1, 37, 8)

                ns = dict(torch=torch, AttentionTensorContainer=lambda value: value,
                          optimized_attention=attention,
                          _star7_chunked_qkv=lambda *args: (q, k, v) if prepared else None)
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), ns)
                value = [source] if use_list else source
                actual = ns["_base_attention"](attn, value, None, {})
                assert observed
                assert torch.equal(actual, expected)
                if use_list:
                    assert value == []
