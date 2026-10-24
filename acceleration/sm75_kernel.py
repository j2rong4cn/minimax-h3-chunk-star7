"""SM75 Spark global-Reweight CUDA ABI; no dependency installation/JIT at import."""
import ctypes
import hashlib
import json
import sys
from pathlib import Path

import torch

_LIBRARY = None
ROOT = Path(__file__).resolve().parents[1]


def load():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    platform = 'windows_x64' if sys.platform == 'win32' else 'linux_x86_64'
    manifest = json.loads((ROOT / 'bin/spark_sm75_manifest.json').read_text('utf-8'))
    entry = manifest['platforms'].get(platform)
    if entry is None:
        raise RuntimeError('No compiled Spark SM75 binary for this platform')
    path = (ROOT / 'bin' / entry['file']).resolve()
    if not path.is_relative_to((ROOT / 'bin').resolve()):
        raise RuntimeError('Invalid Spark binary path')
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise RuntimeError('Spark SM75 binary checksum mismatch')
    lib = ctypes.CDLL(str(path))
    lib.star7_spark_sm75_abi_version.restype = ctypes.c_int
    if lib.star7_spark_sm75_abi_version() != 1:
        raise RuntimeError('Spark SM75 binary ABI mismatch')
    ptr, integer = ctypes.c_uint64, ctypes.c_int
    for name, signature in (
        ('star7_spark_sm75_summaries', [ptr]*6+[integer]*4+[ptr]),
        ('star7_sol_sm75_quantize', [ptr]*3+[integer]*4+[ptr]),
        ('star7_sol_sm75_pack_ordered_lut', [ptr]*2+[integer]*3+[ptr]),
        ('star7_spark_sm75_attention', [ptr]*13+[integer]*6+[ctypes.c_float,ptr]),
        ('star7_sla_sm75_quant_v_int8', [ptr]*3+[integer]*5+[ctypes.c_int64]*3+[integer,ptr]),
        ('star7_spark_sm75_attention_fast', [ptr]*17+[integer]*8+[ctypes.c_float,ptr]),
    ):
        fn = getattr(lib, name); fn.argtypes = signature; fn.restype = integer
    _LIBRARY = lib
    return lib


def check(code, operation):
    if code:
        raise RuntimeError(f'Spark SM75 CUDA {operation} failed: cudaError={code}')


def validate(q, k, v):
    if q.ndim != 4 or q.shape[-1] != 128 or min(q.shape) <= 0:
        raise ValueError('Spark SM75 needs nonempty BHND tensors, head_dim=128')
    if any(x.shape != q.shape or x.dtype != torch.float16 or x.device != q.device
           or not x.is_cuda or not x.is_contiguous() for x in (q,k,v)):
        raise ValueError('Spark SM75 needs equal contiguous FP16 CUDA Q/K/V')
    if torch.cuda.get_device_capability(q.device) != (7,5):
        raise ValueError('This compiled Spark binary targets SM75 only')
    if q.shape[0]*q.shape[1] > 65535 or q.shape[1]*q.shape[2]*128 >= 2**31:
        raise ValueError('Spark SM75 launch dimensions exceed the supported range')


def summaries(k, v, anchor):
    validate(k,k,v)
    b,h,n,d = k.shape
    if anchor.shape != (b,h,d) or anchor.device != k.device or anchor.dtype != torch.float32 or not anchor.is_contiguous():
        raise ValueError('Spark anchor must be contiguous FP32 BHD on the same device')
    leaves=(n+63)//64; padded=(leaves+63)//64*64
    ak=torch.empty((b,h,padded,d),device=k.device,dtype=k.dtype)
    av=torch.empty_like(ak)
    lm=torch.empty((b,h,padded),device=k.device,dtype=torch.float32)
    with torch.cuda.device(k.device):
        stream=torch.cuda.current_stream(k.device).cuda_stream
        check(load().star7_spark_sm75_summaries(
            *[x.data_ptr() for x in (k,v,anchor,ak,av,lm)],b,h,n,padded,stream),'summaries')
    return ak,av,lm


def quantize(x, block):
    b,h,n,d=x.shape
    out=torch.empty_like(x,dtype=torch.int8)
    scale=torch.empty((b,h,(n+block-1)//block),device=x.device,dtype=torch.float32)
    with torch.cuda.device(x.device):
        check(load().star7_sol_sm75_quantize(x.data_ptr(),out.data_ptr(),scale.data_ptr(),
            b,h,n,block,torch.cuda.current_stream(x.device).cuda_stream),'quantize')
    if block == 16 and scale.shape[2] % 4:
        # The Q64 consumer uses four warp scales per query block, including
        # inactive warps in the final partial block. Keep the head stride exact.
        scale=torch.nn.functional.pad(scale,(0,4-scale.shape[2]%4),value=1.).contiguous()
    return out,scale


def consume(q,k,v,mask,ak,av,lm):
    """Single online-softmax merge of exact blocks and weighted skipped blocks."""
    validate(q,k,v)
    b,h,n,d=q.shape; leaves=(n+63)//64; padded=ak.shape[2]
    if mask.shape != (b,h,leaves,leaves) or mask.device != q.device or mask.dtype != torch.uint8 or not mask.is_contiguous():
        raise ValueError('Invalid exact-block mask')
    for x in (ak,av):
        if x.shape != (b,h,padded,128) or padded<leaves or padded%64 or x.dtype!=q.dtype or x.device!=q.device or not x.is_contiguous():
            raise ValueError('Invalid weighted summary tensor')
    if lm.shape!=(b,h,padded) or lm.dtype!=torch.float32 or lm.device!=q.device or not lm.is_contiguous():
        raise ValueError('Invalid weighted log mass')
    # Fixed-width sorted LUT avoids GPU->CPU synchronization for route lengths.
    # RowCount bounds all reads, so unused entries may contain the sentinel.
    lut=torch.empty(mask.shape,device=q.device,dtype=torch.int32)
    rows=mask.sum(-1,dtype=torch.int32).contiguous()
    qi,qs=quantize(q,16); ki,ks=quantize(k,64); aki,aks=quantize(ak,64)
    output=torch.empty_like(q)
    with torch.cuda.device(q.device):
        stream=torch.cuda.current_stream(q.device).cuda_stream
        check(load().star7_sol_sm75_pack_ordered_lut(mask.data_ptr(),lut.data_ptr(),
            b*h*leaves,leaves,leaves,stream),'pack routes')
        check(load().star7_spark_sm75_attention(*[x.data_ptr() for x in
            (qi,ki,v,qs,ks,rows,lut,aki,av,aks,mask,lm,output)],
            b,h,n,leaves,leaves,padded,d**-.5,stream),'attention')
    return output


def quantize_value(value):
    b,h,n,d=value.shape; padded=(n+63)//64*64
    output=torch.empty((b,h,d,padded),device=value.device,dtype=torch.int8)
    scale=torch.empty((b,h,d),device=value.device,dtype=torch.float32)
    with torch.cuda.device(value.device):
        check(load().star7_sla_sm75_quant_v_int8(
            value.data_ptr(),output.data_ptr(),scale.data_ptr(),b,h,n,d,padded,
            *value.stride()[:3],1,torch.cuda.current_stream(value.device).cuda_stream),'quantize value')
    return output,scale


def consume_fast(q,k,v,mask,ak,av,lm,*,query_tokens=None):
    """Spark-only INT8 PV; weighted summaries use FP16, suffix uses native CK."""
    validate(q,k,v)
    b,h,n,d=q.shape;leaves=(n+63)//64;padded=ak.shape[2]
    if mask.shape!=(b,h,leaves,leaves) or mask.device!=q.device or mask.dtype!=torch.uint8 or not mask.is_contiguous():
        raise ValueError('Invalid exact-block mask')
    for x in (ak,av):
        if x.shape!=(b,h,padded,128) or padded<leaves or padded%64 or x.dtype!=q.dtype or x.device!=q.device or not x.is_contiguous():
            raise ValueError('Invalid weighted summary tensor')
    if lm.shape!=(b,h,padded) or lm.dtype!=torch.float32 or lm.device!=q.device or not lm.is_contiguous():
        raise ValueError('Invalid weighted log mass')
    query_tokens=n if query_tokens is None else query_tokens
    if not isinstance(query_tokens,int) or not 1<=query_tokens<=n:raise ValueError('Invalid Spark query prefix')
    query_blocks=(query_tokens+63)//64; split=min(n,query_blocks*64)
    lut=torch.empty(mask.shape,device=q.device,dtype=torch.int32)
    rows=mask.sum(-1,dtype=torch.int32).contiguous()
    qi,qs=quantize(q,16);ki,ks=quantize(k,64);aki,aks=quantize(ak,64)
    vi,vs=quantize_value(v);avi,avs=vi,vs  # summaries retain FP16 PV
    output=torch.empty_like(q)
    with torch.cuda.device(q.device):
        stream=torch.cuda.current_stream(q.device).cuda_stream
        check(load().star7_sol_sm75_pack_ordered_lut(mask.data_ptr(),lut.data_ptr(),
            b*h*leaves,leaves,leaves,stream),'pack routes')
        check(load().star7_spark_sm75_attention_fast(*[x.data_ptr() for x in
            (qi,ki,v,qs,ks,rows,lut,aki,av,aks,mask,lm,output,vi,avi,vs,avs)],
            b,h,n,leaves,leaves,padded,vi.shape[-1],query_blocks,d**-.5,stream),'fast attention')
        if split<n:
            import comfy_kitchen
            output[:,:,split:]=comfy_kitchen.int8_attention(q[:,:,split:].contiguous(),k,v)
    return output
