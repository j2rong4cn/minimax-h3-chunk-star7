"""C ABI numerical checks; optional --legacy PATH verifies the retired extension."""
import importlib.util
from pathlib import Path
import sys
import torch
from w4a8_native import Kernel

ROOT = Path(__file__).resolve().parent
new = Kernel()
old = None
if '--legacy' in sys.argv:
    spec = importlib.util.spec_from_file_location('star7_h3_native._C',sys.argv[sys.argv.index('--legacy')+1])
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
torch.manual_seed(11)
for rows,width in ((3,256),(256,14336),(8193,256)):
    value=torch.randn(rows,width,device='cuda',dtype=torch.float16)
    actual=new.turing_fp16_int8_quantize(value)
    if old is not None:
        expected=old.turing_fp16_int8_quantize(value)
    else:
        scale=(value.float().abs().amax(dim=1,keepdim=True)*(1./127)).clamp_min(1e-30)
        rounded=scale.half(); rounded=torch.where(rounded==0,torch.full_like(rounded,2**-14),rounded)
        expected=((value/rounded).round().clamp(-128,127).to(torch.int8),scale)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
print('FP16 row quantization: bitwise PASS')

stream=torch.cuda.Stream()
cases=0
for m,n,k,g in ((7,128,256,4),(65,512,512,8),(256,512,1024,16),(8201,512,256,16),(2,4104,256,8)):
    with torch.cuda.stream(stream):
        activation=torch.randint(-127,128,(m,k),device='cuda',dtype=torch.int8)
        weight=torch.randint(-128,128,(n,k//2),device='cuda',dtype=torch.int8)
        act_scale=torch.rand(m,1,device='cuda')*.01
        group_scale=(torch.rand(n,k//g,device='cuda')*20).to(torch.float8_e4m3fn).view(torch.uint8)
        channel_scale=torch.rand(n,device='cuda')*.01
        codebook=torch.linspace(-1,1,16,device='cuda')
        for bias in (None,torch.randn(n,device='cuda',dtype=torch.float16)):
            for chunk in (0,128):
                args=(activation,weight,act_scale,group_scale,channel_scale,codebook,bias,g,chunk)
                actual=new.turing_fp16_codebook_w4a8_linear(*args)
                if old is not None:
                    expected=old.turing_fp16_codebook_w4a8_linear(*args)
                    assert torch.equal(actual,expected),(m,n,k,g,bias is None,chunk,(actual-expected).abs().max().item())
                else:
                    raw=weight.to(torch.int32)
                    codes=torch.stack((raw & 15,(raw >> 4) & 15),dim=-1).reshape(n,k)
                    scales=group_scale.view(torch.float8_e4m3fn).float().repeat_interleave(g,dim=1)
                    decoded=(codebook[codes]*scales).round().clamp(-127,127)
                    expected=(activation.float() @ decoded.t())*act_scale*channel_scale[None,:]
                    if bias is not None:expected+=bias.float()
                    torch.testing.assert_close(actual,expected.half(),rtol=.001,atol=.002)
                cases+=1
    stream.synchronize()
print(f'GEMM grouped scales / bias / staged+inline / nondefault stream: {cases} cases PASS')
