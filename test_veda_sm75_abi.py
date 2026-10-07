"""Verify shipped Windows DLL dependencies, standalone loading and CUDA stream interop."""
import ctypes
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent
entry = json.loads((ROOT / 'bin/veda_sm75_manifest.json').read_text())['windows_x64']
binary = ROOT / 'bin' / entry['file']
data = binary.read_bytes()
assert hashlib.sha256(data).hexdigest() == entry['sha256']

# Inspect the PE import table without requiring Microsoft's developer tools.
pe = struct.unpack_from('<I', data, 0x3c)[0]
assert data[pe:pe + 4] == b'PE\0\0'
sections = struct.unpack_from('<H', data, pe + 6)[0]
optional_size = struct.unpack_from('<H', data, pe + 20)[0]
optional = pe + 24
assert struct.unpack_from('<H', data, optional)[0] == 0x20b
section_table = optional + optional_size

def offset(rva):
    for i in range(sections):
        size, address, raw_size, raw = struct.unpack_from('<IIII', data, section_table + i * 40 + 8)
        if address <= rva < address + max(size, raw_size):
            return raw + rva - address
    raise AssertionError(f'Unmapped RVA: {rva}')

descriptor = offset(struct.unpack_from('<I', data, optional + 120)[0])
imports = []
while any(data[descriptor:descriptor + 20]):
    name = offset(struct.unpack_from('<I', data, descriptor + 12)[0])
    imports.append(data[name:data.index(b'\0', name)].decode().lower())
    descriptor += 20
assert imports == ['kernel32.dll'], imports

# No torch import in the child: standalone loading must not need Torch DLLs.
script = '''import ctypes,sys
lib = ctypes.CDLL(sys.argv[1])
assert lib.star7_veda_abi_version() == 1
assert lib.star7_veda_args_size() == 192
assert 'torch' not in sys.modules
print('Standalone C ABI load: PASS')
'''
subprocess.run([sys.executable, '-B', '-c', script, str(binary)], check=True)
print('PE imports / checksum: PASS')

import torch
pkg = types.ModuleType('star7_abi_test')
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg
from star7_abi_test.vendor.veda.backends import base
from star7_abi_test.vendor.veda.backends.torch_gather import TorchGatherBackend
from star7_abi_test.vendor.veda.kernels.native.runtime import Args
assert ctypes.sizeof(Args) == 192
backend = TorchGatherBackend()
base.self_test(backend, torch.device('cuda'), torch.float16)
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    base.self_test(backend, torch.device('cuda'), torch.float16)
stream.synchronize()
print('CUDA numerical self-test / nondefault stream: PASS')
