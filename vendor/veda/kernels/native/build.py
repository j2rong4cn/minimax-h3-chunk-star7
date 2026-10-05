"""Build with ComfyUI's Python/PyTorch runtime and a CUDA compiler toolchain."""
import os
import pathlib
import shutil
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.5')
os.environ.setdefault('MAX_JOBS', '2')
from torch.utils.cpp_extension import load
root = pathlib.Path(__file__).resolve().parent
build = root / '_build'
build.mkdir(exist_ok=True)
windows = os.name == 'nt'
module = load(name='_star7_veda_sm75',
    sources=[str(root / 'bindings.cpp'), str(root / 'sage/veda_sparse.cu')],
    build_directory=str(build),
    extra_cflags=['/O2', '/std:c++20'] if windows else ['-O3', '-std=c++20'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++20'] +
        (['-Xcompiler', '/Zc:preprocessor', '-Xcompiler', '/DNOMINMAX'] if windows else []),
    verbose=True)
shutil.copy2(module.__file__, root / ('_star7_veda_sm75.pyd' if windows else '_star7_veda_sm75.so'))
