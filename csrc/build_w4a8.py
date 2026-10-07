"""Developer build only; users load the bundled CUDA C ABI library."""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent
project = root.parent
native = root / 'w4a8_native'
overlay = native / 'cutlass_overlay' / 'include'
source = native / 'abi.cu'


def find_cutlass_include(explicit=None):
    if explicit:
        p = pathlib.Path(explicit).resolve()
        if (p / 'cutlass' / 'cutlass.h').is_file():
            return p
        raise FileNotFoundError(f'CUTLASS header cutlass/cutlass.h not found in {p}')

    env_val = os.environ.get('CUTLASS_INCLUDE') or os.environ.get('CUTLASS_DIR')
    if env_val:
        p = pathlib.Path(env_val).resolve()
        if (p / 'cutlass' / 'cutlass.h').is_file():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description='Build Star7 W4A8 CUDA C ABI library')
    parser.add_argument('--cutlass-include', '-CutlassInclude', dest='cutlass_include', default=None,
                        help='Path to NVIDIA CUTLASS include directory (containing cutlass/cutlass.h)')
    parser.add_argument('--cuda-root', '-CudaRoot', dest='cuda_root', default=None,
                        help='Path to CUDA Toolkit root directory')
    args = parser.parse_args()

    cutlass_include = find_cutlass_include(args.cutlass_include)
    if not cutlass_include:
        raise RuntimeError(
            'NVIDIA CUTLASS 4.2.0 include directory is required. '
            'Pass --cutlass-include <path> or set the CUTLASS_INCLUDE environment variable.'
        )

    if os.name == 'nt':
        cuda_root_dir = args.cuda_root or os.environ.get('CUDA_ROOT') or os.environ.get('CUDA_PATH')
        if cuda_root_dir:
            cuda_root = pathlib.Path(cuda_root_dir)
            compiler = str(cuda_root / 'bin' / 'nvcc.exe')
        else:
            compiler = shutil.which('nvcc.exe') or shutil.which('nvcc')
            if not compiler:
                candidates = list((pathlib.Path(os.environ.get('ProgramFiles', 'C:\\Program Files')) / 'NVIDIA GPU Computing Toolkit/CUDA').glob('v*/bin/nvcc.exe'))
                if not candidates:
                    raise RuntimeError('CUDA developer compiler nvcc is required to build')
                compiler = str(max(candidates, key=lambda p: tuple(map(int, p.parent.parent.name[1:].split('.')))))
            cuda_root = pathlib.Path(compiler).resolve().parent.parent

        output = project / 'bin/win_amd64/star7_w4a8_sm75_v1.dll'
        output.parent.mkdir(parents=True, exist_ok=True)

        vswhere = pathlib.Path(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
        studio = subprocess.check_output([str(vswhere), '-latest', '-version', '[16.0,18.0)', '-products', '*',
                                         '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                                         '-property', 'installationPath'], text=True).strip()
        if not studio:
            raise RuntimeError('Visual Studio 2019/2022 C++ developer tools are required to build')
        vcvars = pathlib.Path(studio) / 'VC/Auxiliary/Build/vcvars64.bat'

        cccl = cuda_root / 'include' / 'cccl'
        include_args = f'-I"{native}" -I"{overlay}" -I"{cutlass_include}"'
        if cccl.exists():
            include_args += f' -I"{cccl}"'

        command = (
            f'call "{vcvars}" && "{compiler}" -shared -O3 -std=c++20 -arch=sm_75 --cudart static '
            f'-Xcompiler=/MT -Xcompiler=/O2 -Xcompiler=/EHsc -Xcompiler=/bigobj '
            f'-Xcompiler=/Zc:preprocessor -Xcompiler=/permissive '
            f'{include_args} -o "{output}" "{source}"'
        )
        subprocess.run('cmd.exe /d /s /c ' + command, check=True)
        for suffix in ('.lib', '.exp'):
            output.with_suffix(suffix).unlink(missing_ok=True)
        platform = 'windows_x64'
    else:
        cuda_root_dir = args.cuda_root or os.environ.get('CUDA_ROOT') or os.environ.get('CUDA_PATH') or '/usr/local/cuda'
        cuda_root = pathlib.Path(cuda_root_dir)
        compiler = cuda_root / 'bin' / 'nvcc'
        if not compiler.is_file():
            which_nvcc = shutil.which('nvcc')
            if which_nvcc:
                compiler = pathlib.Path(which_nvcc)
            else:
                raise RuntimeError(f'CUDA developer compiler nvcc not found at {compiler} or in PATH')

        output = project / 'bin/linux_x86_64/star7_w4a8_sm75_v1.so'
        output.parent.mkdir(parents=True, exist_ok=True)

        cccl = cuda_root / 'include' / 'cccl'
        includes = ['-I', str(native), '-I', str(overlay), '-I', str(cutlass_include)]
        if cccl.exists():
            includes.extend(['-I', str(cccl)])

        cmd = [
            str(compiler),
            '-shared',
            '-O3',
            '-std=c++20',
            '-arch=sm_75',
            '--cudart', 'static',
            '-Xcompiler=-fPIC,-static-libstdc++,-static-libgcc',
            '-Xlinker=--exclude-libs,ALL',
            *includes,
            '-o', str(output),
            str(source),
        ]
        subprocess.run(cmd, check=True)
        platform = 'linux_x86_64'

    manifest_path = project / 'bin/w4a8_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig')) if manifest_path.exists() else {'abi_version': 1}
    manifest['abi_version'] = 1
    manifest[platform] = {
        'file': output.relative_to(project / 'bin').as_posix(),
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'size': output.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Built independent CUDA C ABI library:', output)


if __name__ == '__main__':
    main()
