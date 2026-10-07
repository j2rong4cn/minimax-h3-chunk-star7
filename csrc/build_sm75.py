"""Developer build only; users load the bundled CUDA C ABI library."""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

root = pathlib.Path(__file__).resolve().parent
project = root.parent
source = root / 'sla_sm75_sparse.cu'


def main():
    parser = argparse.ArgumentParser(description='Build Star7 SM75 SLA CUDA C ABI library')
    parser.add_argument('--cuda-root', '-CudaRoot', dest='cuda_root', default=None,
                        help='Path to CUDA Toolkit root directory')
    parser.add_argument('--visual-studio-root', '-VisualStudioRoot', dest='visual_studio_root', default=None,
                        help='Path to Visual Studio root directory (Windows only)')
    parser.add_argument('--output-name', '-OutputName', dest='output_name', default=None,
                        help='Output library file name')
    parser.add_argument('--max-registers', '-MaxRegisters', dest='max_registers', type=int, default=0,
                        help='Maximum register count (--maxrregcount)')
    args = parser.parse_args()

    if os.name == 'nt':
        default_output = 'star7_sla_sm75_v7.dll'
        output_name = args.output_name or default_output
        output = project / 'bin/win_amd64' / output_name
        output.parent.mkdir(parents=True, exist_ok=True)

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

        studio = args.visual_studio_root
        if not studio:
            vswhere = pathlib.Path(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
            studio = subprocess.check_output([str(vswhere), '-latest', '-version', '[16.0,18.0)', '-products', '*',
                                             '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                                             '-property', 'installationPath'], text=True).strip()
            if not studio:
                raise RuntimeError('Visual Studio 2019/2022 C++ developer tools are required to build')
        vcvars = pathlib.Path(studio) / 'VC/Auxiliary/Build/vcvars64.bat'

        register_flag = f'--maxrregcount={args.max_registers}' if args.max_registers > 0 else ''
        command = (
            f'call "{vcvars}" && "{compiler}" -shared -O3 --use_fast_math -std=c++17 -arch=sm_75 --cudart static '
            f'-Xcompiler=/MT -Xcompiler=/O2 -Xptxas=-v {register_flag} -o "{output}" "{source}"'
        )
        subprocess.run('cmd.exe /d /s /c ' + command, check=True)
        for suffix in ('.lib', '.exp'):
            output.with_suffix(suffix).unlink(missing_ok=True)
        platform_key = 'windows_x64'
    else:
        default_output = 'star7_sla_sm75_v7.so'
        output_name = args.output_name or default_output
        output = project / 'bin/linux_x86_64' / output_name
        output.parent.mkdir(parents=True, exist_ok=True)

        cuda_root_dir = args.cuda_root or os.environ.get('CUDA_ROOT') or os.environ.get('CUDA_PATH') or '/usr/local/cuda'
        cuda_root = pathlib.Path(cuda_root_dir)
        compiler = cuda_root / 'bin' / 'nvcc'
        if not compiler.is_file():
            which_nvcc = shutil.which('nvcc')
            if which_nvcc:
                compiler = pathlib.Path(which_nvcc)
            else:
                raise RuntimeError(f'CUDA developer compiler nvcc not found at {compiler} or in PATH')

        cmd = [
            str(compiler),
            '-shared',
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            '-arch=sm_75',
            '--cudart', 'static',
            '-Xcompiler=-fPIC,-static-libstdc++,-static-libgcc',
            '-Xlinker=--exclude-libs,ALL',
        ]
        if args.max_registers > 0:
            cmd.append(f'--maxrregcount={args.max_registers}')
        cmd.extend(['-o', str(output), str(source)])
        subprocess.run(cmd, check=True)
        platform_key = 'linux_x86_64'

    manifest_path = project / 'bin/sm75_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig')) if manifest_path.exists() else {'abi_version': 7}
    if platform_key not in manifest or not isinstance(manifest[platform_key], dict):
        manifest[platform_key] = {}
    manifest[platform_key]['file'] = output.relative_to(project / 'bin').as_posix()
    manifest[platform_key]['sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest[platform_key]['size'] = output.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Built independent CUDA C ABI library:', output)


if __name__ == '__main__':
    main()
