"""Developer build only; users load the bundled CUDA C ABI library."""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

root = pathlib.Path(__file__).resolve().parent
project = root.parent
source = root / 'h3_preprocess.cu'
if os.name == 'nt':
    output = project / 'bin/win_amd64/star7_h3_preprocess_v1.dll'
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('nvcc.exe') or shutil.which('nvcc')
    if not compiler:
        candidates = list((pathlib.Path(os.environ.get('ProgramFiles', 'C:\\Program Files')) / 'NVIDIA GPU Computing Toolkit/CUDA').glob('v*/bin/nvcc.exe'))
        if not candidates:
            raise RuntimeError('CUDA developer compiler nvcc is required to build')
        compiler = str(max(candidates, key=lambda p: tuple(map(int, p.parent.parent.name[1:].split('.')))))
    vswhere = pathlib.Path(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    studio = subprocess.check_output([str(vswhere), '-latest', '-version', '[16.0,18.0)', '-products', '*',
                                     '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                                     '-property', 'installationPath'], text=True).strip()
    if not studio:
        raise RuntimeError('Visual Studio 2019/2022 C++ developer tools are required to build')
    vcvars = pathlib.Path(studio) / 'VC/Auxiliary/Build/vcvars64.bat'
    command = f'call "{vcvars}" && "{compiler}" -shared -O3 --use_fast_math -std=c++17 -arch=sm_75 --cudart static -Xcompiler=/MT -Xcompiler=/O2 -Xptxas=-v -o "{output}" "{source}"'
    subprocess.run('cmd.exe /d /s /c ' + command, check=True)
    for suffix in ('.lib', '.exp'):
        output.with_suffix(suffix).unlink(missing_ok=True)
    platform = 'windows_x64'
else:
    output = project / 'bin/linux_x86_64/star7_h3_preprocess_v1.so'
    output.parent.mkdir(parents=True, exist_ok=True)
    nvcc = shutil.which('nvcc')
    if not nvcc:
        raise RuntimeError('CUDA developer compiler nvcc is required to build')
    subprocess.run([nvcc, '-shared', '-O3', '--use_fast_math', '-std=c++17', '-arch=sm_75',
                    '--cudart', 'static', '-Xcompiler=-fPIC,-static-libstdc++,-static-libgcc',
                    '-Xlinker=--exclude-libs,ALL', '-o', str(output), str(source)], check=True)
    platform = 'linux_x86_64'
manifest_path = project / 'bin/h3_preprocess_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig')) if manifest_path.exists() else {'abi_version': 1}
manifest[platform] = {'file': output.relative_to(project / 'bin').as_posix(),
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'size': output.stat().st_size}
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print('Built independent CUDA C ABI library:', output)
