#!/usr/bin/env bash
# Linux source build path; the Windows build does not constitute Linux validation.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
nvcc="${CUDA_ROOT:+${CUDA_ROOT}/bin/}nvcc"
output_dir="${project_root}/bin/linux_x86_64"
mkdir -p "${output_dir}"
"${nvcc}" -shared -O3 --use_fast_math -std=c++17 \
    --generate-code=arch=compute_75,code=sm_75 --cudart static \
    -Xcompiler=-fPIC -Xcompiler=-O2 -Xlinker=--exclude-libs,ALL \
    -o "${output_dir}/star7_spark_sm75_v3.so" "${script_dir}/spark_sm75.cu"
"${PYTHON:-python3}" - "${project_root}" <<'PY'
import ctypes, hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1]); binary=root/'bin/linux_x86_64/star7_spark_sm75_v3.so'
library=ctypes.CDLL(str(binary))
assert library.star7_spark_sm75_abi_version()==1, 'Spark ABI mismatch'
path=root/'bin/spark_sm75_manifest.json'
manifest=json.loads(path.read_text()) if path.exists() else dict(abi=1,architecture='sm75',platforms={})
manifest['platforms']['linux_x86_64']=dict(file='linux_x86_64/'+binary.name,sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('Built and ABI-checked',binary)
PY
