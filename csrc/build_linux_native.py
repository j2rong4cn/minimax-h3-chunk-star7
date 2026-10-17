"""Build reviewed SM75 Linux libraries; no compiler is needed by end users."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def build(component, nvcc, cutlass_include=None):
    native = ROOT / "csrc/w4a8_native"
    specs = {
        "preprocess": (ROOT / "csrc/h3_preprocess.cu", "star7_h3_preprocess_v1.so", "h3_preprocess_manifest.json"),
        "veda": (ROOT / "vendor/veda/kernels/native/veda_sm75_abi.cu", "star7_veda_sm75_v1.so", "veda_sm75_manifest.json"),
        "w4a8": (native / "abi.cu", "star7_w4a8_sm75_v1.so", "w4a8_manifest.json"),
    }
    source, filename, manifest_name = specs[component]
    output = ROOT / "bin/linux_x86_64" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [nvcc, "-shared", "-O3", "-std=c++20" if component == "w4a8" else "-std=c++17",
               "-arch=sm_75", "--cudart", "static",
               "-Xcompiler=-fPIC,-fvisibility=hidden,-static-libstdc++,-static-libgcc",
               "-Xlinker=--exclude-libs,ALL"]
    if component == "w4a8":
        if not cutlass_include or not (cutlass_include / "cutlass/cutlass.h").is_file():
            raise RuntimeError("Pass NVIDIA CUTLASS 4.2.0 headers via --cutlass-include")
        for include in (native, native / "cutlass_overlay/include", cutlass_include):
            command += ["-I", str(include)]
        cccl = Path(nvcc).resolve().parent.parent / "include/cccl"
        if cccl.is_dir():
            command += ["-I", str(cccl)]
    # Preserve the FP32 SwiGLU /256 contract. Do not substitute the fork's /128 kernel.
    subprocess.run(command + ["-o", str(output), str(source)], check=True)
    manifest_path = ROOT / "bin" / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if "file" in manifest:
        manifest["windows_x64"] = {key: manifest.pop(key) for key in ("file", "sha256")}
    manifest["linux_x86_64"] = {
        "file": "linux_x86_64/" + filename,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "size": output.stat().st_size,
        "build_commit": os.environ.get("GITHUB_SHA", "local"),
        "validation": "Linux build and host ABI checks; GPU numerical validation pending",
    }
    if component == "preprocess":
        manifest["swiglu_divisor"] = 256
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Built", output.relative_to(ROOT), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("preprocess", "veda", "w4a8", "all"), default="all")
    parser.add_argument("--cutlass-include", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise SystemExit("Run this developer build on Linux, not the active Windows installation")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise SystemExit("CUDA developer nvcc was not found")
    for component in ("preprocess", "veda", "w4a8") if args.component == "all" else (args.component,):
        build(component, nvcc, args.cutlass_include)


if __name__ == "__main__":
    main()
