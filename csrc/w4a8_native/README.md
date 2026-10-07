# Star7 W4A8 SM75 C ABI

The Windows x64 DLL replaces the former CPython/PyTorch extension. Tensor addresses, dimensions, workspace and the current CUDA stream cross a versioned C ABI; Python owns every allocation through public PyTorch APIs. There are no Python headers, ATen symbols or PyTorch DLL imports in this library. Static CUDA 13 and C++ runtimes leave only KERNEL32.dll as a direct PE import. A compatible NVIDIA driver and working CUDA PyTorch installation remain required.

The existing grouped-codebook GEMM source is preserved. FP16 output, bias conversion, group scales, 4096-channel staged workspaces and the group-16/M>8192 inline rule match the previous bridge. SM80+ and unsupported weight layouts retain upstream execution. No Linux binary is bundled or validated.

Update `bin/w4a8_manifest.json`, `bin/win_amd64/star7_w4a8_sm75_v1.dll` and the Python bridge together, then restart ComfyUI. The runtime does not import the old `.pyd`.

Developer build only:

```powershell
./csrc/build_w4a8_windows.ps1 -CutlassInclude <CUTLASS-4.2.0-include-directory>
```

Requires CUDA 13 developer tools, Visual Studio 2019/2022 C++ tools and NVIDIA CUTLASS 4.2.0 headers. Compilation never imports Torch and does not require Python development files. The bundled Windows header overlays preserve the tested CUTLASS/MSVC compatibility fixes. End users load the precompiled DLL and do not need these tools.

Sources derive from the previously shipped Star7 adaptation of ComfyUI Turing Utils / Comfy Kitchen (Apache-2.0) and NVIDIA CUTLASS (BSD-3-Clause). Existing notices and licenses remain in `bin/win_amd64/w4a8/NOTICE`, `LICENSE` and `LICENSES/cutlass-LICENSE.txt`.
