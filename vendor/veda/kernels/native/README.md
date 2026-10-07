# Star7 VEDA SM75 CUDA library

Windows x64 uses `bin/win_amd64/star7_veda_sm75_v1.dll` through a versioned C ABI and `ctypes`. It accepts CUDA tensor addresses, strides and the caller's CUDA stream. The sparse INT8 Tensor Core kernel and quantization are unchanged. There is no CPython or PyTorch C++ ABI dependency and no dependency on another VEDA custom node.

The distributed library embeds the CUDA 13 runtime and Microsoft C++ runtime statically. Its direct PE import is `KERNEL32.dll`. Users need a CUDA-13-compatible NVIDIA driver and working CUDA-enabled PyTorch for ComfyUI; they do not need a CUDA Toolkit, compiler, Python headers or import libraries. Independence from Python/Torch binary versions does not remove the NVIDIA driver requirement.

Update the DLL, `bin/veda_sm75_manifest.json` and Python loader together, then restart ComfyUI. The loader checks SHA-256 and ABI version. An unavailable library or failed numerical startup test disables VEDA and reports the reason; full attention then runs without sparse acceleration. The old `.pyd` is never loaded by this backend; a running process may keep that obsolete file locked until restart.

SM75 selects this library. SM80+ continues to use upstream Triton. Disabling VEDA bypasses predictor and kernel loading. Existing normal and CK attention paths are preserved.

Validation: RTX 2080 Ti sparse numerical self-test, nondefault CUDA stream, normal/CK dispatch, and one bitwise comparison against the previous extension. Standalone DLL loading also passed on Python 3.14 without Torch installed. RTX 2060 and other Torch installations still require device-side testing. No Linux binary is bundled or validated.

Developer build: `python vendor/veda/kernels/native/build.py`. Windows requires CUDA developer tools and Visual Studio 2019/2022 C++ tools; the script builds without importing Torch. Linux source build support is unvalidated.

Sources and distribution notices: [Apache 2.0](LICENSE.Apache-2.0) and [VEDA notices](../../NOTICE.md).
