# Star7 VEDA SM75 kernel

The bundled Windows x64 `_star7_veda_sm75.pyd` was built and tested with Python 3.12.9, PyTorch 2.13.0+cu130 and CUDA 13 on an RTX 2080 Ti. It is a PyTorch extension, not an ABI-independent CUDA DLL. Other Python/PyTorch builds may require recompilation. No Linux VEDA SM75 binary is included.

To rebuild, use the same Python/PyTorch environment as ComfyUI, matching CUDA development tools, Ninja and a C++ compiler. On Windows run from an x64 Visual Studio developer terminal; portable Python must have matching Python headers and import libraries. Run `python vendor/veda/kernels/native/build.py` from the project root. The script copies the resulting binary beside the source. Recompilation on other environments has not been validated for this release.

SM80+ uses the upstream Triton implementation and does not load this extension. Turning VEDA off does not load either sparse kernel.

Sources and distribution notices: [Apache 2.0](LICENSE.Apache-2.0) and [VEDA notices](../../NOTICE.md).
