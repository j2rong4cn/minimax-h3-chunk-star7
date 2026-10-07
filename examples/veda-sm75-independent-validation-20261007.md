# VEDA SM75 independent CUDA library validation

Local environment: Windows x64, RTX 2080 Ti 22 GB (SM75), NVIDIA driver 616.56, Python 3.12.9, PyTorch 2.13.0+cu130. Library built with CUDA 13.0, static CUDA and C++ runtimes.

| Check | Result |
| --- | --- |
| DLL PE imports | Only KERNEL32.dll; no Python, Torch, CUDA runtime or VC++ runtime DLL imports |
| Manifest checksum and C ABI | Passed; ABI 1, argument structure 192 bytes |
| Standalone ctypes load | Passed without importing Torch |
| Python 3.14.2, no Torch installed | DLL load and ABI inspection passed |
| Sparse CUDA numerical self-test | Passed on RTX 2080 Ti |
| Nondefault CUDA stream | Numerical self-test passed |
| Old extension / new DLL parity | One identical-input case, bitwise identical; maximum error 0 |
| Normal H3 / CK forward dispatch | Both reached VEDA override in integration tests |
| VEDA disabled | Predictor and kernel loading bypassed |
| SM80+ dispatch | Existing Triton selection preserved |

Reproduce:

```powershell
python -B test_veda_sm75_abi.py
python -B -c "import sys,runpy;sys.path.insert(0,'.');sys.argv=['test','--cpu'];runpy.run_path('test_veda_preview_integration.py')"
```

These are kernel and integration checks, not a full video benchmark. RTX 2060, other PyTorch installations and Linux GPU execution were not tested. Python 3.14 validation covers standalone library loading, not full Torch inference. A CUDA-13-compatible NVIDIA driver remains necessary.

The old `_star7_veda_sm75.pyd` is removed from distribution and is not loaded by the new backend.
