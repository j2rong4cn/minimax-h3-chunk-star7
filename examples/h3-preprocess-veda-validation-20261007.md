# H3 preprocessing and VEDA validation

Windows x64, RTX 2080 Ti 22 GB, Python 3.12.9, PyTorch 2.13.0+cu130. Version: 2.18.2.

| Check | Result |
| --- | --- |
| SM75 / SM80+ VEDA dispatch, disabled bypass, ordinary / CK integration | Passed |
| Predictor cache reload on file change | Passed |
| Workspace limits and explicit head grouping | Passed; grouped numerical outputs agree |
| Dense fallback with both output layouts | Passed |
| Layout reuse across runs and statistics reset | Passed; at most eight cached layouts |
| Streaming Q/K, global K anchor and CK output | Bitwise parity on tested random tensors |
| Production projection loop and full CK forward | Passed |
| RMSNorm / RoPE and nondefault CUDA stream | Passed |
| CPU / GPU chunking regression suite | Passed |
| Returned mixed W4A8 model | Header: 180 W4A8 + 20 INT8 layers; group-8 fc2 native execution passed |

The actual model's fc2 comparison measured relative L2 error 0.0000778 against the previous staged SwiGLU path. Fused preprocessing is not bitwise identical: the tested half-precision SwiGLU maximum difference was 0.000244141.

Isolated GPU-event measurement, 2048 rows and a 28672-wide SwiGLU input, ten measured iterations after warmup:

| Preprocessing | Time | Additional peak allocation |
| --- | ---: | ---: |
| Previous FP32 tensor operations | 2.647 ms | 336 MiB |
| Fused SwiGLU / rescale / FP16 output | 0.430 ms | 56 MiB |

Synthetic QKV-only measurement (16384 rows, 8 heads, head dimension 128, 8192-row chunks): previous 4.28 ms / 257 MiB peak versus compact 5.36 ms / 224.96 MiB peak. This phase includes the added quantization work, not the complete attention call. Compression is enabled only under memory pressure; ample-memory calls keep their previous path.

Compact Q/K applies to standalone SM75 FP16 CK with aligned projection chunks, a supported native library, and a passing installed-Kitchen quantization check. Its stored Q/K/V buffers use INT8 Q/K and FP16 V instead of three FP16 buffers. VEDA and other attention overrides retain floating Q/K features; they benefit from VEDA's workspace limits/caches and compatible W4A8 SwiGLU fusion, not from this CK-only compression path.

No full video benchmark was run by the agent. Existing node sockets and workflow connections are unchanged. Restart ComfyUI before comparing identical model, seed, steps, resolution and duration. Preprocessing, VEDA and W4A8 GEMM now use independent CUDA DLLs and require a CUDA-13-compatible NVIDIA driver.

## Portability and logging audit

- VEDA and preprocessing DLL PE imports: only `KERNEL32.dll`. Both loaded successfully via ctypes on Python 3.14.2 without importing Torch; full inference on that Python was not tested.
- W4A8 GEMM also uses an independent CUDA DLL. Its former `.pyd` imported `python312.dll`, `c10` / Torch DLLs and Microsoft C++ runtime DLLs; the new runtime does not load it.
- Compact Q/K checks the installed Kitchen's actual quantization contract rather than pinning a package version. CUDA 13 driver compatibility remains necessary for the independent DLLs.
- Owned log prefixes: `[Star7 H3 VEDA]`, `[Star7 H3 Preprocess]`, `[Star7 H3 W4A8]`. Multiline VEDA status tags every line, suppresses consecutive duplicate messages, and preserves system diagnostic text. No progress text is sent to node UI and no global log handlers are installed.
- Logging, unavailable-library fallback, VEDA dispatch and cache lifecycle regressions passed.

## W4A8 independent GEMM validation

- PE imports: only `KERNEL32.dll`; C ABI version 1, argument structure 104 bytes. Python 3.14.2 standalone loading passed without importing Torch. Full inference under a different Python/Torch installation was not tested.
- FP16 row quantization was bitwise equal to the previous extension on three shapes, including the real H3 contracted width 14336.
- Twenty old/new GEMM cases were bitwise equal: groups 4/8/16, bias/no bias, staged/inline decode, multiple output chunks and a nondefault CUDA stream. The default regression test additionally checks against a decoded-weight PyTorch reference without needing the retired extension.
- The returned mixed W4A8 model's actual group-8 fc2 ran through the new DLL and passed the preprocessing regression.
- Isolated ten-iteration GPU-event measurements: group-8 `(M,N,K)=(256,5376,14336)` old/new 1.169/1.208 ms; group-16 inline `(8193,5376,256)` old/new 1.136/0.753 ms. These are short microbenchmarks, not end-to-end speed claims.
- No fixed Python or PyTorch version checks, Python headers, Torch C++ symbols or local compilation are required by the distributed kernel. Checkpoint-format support and public CUDA PyTorch APIs remain necessary.
