# MiniMax H3 Activation Chunk & Attention Acceleration - Star7

[中文](README.md) · [Benchmarks](BENCHMARKS.md) · [Example workflows](examples/workflows)

The 1006 bilingual examples include: first pass -> optional HD -> optional Face Repair -> Star7 Chunked Decode, with FPS-controlled live previews and VEDA bypassed by default. Select installed models and replace reference-media placeholders.

Run high-quality, long-duration MiniMax H3 video generation on GPUs with limited VRAM. The core node combines independent QKV, RoPE, and MLP activation chunking with a selectable attention backend. It does not alter the sampler, sigma schedule, latent layout, VAE, duration, frame count, or output resolution.

## Main features

| Feature | Description |
|---|---|
| Independent QKV / RoPE / MLP chunking | Controls the temporary VRAM peak of each stage while preserving the original H3 block, weights, LoRA, and conditioning layout |
| Targeted OOM reduction | Halves only the chunk size of the stage that actually ran out of memory |
| Attention selection | Supports `existing`, Comfy Kitchen INT8, SLA, Sol, and CK/Sparse/CK Hybrid paths |
| H3 W4A8 model support | Enhanced Loader preserves the `asym_w4a8_int8` layout; SM75 adds native codebook W4A8 linear acceleration, compatible with chunked CK and VEDA |
| Architecture-specific sparse kernels | Bundled native CUDA kernels for SM75; Triton or the official NVIDIA Sol-Attn path for SM80+ |
| Numerical diagnostics | Detects NaN/Inf and can identify the first failing QKV, attention, `out_proj`, or MLP stage |
| Lightweight utilities | Reference-image, reference-video, prompt-loading, and workflow-export helpers |

Chunking reduces temporary activation residency; it does not remove tokens or reduce theoretical FLOPs. It is most useful when an unchunked workload would OOM, spill into shared memory, or page frequently.

## Attention backends

The dropdown intentionally uses stable backend IDs so workflows remain portable across languages and machines.

### General

| ID | Purpose |
|---|---|
| `existing` | Keep the incoming model's current attention implementation, including an upstream Sage or other patch |
| `comfy_kitchen_int8` | Use ComfyUI / Comfy Kitchen INT8 attention |

When an enhanced upstream loader has already installed VSA, select `existing` to add only QKV, RoPE, and MLP chunking. Alternatively, select the direct `vsa_sm75` or `vsa_sm80+` path: SM75 uses Star7's precompiled CUDA producer, while SM80+ uses Comfy Kitchen Sol-Attn. Ordinary H3 checkpoints run the fine sparse branch; gated FastH3/VSA checkpoints additionally run the learned coarse correction, which generally preserves quality better. Ordinary H3 emits an explicit warning instead of being rejected. Both paths still stop before sampling if their producer is unavailable instead of silently running dense attention.

### SM75 / RTX 20 series

| ID | Computation path |
|---|---|
| `sla_sm75_qk_int8_pv_fp16` | SLA with INT8 QK, FP16 PV, and FP32 softmax/accumulation |
| `sla_sm75_all_int8` | SLA with INT8 QK/PV and protected full attention for target-audio queries |
| `sol_sm75_all_int8` | Sol Q64/K64 exact selected blocks plus centroid approximation, with INT8 PV |
| `vsa_sm75` | H3 VSA with 10% keep over the full 0%–100% interval; ordinary H3 is fine-only, while gated FastH3 adds coarse correction |
| `hybrid_sm75_ck_sla_all_int8` | CK / SLA All-INT8 / CK across sampling steps |
| `hybrid_sm75_ck_sol_all_int8` | CK / Sol All-INT8 / CK across sampling steps |
| `hybrid_sm75_ck_vsa` | CK / SM75 VSA / CK across sampling steps |

### SM80+ / RTX 30–50 series and newer

| ID | Computation path |
|---|---|
| `sla_sm80+_qk_int8_pv_bf16` | SLA with INT8 QK, BF16 PV, FP32 softmax/accumulation, and full-attention audio queries |
| `sla_sm80+_all_int8` | SLA INT8 QK/PV comparison mode with full-attention audio queries |
| `sol_sm80+_bf16_official` | Official NVIDIA BF16 exact+approx Sol-Attn with audio KV sinks and full-attention audio queries |
| `sol_sm80+_all_int8` | Star7 exact+centroid Sol with INT8 PV, audio KV sinks, and full-attention audio queries |
| `vsa_sm80+` | H3 VSA with 10% keep over the full 0%–100% interval; ordinary H3 is fine-only, while gated FastH3 adds coarse correction |
| `hybrid_sm80+_ck_sla_qk_int8_pv_bf16` | CK / SLA BF16-PV / CK |
| `hybrid_sm80+_ck_sol_bf16_official` | CK / official NVIDIA BF16 Sol / CK |
| `hybrid_sm80+_ck_sla_all_int8` | CK / SLA All-INT8 / CK |
| `hybrid_sm80+_ck_sol_all_int8` | CK / Star7 Sol All-INT8 / CK |
| `hybrid_sm80+_ck_vsa` | CK / SM80+ VSA / CK |

SLA uses dynamic Top-K block routing. Sol combines exact selected-block contributions with centroid approximations for non-selected blocks. Hybrid switches the backend between complete denoising steps; it does not mix two kernels inside one attention call. CK/VSA Hybrid uses CK for the protected first and last steps and VSA for the middle steps. The BF16 Hybrid IDs remain available for existing workflows; the All-INT8 Hybrid IDs are separate opt-in modes and are not silent migrations.

On SM80+, every SLA and Sol mode replaces sparse results for reference- and generated-audio query ranges with full attention computed from the pre-quantization Q/K/V tensors. Video queries remain sparse. Hybrid inherits the same protection during its sparse steps.

Sparse attention is not guaranteed to outperform CK at every resolution, duration, or GPU. Compare sampling time under the same model, seed, frame count, step count, and offload policy.

## W4A8 model support

Enhanced Loader supports mixed-precision H3 checkpoints in ComfyUI's `asym_w4a8_int8` format, including `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors` and `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`. Put the checkpoint in a configured `diffusion_models` / `unet` search directory and select it in `MiniMax H3 Enhanced Loader - Star7`. Keep the text encoder, VAE and conditioning nodes from the matching H3 workflow. The environment must provide ComfyUI and Comfy Kitchen `AsymW4A8Int8Layout` support.

SM75 native acceleration accepts supported 16-value codebooks, ConvRot groups of 256 and grouped-scale layouts. Layers outside that native contract retain upstream quantized execution. QKV / MLP chunking can reuse prepared weights. W4A8 and VEDA operate at different stages and can be combined with VEDA + chunked CK; do not additionally stack SLA / Sol / VSA sparse attention.

W4A8 GEMM, VEDA and preprocessing use independent CUDA DLLs without Python/PyTorch C++ ABI linkage. CPython 3.12, compilation tools and Python development files are not required by the bundled kernels. Windows x64, a CUDA-13-compatible NVIDIA driver and CUDA PyTorch / Comfy Kitchen supporting the checkpoint format are required. Update each DLL with its checksum manifest; unavailable libraries retain upstream execution and log the reason. This does not imply support for arbitrary INT4 / QuantFunc checkpoints or higher speed than INT8. Compare peak VRAM, runtime and quality under identical settings.

Protected FP16 W4A8 on SM75 fuses SwiGLU, rescaling and casting to reduce FP32 intermediates. Standalone chunked CK can compress Q/K during projection under memory pressure, preserving the global nine-sample K anchor and checking quantization against the installed Kitchen on first use. Ample-memory runs and attention overrides, including VEDA, retain the original QKV path. VEDA keeps floating predictor features with separate workspace limits and caches. No extra nodes or controls are needed.

## Nodes

| Node | Purpose |
|---|---|
| `MiniMax H3 Enhanced Loader - Star7` | Independent H3 model loader bundled with this project; selects protected FP16 or native BF16 by GPU architecture, preserves quantized dispatch, and uses a distinct class ID to avoid conflicts with the standalone FP16 project |
| `MiniMax H3 VDN Acceleration - Star7` | Applies a complete VDN stage with its trained hybrid attention and adapters; supports DMD8 and the 50-step base mode |
| `MiniMax H3 VEDA Sparse Attention - Star7` | Optional predictor-driven sparse attention for normal H3 sampling and chunked CK; standalone CUDA DLL on SM75, Triton on SM80+; disabled mode passes the model through and diagnostics stay in logs |
| `MiniMax H3 VRAM Chunk Acceleration - Star7` | QKV/RoPE/MLP chunking, targeted OOM reduction, and attention selection |
| `MiniMax H3 Live Preview - Star7` | Uses TAEH3 after sampling steps to display a looping animation across the full timeline |
| `Reference Video Load - Star7` | Drag-and-drop video loading, time-range trimming, long-edge limiting, synchronized video/audio output |
| `Reference Image Load - Star7` | Drag-and-drop loading, long-edge limiting, optional upscale, and maximum-area centered cropping for common landscape/portrait ratios |
| `Prompt Load - Star7` | Extract prompts from dropped image, video, or workflow JSON files and retain alternative candidates |
| `Video and Workflow Export - Star7` | Export video alone or with embedded/separate workflow metadata |
| `DLSS Neural Image Enhance V2 - Star7` | Adjustable Neural Rendering for an image or video-frame batch, with a target megapixel count and realistic/portrait/anime presets |
| `MiniMax H3 All-in-one Conditioning - Star7` | Builds text, keyframe, reference image/video, and audio conditioning in one node and emits reusable sampling context |
| `MiniMax H3 One-click HD Upscale - Star7` | Upscales the sampled H3 latent to a target megapixel count with optional short refinement; VAE decoding remains external |
| `MiniMax H3 Chunked Decode - Star7` | Independently decodes a complete H3 audio-video latent using the current H3 VAE's native temporal streaming and spatial tiling |
| `MiniMax H3 One-click Face Repair - Star7` | Detects and tracks faces, samples repair crops, and attaches them to the original H3 latent for final RGB compositing in Star7 Chunked Decode |

Chinese ComfyUI environments display Chinese node and control labels; other locales display English. Attention backend IDs remain unchanged.

Connected media inputs show their prompt tags: `<Picture N>`, `<Video N>`, and `<Audio N>`; driving audio uses `<Audio D>`.

Place the complete DMD8 stage under `ComfyUI/models/vdn/<model folder>` and connect VDN between the H3 model loader and the Star7 chunk node. The INT8 ConvRot stage is recommended. Use an H3 base without Turbo or other LoRAs already fused in; VDN applies its own trained adapters. The chunk node preserves VDN attention automatically.

Reference-image limits accept `0` to preserve source size, `0 < value <= 10` as a downscale-only megapixel cap, and larger values as a pixel long-edge cap. Cropping precedes resizing. Preview decoders are selectable, existing workflow values are restored automatically, and tiled HD logs report each tile prediction.

### H3 One-click Face Repair

Connect Sampled result and Sampling context. The model, LoRA, and attention inherit the first pass by default. Disabling repair or detecting no face passes the latent through unchanged.

- Repairs 1–4 faces selected by Main, Center, or Reference Match; fewer detections automatically reduce the processed count.
- Includes Balanced, Realistic, Distant Face, Anime, and Custom presets, with optional dedicated LoRA and attention settings.
- Star7 Chunked Decode composites the repaired crops in the final RGB frames, avoiding a full-video VAE re-encode. Connect Face Repair directly to that decoder.
- Preserve repair detail only enlarges sources below about 1 MP; larger videos are never reduced.

The verified face detector downloads automatically to `ComfyUI/models/ultralytics/bbox`. Reference Match can optionally use InsightFace; other modes do not require it.

```text
Sampler Sampled result -> One-click HD -> One-click Face Repair -> H3 Chunked Decode
All-in-one Sampling context --------------------^              Video/audio VAEs --^
```

### H3 One-click HD Upscale

The node enlarges an H3 sampled latent to the target megapixel count and optionally applies a short second pass. It passes through unchanged when disabled or when the target does not exceed the source resolution.

- Balanced, High Quality, Distant Face, and Fast Motion presets pair suitable refinement steps and strength; Custom remains adjustable.
- LoRA and attention inherit the first pass by default or can be selected independently for refinement. The node displays the effective Sigma, Shift, and detected model profile.
- Tiling selects an aspect-aware overlapping grid to reduce peak VRAM at the cost of additional runtime.
- The original prompt and reference conditions are retained; first and last frames are rebuilt at the target resolution.

The required model is `minimax_h3_latent_upscaler_3d_fp16.safetensors`. If missing, it is downloaded, verified, and installed under `ComfyUI/models/latent_upscale_models`. VAE decoding remains external; connect the output to MiniMax H3 Chunked Decode.

```text
Sampler Sampled result -> One-click HD Sampled result -> H3 Chunked Decode -> Video Combine
All-in-one Sampling context ----------------^             Video/audio VAEs --^
```

### DLSS Neural Image Enhance V2

Put DLSS NR models in `models/upscale_models`. The default file is:

```text
ComfyUI/models/upscale_models/nvngx_dlssnr.dll
```

The model selector at the top lists the default model and `nvngx_dlssnr*.dll` variants in that directory, including subdirectories. Switching models reinitializes the renderer. Older workflows retain their parameter order and use `nvngx_dlssnr.dll` by default.

The node accepts a single image or a video-frame batch and provides Realistic, Portrait, 3D Anime, 2D Anime, and Custom presets. Custom values are stored in the workflow, and Reset restores the preset defaults.

`Target pixels (MP)` preserves the source aspect ratio and never downsizes when the target is below the input resolution. If the default model is missing, the node tries the HF mirror, Hugging Face, and GitHub, then installs it only after verification. Other selected variants must already exist in the model directory.

`nvngx_dlssnr.dll` provides Neural Rendering. Enlargement combines high-quality resizing with NR at the target size; it is not the complete in-game DLSS Super Resolution rendering pipeline.

## Core parameters

| Parameter | Purpose | Suggested start |
|---|---|---:|
| `chunk_tokens` | RoPE token chunk limit | `8192` |
| `mlp_chunk_tokens` | MLP expanded-activation chunk limit; usually the primary VRAM control | `8192`, then `4096` when memory is tight |
| `qkv_chunk_tokens` | QKV projection workspace chunk limit | `8192`, then lower when needed |
| `auto_halve_on_oom` | Retry only the failed chunk stage at half size | `true` |
| Attention output memory protection | Off installs no `out_proj` wrapper; select Auto explicitly to protect risky long sequences | `Off` |
| `reuse_mlp_weights` | Reuse prepared QKV/MLP weight snapshots when safe | `true` |
| Attention acceleration method | Select existing, CK, SLA, Sol, or Hybrid attention | `comfy_kitchen_int8` |

Automatic downshift and attention-output protection are independent. The first
handles recoverable QKV, RoPE, and MLP chunk OOMs; the second handles only the
attention `out_proj` peak. Retired prefetch values in older workflows migrate
to Auto, and the internal fallback tile is not exposed in the UI.

For RTX 20-series GPUs, pair this project with [MiniMax H3 FP16 Exact Fix - Star7](https://github.com/star7code/minimax-h3-fp16-exact-star7). It adds FP16 numerical protection without converting CK/SLA/Sol INT8 attention calculations to FP16.

Compressed/T8 H3 checkpoints can stack native 8-wide LoRAs with converted full-model 2688-wide LoRAs loaded through ComfyUI's standard LoRA node. The chunk node maps only incompatible full-width AdaLN contributions through the compressed time curve and leaves native T8 and compatible backbone patches unchanged. An unconverted original Turbo LoRA still requires its dedicated loader; FastH3 VSA `adapter_model.safetensors` is conversion input rather than a runtime LoRA.

## Installation

Comfy CLI:

```bash
comfy node install minimax-h3-chunk-star7
```

Manual installation:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/star7code/minimax-h3-chunk-star7.git
```

Restart ComfyUI after installing or updating.

## Example workflow

- [General workflow - English](examples/workflows/MiniMax-H3-Activation-Chunk-Star7-English.json): fully translated canvas and notes with all-in-one conditioning, chunk acceleration, live preview, independent H3 chunked decode, and optional second-pass refinement disabled by default. It is ready to use as a normal workflow immediately after import.
- [通用工作流（中文）](examples/workflows/MiniMax-H3-Activation-Chunk-Star7.json): Chinese version with the same features and defaults.

## Recorded 1.0MP / 10-second result

Test conditions: 1.0MP, 10 seconds, 24fps, MiniMax H3 INT8 Tensorwise + ConvRot, 768p Turbo 4-step LoRA, Euler/simple, RTX 2080 Ti 22GB.

| Attention path | Average sampling | Complete task | Relative CK step throughput |
|---|---:|---:|---:|
| KJNodes SM75 SageAttention 2 | `190.50 s/step` | `863.41 s` | about `0.63×` |
| Comfy Kitchen INT8 | `119.50 s/step` | `620.32 s` | `1.00×` |
| SLA SM75 QK-INT8/PV-FP16 | `96.68 s/step` | `471.12 s` | about `1.24×` |
| SLA SM75 All-INT8 | `60.83 s/step` | `325.51 s` | about `1.96×` |
| Sol SM75 All-INT8 | `88.71 s/step` | `442.57 s` | about `1.35×` |
| Standard CK + Sol Hybrid | `106.12 s/step` | `498.27 s` | about `1.13×` |
| Standard CK + SLA Hybrid | `94.67 s/step` | `454.76 s` | about `1.26×` |

These are observations from one local configuration, not cross-GPU performance guarantees. See [BENCHMARKS.md](BENCHMARKS.md) for methodology and additional details.

## Compatibility notes

- SM75 Windows x64 ships with a CUDA 13 static-runtime DLL and requires an NVIDIA 580+ driver.
- SM75 Linux x86_64 ships with a CUDA 12.6 static-runtime `.so`, targets Ubuntu 20.04 / glibc 2.31 or newer, and requires driver 525.60.13+.
- SM80+ SLA paths use Triton and compile/cache kernels on first use.
- Official BF16 Sol first uses ComfyUI 0.34's compiled `comfy_kitchen.sol_attn`
  dispatcher when available, then falls back to the bundled NVIDIA Triton path.
- BF16 remains the default on SM80+. If the launcher explicitly enables `--fp16-unet`, the latest Star7 loader installs FP16 Exact protection so CK, SLA, Sol, and Hybrid can continue. Only ordinary unprotected FP16 is rejected before sampling with a clear loader/launcher diagnostic.
- The official SM80+ Sol mode bundles the relevant NVlabs/Sana `sol-engine` source.
- Strict SLA/Sol/Hybrid modes stop on architecture, environment, self-test, or computation failures; they do not silently fall back to CK or Sage.
- NaN/Inf guards detect and locate invalid output. They do not replace invalid values with zero and are not an FP16 repair mechanism.

## H3 Live Preview

`MiniMax H3 Live Preview - Star7` has a top-level Show preview switch. Off returns the incoming MODEL unchanged and installs no sampling callback, decoder load/download, decode, encoding worker, or transport. When enabled, it decodes uniformly sampled temporal positions after each eligible H3 sampling step with `taeh3.safetensors`; the installed alias `taeh3_decoder.safetensors` is also accepted. Hovering over the preview reveals a compact timeline: dragging scrubs across the sampled positions, and releasing resumes looping from the selected frame. Returning from another browser tab redraws retained frames or recovers the latest WebP from the backend if the page slept through its websocket event. The animation still updates at each configured step, but repetitive model-residency and successful-encode INFO messages from preview housekeeping are suppressed; initial decoder detection, download state, and all warnings/errors remain visible so sampling-speed lines stay easy to compare. If neither decoder filename is present in `models/vae_approx`, the node races the HF mirror and the pinned madebyollin/taehv source first, then tries the remaining fallbacks, verifying SHA-256 without blocking sampling. Preview starts at the next sampling callback after the download completes. If it becomes ready only at the final callback and no earlier preview was shown, one final preview is emitted. Download or preview failure never stops the main generation.

The default preview uses 5 frames per second across the complete timeline at a 512-pixel long edge. `First step only` is disabled by default; when enabled, only Step 1 is decoded and all later preview work is skipped.

## License

Star7 code is distributed under the [MIT License](LICENSE). Bundled NVIDIA Sol-Attn source is distributed under its [Apache 2.0 license and third-party notices](vendor/sol_attn/THIRD_PARTY_NOTICES.md). The H3 AdaLN curve grid and adaptation provenance are documented in the [Larryvrh H3 Turbo notice](vendor/LARRYVRH-H3-TURBO-NOTICE.md).

## VEDA and live preview (2.18.1)

VEDA has an enable switch and works with normal H3 sampling or Star7 chunked CK. Disable it to pass the model through without loading the predictor or installing its patch. SM75 uses the bundled Star7 CUDA INT8 QK / FP16 PV kernel; SM80+ uses upstream Triton. Do not combine VEDA with another sparse-attention backend. Predictor: `models/veda/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors`. Diagnostics are logged instead of displayed in the node. Speed varies by workload.

TAEH3 previews retain native latent resolution and continuous temporal state. FPS (1-24, default 5) selects frames before the final spatial/RGB decoder tail, transfer and encoding; temporal-state computation still runs. A 10-second video at 5 FPS shows about 50 frames. The 256/384/512/768/1024 long-edge cap is applied after RGB decoding. WebP quality (1-100) affects compression only. Neither setting changes final output resolution. Older workflows retain quality 76; new nodes default to 80.

Since 2.18.1, Windows x64 VEDA uses a standalone CUDA DLL with no fixed Python/PyTorch C++ ABI dependency. Python headers, import libraries and local compilation are unnecessary. CUDA and C++ runtimes are statically linked; a CUDA-13-compatible NVIDIA driver and working CUDA-enabled PyTorch are required. Update the DLL and checksum manifest together, then restart ComfyUI. SM80+ retains upstream Triton. RTX 2080 Ti kernel numerical checks and normal/CK integration tests passed; RTX 2060 still requires device-side validation. No VEDA SM75 Linux binary is bundled. Third-party licenses and sources are included in `vendor/veda/LICENSE` and `vendor/veda/NOTICE.md`.
