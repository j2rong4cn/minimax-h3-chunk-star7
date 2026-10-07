# MiniMax H3 Activation Chunk & Attention Acceleration - Star7

[中文说明](#中文说明) · [English](README_EN.md) · [实测记录](BENCHMARKS.md) · [示例工作流](workflows)

1006 中英文示例包含：一采 → 可选二采高清 → 可选人脸修复 → Star7 分块解码，加入按帧率控制的实时预览及默认旁路的 VEDA。请自行选择已安装的模型并替换参考素材占位文件。

This ComfyUI project helps MiniMax H3 run high-quality, long-duration video generation on GPUs with limited VRAM. Its core node provides independent QKV, RoPE, and MLP activation chunking together with a selectable attention backend. It does not change the sampler, sigma schedule, latent layout, VAE, duration, frame count, or output resolution.

Attention choices include preserving an upstream backend, Comfy Kitchen INT8, architecture-specific SLA/Sol sparse attention, and step-level CK/Sparse/CK Hybrid modes. The package also includes compact reference-image, reference-video, and prompt-loading helpers.

> This is an independent community project. MiniMax, ComfyUI, Comfy Kitchen, KJNodes, LightX2V, and NVIDIA are trademarks or projects of their respective owners.

## 中文说明

让高画质、长时长 MiniMax H3 视频在有限显存的显卡上高效运行。

### 快速看懂

- 核心节点同时提供 **QKV / RoPE / MLP 激活分块**和**注意力选择**。
- QKV 投影、RoPE 与 MLP 扩展激活沿 token 维分块计算，使每次只需保留当前分块的中间张量，分别避免这三处临时工作集按完整长序列规模展开；显存不足时还可只降低发生 OOM 的分块大小。
- 分块降低的是推理临时激活峰值，不减少模型权重、帧数或 token，也不改变采样器、latent、VAE、时长和输出分辨率。
- 注意力可选择传入模型已有实现、Comfy Kitchen INT8、SLA、Sol，以及 CK/Sparse/CK Hybrid；SLA 仅计算路由选中的 K/V 块，Sol 以选中块精确计算和未选中块质心近似替代完整稠密计算，SM75 与 SM80+ 均提供对应路径。

## 主要功能

| 功能 | 说明 |
|---|---|
| QKV / RoPE / MLP 独立分块 | 分别控制三处临时显存峰值，保留 MiniMax H3 原 block、权重、LoRA 和条件布局 |
| 智能显存降档 | 仅在可缩小的分块阶段 OOM 时，将对应分块值减半重试，其他参数保持不变 |
| 多种注意力后端 | 支持 `existing`、Comfy Kitchen INT8、SLA、Sol 和 CK/Sparse/CK Hybrid |
| H3 W4A8 模型支持 | 增强载入保留 `asym_w4a8_int8` 量化布局；SM75 提供原生码本 W4A8 线性层优化，可搭配 CK 分块与 VEDA |
| 架构专用稀疏内核 | SM75 使用随节点分发的原生 CUDA 内核；SM80+ 使用 Triton 或 NVIDIA 官方 Sol-Attn 路径 |
| 数值检查与定位 | 在完整 Transformer block 和 H3 视频/音频输出处检查 NaN/Inf，并提供 QKV、attention、`out_proj`、MLP 分段诊断 |
| 轻量辅助节点 | 附带参考图像、参考视频和提示词载入；支持将对应文件直接拖入节点，参考图像可按常用横竖比例自动进行最大保留裁切，且不影响核心模型补丁 |

## 工作原理

### 激活分块

MiniMax H3 长序列推理主要存在三个可独立控制的临时显存峰值：

- `QKV projection`；
- `RMSNorm -> split-half RoPE (Q/K)`；
- `fc1 -> SwiGLU -> fc2` MLP 扩展激活。

本节点沿 sequence/token 维切分这些行独立计算，并将结果写回原有输出布局。RoPE 原位写回 Q/K，MLP 输出继续交给上游 block；分块不改变公式、token 顺序或条件结构。

分块的作用是降低临时工作集，而不是减少计算量。如果任务原本可以完整驻留显存，分块不一定更快；当原任务会 OOM、进入共享显存或频繁换页时，分块才更可能改善实际吞吐。

### SLA

SLA 按 LightX2V 契约使用 `Q=128`、`K=64` 动态 Top-K 块路由。目标视频查询通常只保留约 15% 的 K 块参与 QK/PV，online softmax 状态和最终累积保持 FP32。

SM75 原生内核对目标音频查询块执行完整注意力。SM80+ 在保留边界路由的同时，以量化前 Q/K/V 对参考音频和生成音频的查询范围执行完整注意力并覆盖稀疏结果；视频查询仍使用 SLA 稀疏计算。

SLA 建议配合 [MiniMax H3 Turbo SLA LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) 使用，以获得更适合该稀疏模式的长时序表现。

### Sol

Sol 使用 `Q64/K64`、`tau=1.0` 的阈值路由。被选中的 K/V 块执行精确注意力；未命中块不是直接丢弃，而是通过 K/V 质心近似贡献，并与精确块合并到同一次 FP32 online softmax 中。

SM80+ 标准 Sol 模式调用随节点内置的 NVIDIA NVlabs/Sana 官方 BF16 Sol-Attn 接口。SM75 与 SM80+ 的 Star7 All-INT8 路径保留“精确块 + 未命中块质心近似”的完整 Sol 语义，但采用不同的 Q/K/PV 量化和 CUDA/Triton 实现，因此不具备与官方 BF16 路径逐值一致性。

SM80+ Sol 保留音频范围的 KV sink，并对参考音频和生成音频查询使用完整注意力结果覆盖；该保护同时适用于官方 BF16、All-INT8 和 Hybrid 中的 Sol 步。

### Hybrid

Hybrid 在完整采样 step 之间切换注意力，而不是在一次 Attention 内混合两套内核。默认前后保护区使用 CK，中间采样阶段使用对应的 SLA、Sol 或 VSA。

以常见 4-step 工作流为例：第 1、4 步使用 CK，第 2、3 步使用所选稀疏模式。Hybrid 需要 ComfyUI 提供真实 sigma 调度上下文；无法可靠确定采样步时将终止任务并报告错误。

RTX 20 系建议配合 [MiniMax H3 FP16 Exact Fix - Star7](https://github.com/star7code/minimax-h3-fp16-exact-star7) 使用。该项目从模型载入阶段采用 FP16 计算，并以 FP32 残差运算和精确溢出保护修复 Turing 上的 FP16 数值问题，可降低长视频中 NaN、棋盘格和音频异常的风险；它不改变注意力后端，也不会把 CK、SLA 或 Sol 的 INT8 计算转换为 FP16。

## 注意力模式

### 通用模式

| `attention_backend` | 作用 | 适用情况 |
|---|---|---|
| `existing` | 保留传入模型已有的注意力实现 | 已接 Sage、Low VRAM 或其他注意力补丁；也可保留当前环境默认后端 |
| `comfy_kitchen_int8` | 使用 ComfyUI / Comfy Kitchen INT8 注意力 | 默认选项，兼容性与速度较均衡；属于近似注意力 |

### SM75 / RTX 20 系

| `attention_backend` | 计算路径 | 定位 |
|---|---|---|
| `sla_sm75_qk_int8_pv_fp16` | QK INT8、PV FP16、FP32 softmax/累积 | SLA 精度优先 |
| `sla_sm75_all_int8` | QK/PV INT8、FP32 softmax/累积、完整音频查询保护 | SLA 性能优先；建议按提示词验证画质与语音 |
| `sol_sm75_all_int8` | Q64/K64，精确块 + 质心近似，PV INT8 | SM75 Sol 标准可见模式 |
| `vsa_sm75` | H3 VSA、10% 保留、0%–100% 全采样区间、SM75 专用路径 | 普通 H3 运行 fine-only；FastH3/VSA 额外使用学习型 coarse 补偿 |
| `hybrid_sm75_ck_sla_all_int8` | CK / SLA All-INT8 / CK | 采样步级质量与速度折中 |
| `hybrid_sm75_ck_sol_all_int8` | CK / Sol All-INT8 / CK | 采样步级质量与速度折中 |
| `hybrid_sm75_ck_vsa` | CK / SM75 VSA / CK | 首尾 CK、中间使用 SM75 预编译 VSA |

### SM80+ / RTX 30–50 系及更新架构

| `attention_backend` | 计算路径 | 定位 |
|---|---|---|
| `sla_sm80+_qk_int8_pv_bf16` | QK INT8、PV BF16、FP32 softmax/累积、完整音频查询保护 | SM80+ SLA 标准模式 |
| `sla_sm80+_all_int8` | QK/PV INT8、FP32 softmax/累积、完整音频查询保护 | 性能/质量对照模式，需实机验证 |
| `sol_sm80+_bf16_official` | NVIDIA 官方 BF16 exact+approx Sol-Attn、音频 KV sink 与完整音频查询保护 | SM80+ Sol 标准模式 |
| `sol_sm80+_all_int8` | Star7 exact+centroid Sol、PV INT8、音频 KV sink 与完整音频查询保护 | 性能/质量对照模式，需实机验证 |
| `vsa_sm80+` | H3 VSA、10% 保留、0%–100% 全采样区间、SM80+ 路径 | 普通 H3 运行 fine-only；FastH3/VSA 额外使用学习型 coarse 补偿 |
| `hybrid_sm80+_ck_sla_qk_int8_pv_bf16` | CK / SLA BF16-PV / CK | SM80+ Hybrid SLA 标准模式 |
| `hybrid_sm80+_ck_sol_bf16_official` | CK / NVIDIA 官方 BF16 Sol / CK | SM80+ Hybrid Sol 标准模式 |
| `hybrid_sm80+_ck_sla_all_int8` | CK / SLA All-INT8 / CK | SM80+ Hybrid SLA 性能模式 |
| `hybrid_sm80+_ck_sol_all_int8` | CK / Star7 Sol All-INT8 / CK | SM80+ Hybrid Sol 性能模式，需实机验证 |
| `hybrid_sm80+_ck_vsa` | CK / SM80+ VSA / CK | 首尾 CK、中间使用 Comfy Kitchen VSA |

### 如何选择

- 追求最高兼容性或保留已有 Sage：选择 `existing`。
- 通用配置：优先选择 `comfy_kitchen_int8`。
- SM80+ 默认推荐 BF16；若启动器明确开启 `--fp16-unet`，最新版 Star7 载入节点会安装 FP16 Exact 保护，CK、SLA、Sol 与 Hybrid 均可继续运行。只有未经保护的普通 FP16 会在采样前被拦截并提示检查载入节点与启动参数。
- SM75 使用 SLA：先以 `sla_sm75_qk_int8_pv_fp16` 验证质量，再根据需求测试 All-INT8 或 Hybrid。
- SM80+ Hybrid 可按需求选择 BF16 标准模式（`hybrid_sm80+_ck_sla_qk_int8_pv_bf16`
  或 `hybrid_sm80+_ck_sol_bf16_official`），也可显式选择新增的 All-INT8
  性能模式（`hybrid_sm80+_ck_sla_all_int8` 或
  `hybrid_sm80+_ck_sol_all_int8`）。旧工作流中的 BF16 Hybrid 值会原样保留，
  不会静默改变为 All-INT8。
- SM80+ 单独 Sol 优先从 `sol_sm80+_bf16_official` 开始；All-INT8 只应在同配置 A/B 测试后采用。
- 稀疏模式并非所有分辨率、时长和显卡上都必然快于 CK，应比较同模型、同 seed、同帧数、同步数和同卸载策略下的采样耗时。

## W4A8 模型支持

增强载入节点支持使用 ComfyUI `asym_w4a8_int8` 格式的 H3 混合精度模型，包括 `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors` 和 `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`。将模型放入已配置的 `diffusion_models` / `unet` 搜索目录，使用「MiniMax H3 增强载入 - Star7」选择模型；文本编码器、VAE 和采样条件继续使用对应 H3 工作流的原有节点。环境需提供 ComfyUI 与 Comfy Kitchen 的 `AsymW4A8Int8Layout` 支持。

SM75 原生优化适用于受支持的 16 值码本、ConvRot 256、分组尺度布局；不满足原生优化条件的层保留上游量化计算路径。QKV / MLP 分块可复用权重，减少重复准备。W4A8 与 VEDA 作用于不同环节，可搭配 VEDA + CK 分块；不要再叠加 SLA / Sol / VSA 等其他稀疏注意力。

W4A8 原生 GEMM、VEDA 与预处理均使用独立 CUDA DLL，不链接 Python / PyTorch C++ ABI，无需固定 CPython 3.12，也不要求用户安装编译工具或 Python 开发文件。需要 Windows x64、兼容 CUDA 13 的 NVIDIA 驱动，以及具备对应模型格式支持的 CUDA PyTorch / Comfy Kitchen 环境。DLL 与校验清单须一起更新；库不可用时保留上游路径并写明原因。这里的支持不等于兼容任意 INT4 / QuantFunc 模型，也不保证比 INT8 更快；峰值显存、速度与画质应按相同参数比较。

SM75 的受保护 FP16 W4A8 路线融合 SwiGLU、缩放和转换，减少 FP32 临时张量。独立 CK 分块在显存压力下可提前压缩 Q/K，保留全序列 9 点 K 锚定规则，并在首次使用时核对当前 Kitchen 的量化结果。显存充足或连接注意力 override（包括 VEDA）时保持原 QKV 路线；VEDA 预测器继续读取浮点特征，并使用独立的工作区限制和缓存。无需新增节点或参数。

## 环境与分发

- SM75 Windows x64：节点内置预编译 CUDA 13 静态运行时 DLL，需要支持 CUDA 13 的 NVIDIA 580+ 驱动。
- SM75 Linux x86_64：节点内置 CUDA 12.6 静态运行时 `.so`，面向 Ubuntu 20.04 / glibc 2.31 及更新系统，需要 NVIDIA 525.60.13+ 驱动。
- SM75 原生库只接收张量地址、形状和当前 CUDA stream，不链接 PyTorch C++ ABI，也不依赖 SageAttention。Turing Triton 不可用时，路由/量化预处理可使用有界显存 PyTorch 路径，核心稀疏注意力仍由原生 CUDA 库执行。
- SM80+ SLA 与 All-INT8 路径使用 Triton，首次运行会编译并写入缓存，后续运行复用。
- SM80+ 官方 BF16 Sol 优先使用 ComfyUI 0.34 `comfy_kitchen.sol_attn` 的已编译后端；该接口不可用时才使用节点内置 NVIDIA Triton 实现。
- `sol_sm80+_bf16_official` 已内置 NVlabs/Sana `sol-engine` 源码，无需另外安装 Sana。SM80/SM86 使用官方 Triton；支持的 SM89/SM90/SM100/SM120 环境在 CuTe DSL 与 `cuda-python` 可用时使用对应专用内核，否则由官方接口使用 Triton。
- `STAR7_SOL_ATTN_PATH` 仅用于开发者可选覆盖为更新的官方 Sol 源码，普通用户不需要设置。
- 架构选项不按当前显卡动态隐藏，便于跨机器保存和分发工作流。架构不匹配时任务终止，错误信息包含所需与检测到的计算能力，且不执行后端回退。

## 安装

ComfyUI Manager / Comfy Registry：

```text
搜索：MiniMax H3 Activation Chunk - Star7
包名：minimax-h3-chunk-star7
```

Comfy CLI：

```bash
comfy node install minimax-h3-chunk-star7
```

GitHub 手动安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/star7code/minimax-h3-chunk-star7.git
```

安装或更新后重启 ComfyUI。

## 包含的节点

| 节点 | 用途 |
|---|---|
| `MiniMax H3 增强载入 - Star7` | 分块项目内置的独立 H3 模型载入节点；按 GPU 架构选择受保护 FP16 或原生 BF16，保留量化分发，并使用独立类 ID 避免与 FP16 项目冲突 |
| `MiniMax H3 VDN 加速 - Star7` | 加载完整 VDN stage，自动安装混合注意力与配套 Adapter；支持 8 步 DMD 和 50 步原始模式 |
| `MiniMax H3 VEDA 稀疏注意力 - Star7` | 可选的预测器驱动稀疏注意力；支持普通 H3 采样和 CK 分块，SM75 使用独立 CUDA DLL，SM80+ 使用 Triton；关闭时直接透传模型，诊断只写日志 |
| `MiniMax H3 显存分块加速 - Star7` | QKV/RoPE/MLP 分块、注意力输出显存保护、自动降档和注意力加速选择 |
| `MiniMax H3 实时预览 - Star7` | 每个采样步骤后用 TAEH3 显示覆盖完整时间轴的循环动画 |
| `参考视频载入 - Star7` | 支持直接拖入视频，完成载入、时间范围裁切和最长边限制，输出同一时间窗的画面与音频 |
| `参考图像载入 - Star7` | 支持直接拖入图片，在一个节点中完成载入、最长边限制、可选小图放大及常用横竖比例的最大保留裁切 |
| `提示词载入 - Star7` | 支持拖入图片、视频或工作流 JSON，自动提取长文本并保留候选词 |
| `DLSS 神经画质增强 V2 - Star7` | 对图片或视频帧批次执行可调 Neural Rendering，可选目标百万像素与写实/人像/动漫预设 |
| `MiniMax H3 多合一条件载入 - Star7` | 在一个节点中组织提示词、首尾帧、参考图、参考视频和音频，并输出可供高清放大与人脸修复复用的采样上下文 |
| `MiniMax H3 一键高清放大 - Star7` | 将采样结果直接放大至目标百万像素并可选短程高清修复；输出仍为标准 H3 采样结果，VAE 在节点外独立解码 |
| `MiniMax H3 分块解码 - Star7` | 独立解码完整 H3 音视频 latent；视频使用当前 H3 VAE 自带的时序流式与空间分块路径 |
| `MiniMax H3 一键人脸修复 - Star7` | 接收采样后的 H3 音视频 latent，检测、跟踪并局部修脸；保留独立修复裁切，由 Star7 分块解码在最终 RGB 图像上贴回 |

三个载入节点均支持将对应文件直接拖到节点上完成载入；它们都是独立工具，不会向模型注入注意力或精度补丁。

多合一条件节点会在已连接素材后显示提示词标签：`<Picture N>`、`<Video N>`、`<Audio N>`；驱动音频固定为 `<Audio D>`。

VDN 节点连接在 H3 模型载入与 Star7 分块节点之间。将完整 DMD8 stage 放入 `ComfyUI/models/vdn/<模型目录>`；推荐使用 INT8 ConvRot 版。请载入未预先融合 Turbo/其他 LoRA 的 H3 基础模型；VDN 自带配套适配器。分块节点会自动保留 VDN 注意力。

参考图最长边限制支持 `0` 保持原尺寸、`0 < 值 ≤ 10` 按百万像素只缩小，以及大于 `10` 的最长边像素限制；启用比例调整时先裁切后缩小。预览模型可在节点内选择，旧工作流自动恢复参数。高清分格日志逐格记录每轮预测耗时。

### H3 一键人脸修复

接入“采样结果”和“采样上下文”即可使用，模型、LoRA 与注意力默认继承一采。关闭修复或未检测到人脸时原样透传，不增加采样与 VAE 编码。

- 支持 1–4 人，可按主人物、画面中央或参考图选择目标；人数不足时自动按实际检测数量处理。
- 提供自动平衡、真人保真、远景小脸、动漫角色和自定义预设；也可单独指定修脸 LoRA 与注意力。
- 修复结果由 Star7 分块解码在最终画面贴回，避免整段视频再次 VAE 编码；请将人脸修复直接接到 Star7 分块解码。
- “保持修复清晰度”默认开启，仅在源视频低于约 1.0MP 时等比例放大，较大视频不会缩小。

首次使用会自动下载并校验人脸检测模型到 `ComfyUI/models/ultralytics/bbox`。参考图匹配可选用 InsightFace，其余模式不需要。

```text
采样器“采样结果” -> 一键高清“采样结果” -> 一键人脸修复“采样结果” -> H3 分块解码
多合一“采样上下文” --------------------------^                    视频/音频 VAE --^
```

### H3 一键高清放大

节点将 H3 采样结果放大到目标像素并进行短程二采。目标不高于原分辨率时直接透传；关闭“启用高清二采”时也不会加载高清模型。

- 平衡高清、高质量、远景小脸和高速运动预设会自动匹配二采步数与强度；自定义模式可手动调整。
- LoRA 与注意力默认继承一采，也可仅为二采单独指定。底部会显示实际 Sigma、Shift 与识别出的模型类型。
- 分格会按画幅自动选择接近正方形的网格并保留重叠，可降低峰值显存，但会增加运行时间。
- 二采继续使用原提示词和参考条件；首尾帧会按目标分辨率重新编码。

高清模型为 `minimax_h3_latent_upscaler_3d_fp16.safetensors`，缺失时会自动下载、校验并保存到 `ComfyUI/models/latent_upscale_models`。节点不内置 VAE 解码，输出请连接“MiniMax H3 分块解码”。

```text
采样器“采样结果” -> 一键高清“采样结果” -> H3 分块解码 -> 视频合成
多合一“采样上下文” --------^             视频/音频 VAE --^
```

### DLSS 神经画质增强 V2

将 DLSS NR 模型放入 `models/upscale_models`，默认文件位置为：

```text
ComfyUI/models/upscale_models/nvngx_dlssnr.dll
```

节点顶部的「模型」选项显示默认模型及该目录（含子目录）中的 `nvngx_dlssnr*.dll` 变体，切换模型会重新初始化渲染器。旧工作流默认使用 `nvngx_dlssnr.dll`，原有参数位置与设置保持兼容。

节点支持单图和视频帧批次，提供真实风格、真实人像、3D 动漫、2D 动漫和自定义预设。自定义参数会随工作流保存，“重置参数”恢复预设默认值。

“目标像素 (MP)”按原图比例计算输出尺寸，设置低于原图时不会降分辨率。默认模型缺失时优先从 HF 国内镜像自动下载，再回退到 Hugging Face 和 GitHub，并在校验通过后安装；手动选择的其他模型须已存在于目录中。

该节点使用 `nvngx_dlssnr.dll` 进行 Neural Rendering；放大部分由高质量缩放配合目标尺寸 NR 完成，并非游戏渲染管线中的完整 DLSS Super Resolution。

## 推荐连接顺序

通用链路：

```text
UNET Loader -> LoRA -> Attention patch（可选）
            -> Activation Chunk - Star7 -> Live Preview - Star7 -> Guider -> Sampler
                                      `-> Scheduler -----------------> Sampler
```

实时预览顶部提供“显示预览”总开关；关闭后直接返回原 MODEL，不安装采样回调，也不加载、下载或解码 TAEH3。开启时默认按每秒 5 帧预览完整时间轴、预览长边 512；帧率可设置为 1–24。“只显示第一步预览”默认关闭，开启后仅在 Step 1 解码一次，后续采样步骤不再产生预览开销。鼠标悬浮在预览画面上会显示迷你时间轴，拖动时定位画面，松开后从当前位置继续循环播放。切换到其他网页后返回时会重绘保留帧；若页面休眠期间漏收事件，会从后端取回该节点最近一次 WebP。动态画面仍按设置逐步更新，但每步重复的模型驻留和成功编码 INFO 会被安静处理；首次解码器识别、下载状态及所有异常仍会记录，使采样速度行保持清晰。它只接在 Guider 的 MODEL 路径，Scheduler 可继续直接连接 Chunk。

实时预览使用约 22MB 的 `taeh3.safetensors`；也兼容已安装的 `taeh3_decoder.safetensors` 文件名。若两者都缺少，节点会优先并行尝试 HF 国内镜像与官方 madebyollin/taehv 固定版本，再按顺序切换其他来源，并校验 SHA-256；采样不会等待下载。下载完成后的下一个采样步骤会立即开始预览；若直到最终步骤才完成且此前没有显示过预览，则最终步骤只补发一次预览。下载失败只关闭预览，不影响正式生成。

RTX 20 系：

```text
MiniMax H3 Native FP16 Loader - Star7 -> LoRA
    -> Activation Chunk - Star7 -> Guider / Scheduler / Sampler
```

FP16 Loader 与 Chunk 是两个可以独立运行的节点。Chunk 不注入 FP16 Exact；SM75 未检测到独立修复节点时输出一次提示，SM80+ 跳过该项检测提示。

剪枝/T8 H3 可以同时使用原生 8 维 LoRA 与通过 ComfyUI 标准加载器载入的 2688 维完整模型转换版 LoRA。本节点只把尺寸不匹配的完整 AdaLN 增量映射到压缩时间曲线，原生 T8 AdaLN 和其余可匹配权重继续使用 ComfyUI 原路径。原版未转换 Turbo LoRA 仍需其专用加载器；FastH3 VSA 的 `adapter_model.safetensors` 是模型合并原料，不能作为普通 LoRA 加载。

如果前面已经安装 Sage 或其他注意力补丁，把本节点设为：

```text
attention_backend = existing
```

选择 `comfy_kitchen_int8` 或任一 SLA/Sol/Hybrid 时，本节点会在其输出模型上安装相应注意力路径。

## 核心参数

| 参数 | 作用 | 建议起点 |
|---|---|---:|
| `chunk_tokens` | RoPE token 分块上限 | `8192` |
| `mlp_chunk_tokens` | MLP 扩展激活分块上限，通常是主要显存调节项 | `8192`；显存紧张时先降至 `4096` |
| `qkv_chunk_tokens` | QKV 投影临时工作集 | `8192`；显存紧张时再降低 |
| `auto_halve_on_oom` | 当前分块 OOM 时只对失败阶段减半重试，最低 `256` | `true` |
| 注意力输出显存保护 | 默认关闭且不安装任何 `out_proj` 包装；仅在明确选择自动后保护高风险长序列 | `关闭` |
| `reuse_mlp_weights` | 安全时复用已准备的 QKV/MLP 权重快照，显存压力下改用 streamed 路径 | `true` |
| 注意力加速方式 | 选择已有、CK、SLA、Sol 或 Hybrid 注意力 | `comfy_kitchen_int8` |
| `verbose` | 输出紧凑配置、首个同形状 block 和注意力路由信息 | `true` |

注意力输出显存保护与自动降档互不重叠：前者仅处理 attention
`out_proj`，后者处理 QKV、RoPE 和 MLP。旧工作流中的“提前加载下一层”
字段会迁移为默认的“自动”，无需重新创建节点。内部 `out_proj` 分块大小不
对用户显示。

节点会分别显示 RoPE、MLP、QKV 的实际使用值：

- `当前使用 N（设定值）`：按设定运行；
- `当前使用 N（受序列长度限制）`：实际序列比设定短，不是降档；
- `已降级为 N（设定 M）`：该阶段曾 OOM，后续 block 和同一模型会话复用已验证值。

工作流输入不会被后台改写，临时降档值也不会保存回工作流。用户重新执行节点并更改参数后，之前记忆的降档值会重置。

`qkv_chunk_tokens` 在所有架构和生成类型中严格采用工作流设定值；设为 `0` 表示先尝试整段计算。开启自动降档后，只有实际发生 QKV 投影 OOM 才会降低该值。

### 按错误位置调节

| 错误位置或现象 | 优先处理 |
|---|---|
| `fc1` / SwiGLU / `fc2` MLP OOM | 降低 `mlp_chunk_tokens`：`8192 -> 4096 -> 2048 -> 1024` |
| `rms_rope_split_half_` / RoPE OOM | 降低 `chunk_tokens`：`8192 -> 4096 -> 2048` |
| QKV 投影临时张量 OOM | 降低 `qkv_chunk_tokens`：`8192 -> 4096 -> 2048 -> 1024` |
| attention kernel OOM | 更换注意力后端或减少序列/参考 token；激活分块不能缩小注意力核心工作集 |
| 模型加载阶段 OOM | 调整模型量化、加载或卸载策略；此时分块节点尚未执行 |

本机单 MLP 工作集测试中，MLP `8192 / 4096 / 2048 / 1024` 约占 `1970 / 1228 / 858 / 672 MiB`；同条件 RoPE 约为 `820 / 638 / 550 / 501 MiB`。因此显存紧张时通常先降低 MLP，无需同步降低全部参数。

## 1.0MP / 10秒注意力路径实测

统一测试条件：1.0MP、10 秒、24fps，MiniMax H3 INT8 Tensorwise + ConvRot 主模型、768p Turbo 4-step LoRA、Euler/simple。测试硬件为 RTX 2080 Ti 22GB。

| 注意力路径 | 平均采样 | 完整任务 | 相对 CK 单步吞吐 |
|---|---:|---:|---:|
| KJNodes SM75 SageAttention 2 | `190.50秒/步` | `863.41秒` | 约 `0.63×` |
| Comfy Kitchen INT8（基准） | `119.50秒/步` | `620.32秒` | `1.00×` |
| SLA SM75 QK-INT8/PV-FP16 | `96.68秒/步` | `471.12秒` | 约 `1.24×` |
| SLA SM75 All-INT8 | `60.83秒/步` | `325.51秒` | 约 `1.96×` |
| Sol SM75 All-INT8 | `88.71秒/步` | `442.57秒` | 约 `1.35×` |
| 标准 CK + Sol Hybrid | `106.12秒/步` | `498.27秒` | 约 `1.13×` |
| 标准 CK + SLA Hybrid | `94.67秒/步` | `454.76秒` | 约 `1.26×` |

Hybrid 的平均采样耗时按采样进度中各步实际耗时的算术平均值计算，不以包含模型初始化、VAE 和视频封装的完整任务耗时除以步数。

完整记录见 [BENCHMARKS.md](BENCHMARKS.md)。所有数字都是本机单配置观察值，不是跨显卡性能保证。

## 耗时与显存关系

令 `S` 为 H3 packed sequence token 数，可粗略写为：

```text
S ≈ S_condition + k × spatial_tokens × temporal_tokens
T ≈ T_fixed + aS + bS² + T_post
```

- RoPE、RMSNorm、投影和 MLP 主要随 `S` 线性增长；
- 全局注意力包含近似 `S²` 的计算项；
- 激活分块降低临时显存，不减少理论 FLOPs；
- SLA 通过减少参与 QK/PV 的 K 块降低注意力计算；
- Sol 同时保留命中块的精确贡献和未命中块的质心近似；
- 完整任务还包含模型调度、VAE、音频、超分和编码，不能只用采样步耗时推算总时间。

## Sage、第三方补丁与兼容性

已有 Sage 节点时可按以下顺序连接：

```text
Loader -> LoRA -> Sage Attention Patch -> Activation Chunk - Star7
```

并选择 `attention_backend=existing`。本节点只安装 QKV/RoPE/MLP 分块并保留传入模型的注意力实现。

上游增强载入节点已经安装 VSA 时，可以选择 `attention_backend=existing`，本节点只追加 QKV、RoPE 与 MLP 分块。也可以直接选择架构对应的 `vsa_sm75` 或 `vsa_sm80+`：SM75 使用 Star7 预编译 CUDA producer，SM80+ 使用 Comfy Kitchen Sol-Attn。普通 H3 模型可以运行 VSA fine 稀疏分支；带压缩门的 FastH3/VSA 模型还会运行学习型 coarse 补偿，通常更有利于保持画质。普通 H3 启用时会给出一次明确警告而不会被拒绝。producer 不可用时仍会在采样前终止，不会静默退回 dense。

外部 TE-Speed 等 block-loop 缓存可以接在本节点之前；完整步与缓存前缀仍会经过 Star7 block/attention 补丁。节点使用 model clone、弱绑定 forward 和可识别 wrapper 标记，避免重复包装与旧模型被闭包长期强引用。

数值兼容说明：

- RoPE 分块可以逐元素一致；
- MLP 公式、权重、dtype 和 token 顺序不变，但大 GEMM 拆成小 GEMM 后可能存在浮点末位差，因此是数值等价而非保证 bitwise identical；
- CK、SLA、Sol、Sage 和 All-INT8 都属于显式近似路径，输出像素可能不同；
- 只有检测到 MiniMax H3 典型 shape 与 RoPE frequency 布局时才安装分块补丁，其他模型保留原路径。

## 错误处理与日志

- Chunk 的自动降档只处理当前 QKV/RoPE/MLP 分块的可恢复 OOM；注意力输出显存保护独立处理 `out_proj` 峰值。完整 Q/K/V 分配、attention kernel 或模型加载 OOM 会在错误信息中单独区分。
- 严格 SLA/Sol/Hybrid 不进行 CK 或 Sage 回退。环境、自检、架构或计算失败时，需根据错误信息选择其他后端并重新运行。
- NaN/Inf 检查负责阻止损坏的 latent/音频继续进入 VAE 或 FFmpeg，不会把无效值替换成 0。它是检测与定位机制，不是 FP16 修复。
- 首次异常会记录 block、token 行范围和 head；下次运行仅对目标 block 启用 QKV、attention、`out_proj` 和 MLP 分段诊断。SM100/SM120 会自动进行首次分段探测。
- `verbose=true` 仅输出配置、首个同形状 block、路由和采样 step 的紧凑摘要，避免每个 block 重复刷屏。

远程定位可使用：

```text
STAR7_SLA_DEBUG_BLOCK=N
STAR7_SLA_LONG_SELF_TEST=1
```

第二项会增加一次长序列内核检查，仅在排查 SM80+ 实机问题时使用。

## 示例工作流

- [通用工作流（中文）](workflows/MiniMax-H3-Activation-Chunk-Star7.json)：包含多合一条件载入、分块、实时预览、独立 H3 分块解码和默认关闭的可选二采；导入后可直接作为普通工作流使用。
- [General workflow (English)](workflows/MiniMax-H3-Activation-Chunk-Star7-English.json)：完整翻译的英文画布与说明版本，功能和默认设置与中文版一致。

## License

Star7 项目代码使用 [MIT License](LICENSE)。内置的 NVIDIA Sol-Attn 源码按其 [Apache 2.0 License](vendor/LICENSE.NVIDIA-Sana-Apache-2.0) 与 [第三方声明](vendor/sol_attn/THIRD_PARTY_NOTICES.md) 分发。H3 AdaLN 曲线网格及适配机制的来源与许可见 [Larryvrh H3 Turbo 声明](vendor/LARRYVRH-H3-TURBO-NOTICE.md)。

### VEDA 与实时预览

VEDA 节点顶部的「启用 VEDA」开关关闭时直接透传模型，不加载预测器、不安装补丁。参数使用中文名称，旧工作流无需补填开关。显卡自动分流：SM75 使用 Star7 CUDA 内核，SM80 及以上使用上游 Triton 内核；普通 H3 采样与 Star7 分块 CK 链路均可连接。路由、耗时及诊断信息只输出到日志。

内置 `Star7VedaSparseAttention`，无需单独安装 VEDA 插件。模型链路：LoRA / Sigma Shift → VEDA → Star7 分块（CK）→ 实时预览 → Guider。SM75 使用内置 INT8 QK / FP16 PV Tensor Core 稀疏内核，直接读取 VEDA 选中的块，支持填充位遮罩；CK 负责未进入稀疏计算的注意力，分块继续负责 QKV / RoPE / MLP。其他稀疏注意力选择不要与 VEDA 叠加。SM75 尚属测试实现，完整工作流速度以实测为准。上游 VEDA 的 MIT 许可证及来源见 `vendor/veda/LICENSE` 和 `NOTICE.md`。

实时预览长边支持 256 / 384 / 512 / 768 / 1024，并可调动画 WebP 质量 1–100。预览使用 TAEH3 原始 latent 分辨率并保留连续时间状态，末端 RGB 按 FPS 选择帧，再缩小输出图像；不再缩小 latent 或逐帧重置时间状态。提高 FPS 会增加末端解码计算量；预览不改变最终输出分辨率。旧工作流未保存质量值时保留 76，新节点默认 80。

实时预览按每秒帧数（1–24，默认 5）覆盖整段视频：10 秒约 50 帧。TAEH3 时间状态连续更新，但未选中的帧跳过最后的空间上采样/RGB 卷积、搬运与编码；保留帧与完整解码对应帧一致。预览长边在 RGB 解码后缩放，不降低时间状态部分的解码分辨率；压缩质量只影响编码。

2.18.1 的 Windows x64 VEDA 内核使用独立 CUDA DLL，不再绑定固定 Python / PyTorch C++ ABI，无需复制 Python 的 `include`、`libs` 或自行编译。CUDA 与 C++ 运行时已静态链接；需要支持 CUDA 13 的 NVIDIA 驱动和可用的 CUDA PyTorch 环境。更新节点后重启 ComfyUI，DLL 与校验清单需一并更新。SM80+ 保留原 Triton 路线。已通过 RTX 2080 Ti 内核数值及普通 / CK 调用测试；RTX 2060 尚需实机验证，未提供 VEDA SM75 Linux 预编译库。
