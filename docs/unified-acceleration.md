# MiniMax H3 Spark / VPack 加速

这是 Star7 Chunk 的配套节点，只管理 Spark 注意力和 VPack 参考视频压缩。原节点负责 QKV、MLP、输出投影分块、精度保护及解码，加速节点不提供第二套分块设置。

## 接线

将 Star7 Chunk 的 MODEL 输出接到原条件载入流程，再把条件载入输出的 MODEL、positive 接入加速节点；加速节点的 MODEL、positive 接采样流程。需要二采或修脸继承时，将条件载入的 h3_context 同时经过加速节点，再连接 HD / Face 节点。

保存的工作流仍使用历史类 ID，端口顺序保持不变。旧版本的 13 项设置会迁移成以下 5 项，已有连线无需删除重接。

| 参数 | 用途 |
| --- | --- |
| 启用加速 | 独立开关；关闭时直接传递输入 MODEL、positive、h3_context，不加载加速后端 |
| 加速模式 | off、spark、spark+vpack、vpack-only |
| Spark Top-K 比例 | 保留完整计算的视频块比例，默认 0.2 |
| Spark 预热比例 | 按实际采样步数计算，默认 0.2 |
| VPack 压缩比例 | 压缩参考视频的时间维度，默认 2/5；没有参考视频时跳过 |

Spark 必须接在 Star7 Chunk 后，使用其原有 QKV 生产路径及投影分块。选择 Spark 后，该分支由 Spark 负责注意力，密集预热使用 CK；不会在同一注意力上重复叠加 SLA / Sol / VEDA。off 和 vpack-only 保持输入模型的注意力选择。二采、修脸显式选择其他注意力时，以该分支的选择为准。

## 硬件与验证范围

SM75 使用独立编译的 FP16 Spark DLL，其他受支持显卡使用匹配的原生 Spark 后端；不支持的组合明确报错，不假装加速成功。SM75 重排走普通 CUDA / Triton 调用，禁用引发 cudaMallocAsync 闪退的 CUDA Graph 捕获。原有 CK / SLA / Sol 源码及 DLL不由此加速节点替换。

SM75 的评分保持 FP32 归一化，以 FP16 操作数和 FP32 输出使用 cuBLAS Tensor Core；稀疏消费阶段使用 INT8 PV，Reweight 加权摘要保留 FP16 PV，条件尾部查询交给原生 CK。这些操作只在 Spark 分支内执行。模式为 spark 时，VPack 比例不参与运算；spark+vpack 也只有在存在参考视频时才有压缩收益。Top-K 是计算预算而非整段耗时比例，重排、投影、MLP、预览与解码仍然需要时间。

本机 Windows SM75 已进行 CUDA 数值、小型 H3 联合测试，以及同种子、同条件的完整视频对比（896×1184、124 帧、4 步，Spark Top-K 0.1、预热 0）。本次纯 CK、Spark、CK＋VEDA 的任务总耗时分别约 240、217、200 秒；Spark 后续三步的中位耗时比纯 CK 低约 16.5%，但没有超过原 CK＋VEDA。总耗时包含加载、首次初始化、预览与导出，不能当作单独注意力耗时。此结果只适用于本次配置；抽帧对比可见近似计算改变了背景与运镜，不能视为画质无损。Linux 构建脚本保留，但尚未在 Linux GPU 上验证。Spark 和 VPack 均是近似计算，应使用相同种子和条件进行完整视频对比。
