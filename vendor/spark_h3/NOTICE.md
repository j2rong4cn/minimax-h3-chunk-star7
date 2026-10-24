Spark-H3 by zechengtang, Apache-2.0.
Source: https://github.com/zechengtang/Spark-H3
Commit: 36e78b6f15a6d4efef2cbfa383d73a54eb3ba6ed

Local changes: include only ComfyUI Spark classes and required landmark helpers; omit unrelated Sol/cache/DMAD nodes and replace the helper package initializer with a lazy, empty initializer. No CUDA binary, architecture claim, or dependency replacement is included. See LICENSE.

The prepared LMv2 planner additionally accepts an explicit FP16 dtype (default
BF16 behavior is unchanged), and propagates OOM/cancellation from graph capture.
The separate Star7 SM75 port uses this FP16 planner with its own CUDA binary;
it is not an upstream-supported SM75 build.

The FP16 SM75 cosine-score path delegates to Star7's separate cuBLAS Tensor
Core scoring adapter. Other architectures and scoring modes retain the
upstream implementation. This port does not modify the original CK/SLA/Sol
backend or imply upstream validation of the SM75 implementation.
