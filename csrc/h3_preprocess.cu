// SPDX-License-Identifier: Apache-2.0
#include "third_party/comfy_kitchen_sage/quant_qk_int8.cu"
#include <cuda_fp16.h>

#if defined(_WIN32)
#define STAR7_EXPORT extern "C" __declspec(dllexport)
#else
#define STAR7_EXPORT extern "C" __attribute__((visibility("default")))
#endif

STAR7_EXPORT int star7_h3_preprocess_abi() { return 1; }

STAR7_EXPORT int star7_h3_k_anchor(const void* samples, void* indices, int heads, uint64_t stream) {
    detect_k_anchor<half><<<dim3(heads, 1), 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const half*>(samples), static_cast<int*>(indices), 9, 128, heads,
        heads * 9 * 128, 9 * 128, 128);
    return static_cast<int>(cudaGetLastError());
}

__global__ void quant_k_external(const half* __restrict__ key,
                                const half* __restrict__ anchor,
                                int8_t* __restrict__ output,
                                float* __restrict__ scales,
                                int heads, int length) {
    const int head = blockIdx.y;
    process_k<half, 16, 128, 1, 128, true>(key + head * length * 128,
        output + head * length * 128, scales + head * ((length + 127) / 128) * 4,
        blockIdx.x, length, 128, -1, 128, anchor + head * 128);
}

STAR7_EXPORT int star7_h3_quant_k(const void* key, const void* anchor, void* output,
                                 void* scales, int heads, int length, uint64_t stream) {
    quant_k_external<<<dim3((length + 127) / 128, heads, 1), 128, 0,
                       reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const half*>(key), static_cast<const half*>(anchor),
        static_cast<int8_t*>(output), static_cast<float*>(scales), heads, length);
    return static_cast<int>(cudaGetLastError());
}

STAR7_EXPORT int star7_h3_quant_q(
    const void* q, void* output, void* scale, int heads, int length,
    int64_t stride_b, int64_t stride_h, int64_t stride_n, uint64_t stream) {
    quant_q_kernel<half, 4, 128, 32, 1, 128, true>
        <<<dim3((length + 127) / 128 * 4, heads, 1), 128, 0,
           reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const half*>(q), static_cast<int8_t*>(output),
            static_cast<float*>(scale), length, 128, heads,
            (length + 127) / 128 * 32, stride_b, stride_h, stride_n);
    return static_cast<int>(cudaGetLastError());
}

__global__ void swiglu_fp16(const half* __restrict__ input,
                            half* __restrict__ output,
                            int64_t count, int width)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const int64_t row = index / width;
    const int col = index - row * width;

    const half* base = input + row * (width * 2) + col;
    const float gate = __half2float(base[0]);
    const half up = base[width];

    const half silu = __float2half_rn(gate / (1.0f + expf(-gate)));
    const half inv_ksilu = __float2half(1.0f / 16.0f);
    const half inv_kup = __float2half(1.0f / 8.0f);
    const half s = __hmul(silu, inv_ksilu);
    const half a = __hmul(up, inv_kup);
    output[index] = __hmul(s, a);
}

STAR7_EXPORT int star7_h3_swiglu_fp16(const void* input, void* output,
                                    int rows, int width, uint64_t stream) {
    const int64_t count = static_cast<int64_t>(rows) * width;
    swiglu_fp16<<<(count + 255) / 256, 256, 0,
                   reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const half*>(input), static_cast<half*>(output), count, width);
    return static_cast<int>(cudaGetLastError());
}
