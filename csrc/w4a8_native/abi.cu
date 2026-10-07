// SPDX-License-Identifier: Apache-2.0
#include "w4a8.cu"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <string>

#ifdef _WIN32
#define STAR7_EXPORT extern "C" __declspec(dllexport)
#else
#define STAR7_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct W4Args {
    uint64_t activation, weight, activation_scale, group_scale, channel_scale;
    uint64_t codebook, bias, workspace, output, stream;
    int32_t rows, channels, width, group_size, workspace_rows, device;
};

static thread_local std::string last_error;
STAR7_EXPORT int star7_w4a8_abi_version() { return 1; }
STAR7_EXPORT int star7_w4a8_args_size() { return sizeof(W4Args); }
STAR7_EXPORT const char* star7_w4a8_last_error() { return last_error.c_str(); }

namespace {
struct StreamScope {
    StreamScope(int device, uint64_t stream) {
        checkCUDA(cudaSetDevice(device));
        stackCUDAStreams.push_back(reinterpret_cast<cudaStream_t>(stream));
    }
    ~StreamScope() { stackCUDAStreams.pop_back(); }
};

Tensor tensor(uint64_t pointer, Tensor::ScalarType type,
              std::initializer_list<int> shape, int device) {
    Tensor result;
    result.ptr = reinterpret_cast<void*>(pointer);
    result.scalarType = type;
    result.shape.dataExtent = shape;
    result.dev = Device{Device::CUDA, device};
    return result;
}

// Preserve the shipped FP16 row quantizer, including the rounded scale and
// FP16 division boundary. The GEMM kernel is included without modifications.
__device__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    return value;
}

__global__ void quantize_fp16(const half* input, int8_t* output, float* scales, int width) {
    __shared__ float warp_values[8];
    __shared__ float block_value;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int64_t start = static_cast<int64_t>(blockIdx.x) * width;
    float maximum = 0.0f;
    bool has_nan = false;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        float value = __half2float(input[start + col]);
        has_nan |= isnan(value);
        maximum = fmaxf(maximum, fabsf(value));
    }
    const bool row_nan = __syncthreads_or(has_nan);
    maximum = warp_max(maximum);
    if (lane == 0) warp_values[warp] = maximum;
    __syncthreads();
    if (warp == 0) {
        float total = lane < 8 ? warp_values[lane] : 0.0f;
        total = warp_max(total);
        if (lane == 0) block_value = total;
    }
    __syncthreads();
    const float scale = row_nan ? CUDART_NAN_F : fmaxf(block_value * (1.0f / 127.0f), 1.0e-30f);
    if (threadIdx.x == 0) scales[blockIdx.x] = scale;
    float math_scale = __half2float(__float2half_rn(scale));
    if (math_scale == 0.0f) math_scale = 0.00006103515625f;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const float divided = __half2float(__float2half_rn(__fdiv_rn(__half2float(input[start + col]), math_scale)));
        output[start + col] = isnan(divided) ? 0 : static_cast<int8_t>(
            fminf(127.0f, fmaxf(-128.0f, nearbyintf(divided))));
    }
}
}

STAR7_EXPORT int star7_w4a8_quantize(uint64_t input, uint64_t output, uint64_t scales,
                                  int rows, int width, int device, uint64_t stream) {
    try {
        last_error.clear();
        if (!input || !output || !scales || rows <= 0 || width <= 0)
            throw std::runtime_error("Invalid W4A8 quantization input");
        StreamScope scope(device, stream);
        quantize_fp16<<<rows, 256, 0, getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(input), reinterpret_cast<int8_t*>(output),
            reinterpret_cast<float*>(scales), width);
        checkCUDA(cudaGetLastError());
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what(); return 1;
    }
}

STAR7_EXPORT int star7_w4a8_linear(const W4Args* a) {
    try {
        last_error.clear();
        if (!a || a->rows <= 0 || a->channels <= 0 || a->width <= 0 ||
            a->width % 16 || a->channels % 8 || a->group_size < 4 ||
            a->width % a->group_size || !a->activation || !a->weight ||
            !a->activation_scale || !a->group_scale || !a->channel_scale ||
            !a->codebook || !a->output)
            throw std::runtime_error("Invalid W4A8 C ABI arguments");
        StreamScope scope(a->device, a->stream);
        const int m = a->rows, n = a->channels, k = a->width, device = a->device;
        comfyui_turing_utils::kernels::turing_codebook_w4a8_linear(
            tensor(a->activation,Tensor::INT8,{m,k},device),
            tensor(a->weight,Tensor::INT8,{n,k/2},device),
            tensor(a->activation_scale,Tensor::FP32,{m},device),
            tensor(a->group_scale,Tensor::INT8,{n,k/a->group_size},device),
            tensor(a->channel_scale,Tensor::FP32,{n},device),
            tensor(a->codebook,Tensor::FP32,{16},device),
            a->bias ? tensor(a->bias,Tensor::FP32,{n},device) : Tensor{},
            a->workspace ? tensor(a->workspace,Tensor::INT8,{a->workspace_rows,k},device) : Tensor{},
            tensor(a->output,Tensor::FP16,{m,n},device),a->group_size,a->workspace_rows==0);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what(); return 1;
    }
}
