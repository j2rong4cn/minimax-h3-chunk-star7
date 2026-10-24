// Star7 SM75 Spark global-anchor Reweight port.
// Formula: Spark-H3 (Apache-2.0), pinned in vendor/spark_h3/SOURCE.json.
// The attention consumer reuses the separately licensed Star7/Sage/CK SM75
// QK-INT8 / PV-FP16 core; existing SLA/Sol exports retain their old semantics.
#include <math_constants.h>
#include "sla_sm75_sparse.cu"
#include "spark_sm75_attention.cuh"

namespace star7_spark {
__device__ float warp_sum(float x) {
  for (int delta = 16; delta; delta >>= 1)
    x += __shfl_xor_sync(0xffffffff, x, delta);
  return x;
}

// Input BHND FP16; anchor BHD FP32. One CTA per physical 64-token leaf.
// No [parents,heads,leaves,dim] workspace: global anchor needs one summary/leaf.
__global__ void summaries(const half* k, const half* v, const float* anchor,
                          half* ak, half* av, float* lm,
                          int length, int padded) {
  const int bh = blockIdx.y, leaf = blockIdx.x, d = threadIdx.x;
  const int lane = d % 32, warp = d / 32;
  const int count = min(64, length - leaf * 64);
  const int64_t base = static_cast<int64_t>(bh) * length * 128;
  const int64_t out = (static_cast<int64_t>(bh) * padded + leaf) * 128;
  __shared__ float probability[64], reduction[128], logz;
  if (count <= 0) {
    ak[out+d] = av[out+d] = __float2half(0.f);
    if (d == 0) lm[static_cast<int64_t>(bh)*padded+leaf] = 0.f;
    return;
  }
  for (int token = warp; token < 64; token += 4) {
    float dot = 0.f;
    if (token < count) {
      for (int chunk = 0; chunk < 4; ++chunk) {
        int dim = lane + chunk*32;
        dot = fmaf(anchor[bh*128+dim],
            __half2float(k[base+(leaf*64+token)*128+dim]), dot);
      }
    }
    dot = warp_sum(dot) * 0.08838834764831845f;
    if (lane == 0) probability[token] = token < count ? dot : -CUDART_INF_F;
  }
  __syncthreads();
  if (d == 0) {
    float maximum = -CUDART_INF_F, denominator = 0.f;
    for (int t=0; t<count; ++t) maximum=fmaxf(maximum, probability[t]);
    for (int t=0; t<count; ++t) {
      probability[t]=expf(probability[t]-maximum);
      denominator+=probability[t];
    }
    logz=maximum+logf(denominator);
    for (int t=0; t<count; ++t) probability[t]/=denominator;
  }
  __syncthreads();
  float key=0.f, value=0.f;
  for (int t=0; t<count; ++t) {
    const int64_t i=base+(leaf*64+t)*128+d;
    key=fmaf(probability[t],__half2float(k[i]),key);
    value=fmaf(probability[t],__half2float(v[i]),value);
  }
  half stored=__float2half_rn(key);
  ak[out+d]=stored; av[out+d]=__float2half_rn(value);
  reduction[d]=anchor[bh*128+d]*__half2float(stored);
  __syncthreads();
  for (int step=64; step; step>>=1) {
    if (d<step) reduction[d]+=reduction[d+step];
    __syncthreads();
  }
  if (d==0) lm[static_cast<int64_t>(bh)*padded+leaf]=logz-reduction[0]*0.08838834764831845f;
}
}

STAR7_EXPORT int star7_spark_sm75_abi_version() { return 1; }
STAR7_EXPORT int star7_spark_sm75_summaries(
    uintptr_t k, uintptr_t v, uintptr_t anchor, uintptr_t ak, uintptr_t av,
    uintptr_t lm, int batch, int heads, int length, int padded, uintptr_t stream) {
  if (!k || !v || !anchor || !ak || !av || !lm || batch<1 || heads<1 ||
      length<1 || padded<(length+63)/64 || padded%64) return cudaErrorInvalidValue;
  star7_spark::summaries<<<dim3(padded,batch*heads),128,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const half*>(k),reinterpret_cast<const half*>(v),
      reinterpret_cast<const float*>(anchor),reinterpret_cast<half*>(ak),
      reinterpret_cast<half*>(av),reinterpret_cast<float*>(lm),length,padded);
  return cudaPeekAtLastError();
}

STAR7_EXPORT int star7_spark_sm75_attention(
    uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t qs, uintptr_t ks,
    uintptr_t rows, uintptr_t lut, uintptr_t ak, uintptr_t av, uintptr_t aks,
    uintptr_t mask, uintptr_t lm, uintptr_t output,
    int batch, int heads, int length, int stride, int leaves, int padded,
    float scale, uintptr_t stream) {
  if (!q || !k || !v || !qs || !ks || !rows || !lut || !ak || !av || !aks ||
      !mask || !lm || !output || batch<1 || heads<1 || length<1 || stride<1 ||
      leaves!=(length+63)/64 || padded<leaves || padded%64) return cudaErrorInvalidValue;
  auto kernel=star7_spark_attention::spark75_sm75_sparse_qk_i8_pv_f16<64,64,16,64,128,true>;
  const int shared=(64+64)*128+64*128*sizeof(half);
  cudaError_t e=configure_sol_kernel(kernel,shared);
  if (e!=cudaSuccess) return e;
  kernel<<<dim3(leaves,heads,batch),dim3(32,4),shared,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const int8_t*>(q),reinterpret_cast<const int8_t*>(k),
      reinterpret_cast<const half*>(v),reinterpret_cast<half*>(output),
      reinterpret_cast<const float*>(qs),reinterpret_cast<const float*>(ks),
      reinterpret_cast<const int32_t*>(lut),length,stride,0,
      heads*length*128,length*128,scale,reinterpret_cast<const int32_t*>(rows),stride,
      reinterpret_cast<const int8_t*>(ak),reinterpret_cast<const half*>(av),
      reinterpret_cast<const float*>(aks),reinterpret_cast<const uint8_t*>(mask),
      leaves,padded,nullptr,nullptr,nullptr,nullptr,0,nullptr,
      reinterpret_cast<const float*>(lm));
  return cudaPeekAtLastError();
}


// Fast variant is private to Spark: INT8 probability/value Tensor Cores,
// with a query-prefix launch so dense conditioning rows can use native CK.
STAR7_EXPORT int star7_spark_sm75_attention_fast(
    uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t qs, uintptr_t ks,
    uintptr_t rows, uintptr_t lut, uintptr_t ak, uintptr_t av, uintptr_t aks,
    uintptr_t mask, uintptr_t lm, uintptr_t output,
    uintptr_t vi, uintptr_t avi, uintptr_t vs, uintptr_t avs,
    int batch, int heads, int length, int stride, int leaves, int padded,
    int v_padded, int query_blocks, float scale, uintptr_t stream) {
  if (!q || !k || !v || !qs || !ks || !rows || !lut || !ak || !av || !aks ||
      !mask || !lm || !output || !vi || !avi || !vs || !avs || batch<1 || heads<1 ||
      length<1 || stride<1 || leaves!=(length+63)/64 || padded<leaves || padded%64 ||
      v_padded<length || v_padded%64 || query_blocks<1 || query_blocks>leaves)
    return cudaErrorInvalidValue;
  auto kernel=star7_spark_attention::spark75_sm75_sparse_qk_i8_pv_f16<64,64,16,64,128,true,true,true>;
  const int shared=(64+64)*128+64*128*sizeof(half);
  cudaError_t e=configure_sol_kernel(kernel,shared);
  if (e!=cudaSuccess) return e;
  kernel<<<dim3(query_blocks,heads,batch),dim3(32,4),shared,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const int8_t*>(q),reinterpret_cast<const int8_t*>(k),
      reinterpret_cast<const half*>(v),reinterpret_cast<half*>(output),
      reinterpret_cast<const float*>(qs),reinterpret_cast<const float*>(ks),
      reinterpret_cast<const int32_t*>(lut),length,stride,0,
      heads*length*128,length*128,scale,reinterpret_cast<const int32_t*>(rows),stride,
      reinterpret_cast<const int8_t*>(ak),reinterpret_cast<const half*>(av),
      reinterpret_cast<const float*>(aks),reinterpret_cast<const uint8_t*>(mask),
      leaves,padded,reinterpret_cast<const int8_t*>(vi),reinterpret_cast<const int8_t*>(avi),
      reinterpret_cast<const float*>(vs),reinterpret_cast<const float*>(avs),v_padded,nullptr,
      reinterpret_cast<const float*>(lm));
  return cudaPeekAtLastError();
}
