#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "Norm.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "cutlass/float8.h"

// TODO: Remove this when sycl float8 is supported
using cutlass::float_e4m3_t;

namespace at::native::xpu {

template <typename T>
inline T divUp(T m, T n) {
  return (m + n - 1) / n;
}

// SYCL Kernel for fused QK RMSNorm.
template <int head_dim, typename scalar_t>
struct FusedQKNormKernel {
  scalar_t* qkv;
  int num_heads_q;
  int num_heads_k;
  int num_heads_v;
  float eps;
  const scalar_t* q_weight;
  const scalar_t* k_weight;
  int num_tokens;

  void operator()(sycl::nd_item<1> item) const {
    using accscalar_t = float;

    const int sg_size = item.get_sub_group().get_max_local_range()[0];
    const int warpsPerBlock = item.get_local_range(0) / sg_size;
    const int warpId = item.get_local_id(0) / sg_size;
    const int laneId = item.get_local_id(0) % sg_size;

    const int globalWarpIdx = item.get_group(0) * warpsPerBlock + warpId;
    const int total_qk_heads = num_heads_q + num_heads_k;

    const int tokenIdx = globalWarpIdx / total_qk_heads;
    const int localHeadIdx = globalWarpIdx % total_qk_heads;

    if (tokenIdx >= num_tokens) {
      return;
    }

    const bool isQ = localHeadIdx < num_heads_q;
    const int headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;
    const int num_heads = num_heads_q + num_heads_k + num_heads_v;

    constexpr int numElemsPerThread = head_dim / 32;
    accscalar_t elements[numElemsPerThread];

    const int offsetWarp = isQ ? tokenIdx * num_heads * head_dim + headIdx * head_dim
                               : tokenIdx * num_heads * head_dim + num_heads_q * head_dim + headIdx * head_dim;
    const int offsetThread = offsetWarp + laneId * numElemsPerThread;

    accscalar_t sumOfSquares = 0.0f;
    for (int i = 0; i < numElemsPerThread; i++) {
      elements[i] = static_cast<accscalar_t>(qkv[offsetThread + i]);
      sumOfSquares += elements[i] * elements[i];
    }

    auto sg = item.get_sub_group();
    sumOfSquares = sycl::reduce_over_group(sg, sumOfSquares, sycl::plus<accscalar_t>{});

    const float rms_rcp = sycl::rsqrt(sumOfSquares / static_cast<float>(head_dim) + eps);
    const scalar_t* weight_ptr = isQ ? q_weight : k_weight;

    for (int i = 0; i < numElemsPerThread; i++) {
      const int dim = laneId * numElemsPerThread + i;
      const accscalar_t weight = static_cast<accscalar_t>(weight_ptr[dim]);
      elements[i] *= rms_rcp * weight;
    }

    for (int i = 0; i < numElemsPerThread; i++) {
      qkv[offsetThread + i] = static_cast<scalar_t>(elements[i]);
    }
  }
};

template <int head_dim, typename scalar_t>
void launchFusedQKNormImpl(
    void* qkv,
    int num_tokens,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    float eps,
    const void* q_weight,
    const void* k_weight,
    sycl::queue& q) {
  constexpr int blockSize = 256;
  const int warpsPerBlock = blockSize / 32;
  const int totalQKHeads = num_heads_q + num_heads_k;
  const int totalWarps = num_tokens * totalQKHeads;
  const int gridSize = divUp(totalWarps, warpsPerBlock);

  FusedQKNormKernel<head_dim, scalar_t> kernel{
      static_cast<scalar_t*>(qkv),
      num_heads_q,
      num_heads_k,
      num_heads_v,
      eps,
      static_cast<const scalar_t*>(q_weight),
      static_cast<const scalar_t*>(k_weight),
      num_tokens};

  sycl_kernel_submit(sycl::range<1>(gridSize * blockSize), sycl::range<1>(blockSize), q, kernel);
}

void fused_qk_norm(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight) {
  TORCH_CHECK(qkv.dim() == 2, "QKV tensor must be 2D: [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]");
  TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
  TORCH_CHECK(q_weight.size(0) == head_dim, "Query weights size must match head dimension");
  TORCH_CHECK(k_weight.size(0) == head_dim, "Key weights size must match head dimension");
  TORCH_CHECK(head_dim % 32 == 0, "head_dim must be divisible by 32");

  CHECK_DEVICE(qkv);
  CHECK_CONTIGUOUS(qkv);
  CHECK_DEVICE(q_weight);
  CHECK_CONTIGUOUS(q_weight);
  CHECK_DEVICE(k_weight);
  CHECK_CONTIGUOUS(k_weight);

  TORCH_CHECK(qkv.scalar_type() == q_weight.scalar_type(), "qkv and q_weight must have the same dtype");
  TORCH_CHECK(qkv.scalar_type() == k_weight.scalar_type(), "qkv and k_weight must have the same dtype");

  const int64_t num_tokens = qkv.size(0);
  const int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total number of heads and head dimension");

  auto queue = dpcppGetCurrentQueue();

// Maps at::ScalarType to the corresponding SYCL/cutlass C++ type (scalar_t)
// and immediately invokes the callable passed as the variadic argument.
// Float8_e4m3fn uses the cutlass type because native SYCL float8 is not yet
// supported; Half and BFloat16 use their respective SYCL types.
#define SYCL_DISPATCH_FLOATING_TYPES(SCALAR_TYPE, KERNEL_NAME, ...)                 \
  [&]() {                                                                           \
    switch (SCALAR_TYPE) {                                                          \
      case at::ScalarType::Half: {                                                  \
        using scalar_t = sycl::half;                                                \
        __VA_ARGS__();                                                              \
        break;                                                                      \
      }                                                                             \
      case at::ScalarType::BFloat16: {                                              \
        using scalar_t = sycl::ext::oneapi::bfloat16;                               \
        __VA_ARGS__();                                                              \
        break;                                                                      \
      }                                                                             \
      case at::ScalarType::Float8_e4m3fn: {                                         \
        using scalar_t = float_e4m3_t;                                              \
        __VA_ARGS__();                                                              \
        break;                                                                      \
      }                                                                             \
      default:                                                                      \
        TORCH_CHECK(false, "Unsupported dtype for " KERNEL_NAME ": ", SCALAR_TYPE); \
    }                                                                               \
  }()

#define LAUNCH_KERNEL(HD)                                                           \
  SYCL_DISPATCH_FLOATING_TYPES(qkv.scalar_type(), "fused_qk_norm", ([&]() {       \
                                 launchFusedQKNormImpl<HD, scalar_t>(              \
                                     qkv.data_ptr(),                               \
                                     static_cast<int>(num_tokens),                 \
                                     static_cast<int>(num_heads_q),                \
                                     static_cast<int>(num_heads_k),                \
                                     static_cast<int>(num_heads_v),                \
                                     static_cast<float>(eps),                      \
                                     q_weight.data_ptr(),                          \
                                     k_weight.data_ptr(),                          \
                                     queue);                                       \
                               }))

  switch (head_dim) {
    case 64:
      LAUNCH_KERNEL(64);
      break;
    case 128:
      LAUNCH_KERNEL(128);
      break;
    case 256:
      LAUNCH_KERNEL(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head dimension for fusedQKNorm: ", head_dim);
  }

#undef LAUNCH_KERNEL
#undef SYCL_DISPATCH_FLOATING_TYPES
}

}  // namespace at::native::xpu
