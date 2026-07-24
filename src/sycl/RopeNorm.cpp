#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

namespace {

using bf16_t = sycl::ext::oneapi::bfloat16;  // TODO: ref to RMSNorm.cpp, support mant dtypes
                                             // TODO: 这里所有与硬件有关的 size 都不该直接指定
constexpr int kBlockSize = 128;

// 真正实现的地方
// vec_size: 每个线程一次批量读/写的元素个数(load/store 向量宽度)。
// 计算逻辑始终是标量的(逐分量算),只有内存访问是向量化的，
// 与 RMSNorm.cpp 里 RMSNormNoRstdForward::update() 的模式一致。
//
// rope 部分要求 vec_size == 1 或 vec_size 为偶数：因为 rope 区域里数据是
// (real, imag) 相邻成对存储的，vec_size 为偶数时每个 vec_t chunk 恰好装
// 整数个 pair，可以在同一个 chunk 内部按 (v, v+1) 配对处理；调用方还需要
// 保证 nope_dim % vec_size == 0 且 rope_dim % vec_size == 0，否则向量化
// 读写会跨出 nope/rope 各自的边界。这些检查放在外层 TORCH_CHECK 里，不在
// kernel 内部检查。
template <int vec_size>
struct FusedQNormRopeKernel {
  using vec_t = aligned_vector_loop<bf16_t, vec_size>;
  using freq_vec_t = aligned_vector_loop<float, vec_size>;

  const bf16_t* q_input;
  bf16_t* q_output;
  const float* freqs_cis;
  const int32_t* positions;
  int64_t q_input_stride_batch;
  int64_t q_output_stride_batch;
  int64_t batch_size;
  int64_t num_q_heads;
  int64_t head_dim;
  int64_t rope_dim;
  int64_t nope_dim;
  float eps;
  sycl::local_accessor<float, 1> local_sum;

  void operator()(sycl::nd_item<1> item) const {
    const int lid = static_cast<int>(item.get_local_id(0));
    const int row = static_cast<int>(item.get_group(0));
    const int total_rows = static_cast<int>(batch_size * num_q_heads);
    if (row >= total_rows) {
      return;
    }

    const int batch_id = row / static_cast<int>(num_q_heads);
    const int head_id = row % static_cast<int>(num_q_heads);

    const bf16_t* in_ptr =
        q_input + static_cast<int64_t>(batch_id) * q_input_stride_batch + static_cast<int64_t>(head_id) * head_dim;
    bf16_t* out_ptr =
        q_output + static_cast<int64_t>(batch_id) * q_output_stride_batch + static_cast<int64_t>(head_id) * head_dim;

    // reduce: 批量读 head_dim, 标量累加平方和
    float thread_sum = 0.0f;
    for (int d = lid * vec_size; d < head_dim; d += kBlockSize * vec_size) {
      vec_t val = *(reinterpret_cast<const vec_t*>(in_ptr + d));
#pragma unroll
      for (int v = 0; v < vec_size; ++v) {
        const float x = static_cast<float>(val[v]);
        thread_sum += x * x;
      }
    }

    local_sum[lid] = thread_sum;
    item.barrier(sycl::access::fence_space::local_space);

    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        local_sum[lid] += local_sum[lid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    const float norm_factor = sycl::rsqrt(local_sum[0] / static_cast<float>(head_dim) + eps);

    // nope: 批量读/写, 标量算 y = x * norm_factor
    for (int d = lid * vec_size; d < nope_dim; d += kBlockSize * vec_size) {
      vec_t val = *(reinterpret_cast<const vec_t*>(in_ptr + d));
      vec_t out_val;
#pragma unroll
      for (int v = 0; v < vec_size; ++v) {
        out_val[v] = static_cast<bf16_t>(static_cast<float>(val[v]) * norm_factor);
      }
      *(reinterpret_cast<vec_t*>(out_ptr + d)) = out_val;
    }

    // rope: 批量读 q 和 freq, 标量按 (real, imag) 配对计算旋转
    const int32_t pos = positions[batch_id];
    const float* freq_ptr = freqs_cis + static_cast<int64_t>(pos) * rope_dim;

    if constexpr (vec_size == 1) {
      // vec_size == 1 时一个 vec_t chunk 只装 1 个元素，无法在同一个 chunk
      // 里凑出一对 (real, imag)，因此不走 vec_t 路径，直接按 pair 索引 p
      // 标量读写，等价于最初未向量化的实现。
      const int rope_pairs = static_cast<int>(rope_dim / 2);
      for (int p = lid; p < rope_pairs; p += kBlockSize) {
        const int rope_offset = nope_dim + 2 * p;
        const float xr = static_cast<float>(in_ptr[rope_offset]);
        const float xi = static_cast<float>(in_ptr[rope_offset + 1]);
        const float fr = freq_ptr[2 * p];
        const float fi = freq_ptr[2 * p + 1];

        const float rot_r = xr * fr - xi * fi;
        const float rot_i = xr * fi + xi * fr;

        out_ptr[rope_offset] = static_cast<bf16_t>(rot_r * norm_factor);
        out_ptr[rope_offset + 1] = static_cast<bf16_t>(rot_i * norm_factor);
      }
    } else {
      // vec_size >= 2 (偶数): 每个 vec_t chunk 恰好装整数个 pair，
      // 在 chunk 内部按 (v, v+1) 配对计算。
      for (int d = lid * vec_size; d < rope_dim; d += kBlockSize * vec_size) {
        vec_t q_val = *(reinterpret_cast<const vec_t*>(in_ptr + nope_dim + d));
        freq_vec_t f_val = *(reinterpret_cast<const freq_vec_t*>(freq_ptr + d));
        vec_t out_val;
#pragma unroll
        for (int v = 0; v < vec_size; v += 2) {
          const float xr = static_cast<float>(q_val[v]);
          const float xi = static_cast<float>(q_val[v + 1]);
          const float fr = f_val[v];
          const float fi = f_val[v + 1];

          const float rot_r = xr * fr - xi * fi;
          const float rot_i = xr * fi + xi * fr;

          out_val[v] = static_cast<bf16_t>(rot_r * norm_factor);
          out_val[v + 1] = static_cast<bf16_t>(rot_i * norm_factor);
        }
        *(reinterpret_cast<vec_t*>(out_ptr + nope_dim + d)) = out_val;
      }
    }
  }
};

// VEC_LAUNCH_ROPE(N): 对应 TripleOps.cpp 里的 VEC_LAUNCH 宏，把"构造 kernel
// 对象 + submit"直接内联在 switch 分支里，不再额外包一层 fused_xxx_kernel<N>
// 函数。与 TripleOps.cpp 的 VEC_LAUNCH 不同的是：这里的 kernel 需要
// local_accessor，而 local_accessor 必须绑定 cgh 构造，所以不能像
// TripleOps.cpp 那样直接调用 sycl_kernel_submit(把已经构造好的、不依赖 cgh
// 的 kernel 对象传进去)，而是要自己写 cgf + queue.submit(cgf)。
#define VEC_LAUNCH_ROPE(N)                                                                      \
  case N: {                                                                                     \
    auto cgf = [&](sycl::handler& cgh) {                                                        \
      sycl::local_accessor<float, 1> local_sum(kBlockSize, cgh);                                \
      FusedQNormRopeKernel<N> kernel{                                                           \
          q_input_ptr,                                                                          \
          q_output_ptr,                                                                         \
          freqs_cis_ptr,                                                                        \
          positions_ptr,                                                                        \
          q_input_stride_batch,                                                                 \
          q_output_stride_batch,                                                                \
          B,                                                                                    \
          H,                                                                                    \
          D,                                                                                    \
          rope_dim,                                                                             \
          nope_dim,                                                                             \
          eps,                                                                                  \
          local_sum};                                                                           \
      cgh.parallel_for<decltype(kernel)>(sycl::nd_range<1>(global_range, local_range), kernel); \
    };                                                                                          \
    queue.submit(cgf);                                                                          \
    break;                                                                                      \
  }

// 只做两件事：(1) 根据 nope_dim/rope_dim 能否整除算出 vec_size(简单版本，不
// 检查指针对齐)；(2) switch 到对应的编译期模板实例并 submit。不做参数校验。
// TODO: 对齐 RMSNorm 的调度设计，后续改为通过 NormConfig 推导
// update_vec_size/workgroup_size，而不是这里的简化规则和固定 kBlockSize。
void launch_vectorized_fused_q_norm_rope_kernel(
    const bf16_t* q_input_ptr,
    bf16_t* q_output_ptr,
    const float* freqs_cis_ptr,
    const int32_t* positions_ptr,
    int64_t q_input_stride_batch,
    int64_t q_output_stride_batch,
    int64_t B,
    int64_t H,
    int64_t D,
    int64_t rope_dim,
    int64_t nope_dim,
    float eps,
    sycl::queue& queue) {
  int vec_size = 4;

  // Fallback to smaller vector width if any participating pointer is not
  // sufficiently aligned for reinterpret_cast<vec_t*> loads/stores.
  auto aligned_for_vec = [&](int vs) {
    const auto q_in_addr = reinterpret_cast<uintptr_t>(q_input_ptr);
    const auto q_out_addr = reinterpret_cast<uintptr_t>(q_output_ptr);
    const auto freq_addr = reinterpret_cast<uintptr_t>(freqs_cis_ptr);
    const uintptr_t q_align = static_cast<uintptr_t>(vs * static_cast<int>(sizeof(bf16_t)));
    const uintptr_t f_align = static_cast<uintptr_t>(vs * static_cast<int>(sizeof(float)));
    return (q_in_addr % q_align == 0) && (q_out_addr % q_align == 0) && (freq_addr % f_align == 0);
  };

  while (vec_size > 1 && (nope_dim % vec_size != 0 || rope_dim % vec_size != 0)) {
    vec_size >>= 1;
  }
  while (vec_size > 1 && !aligned_for_vec(vec_size)) {
    vec_size >>= 1;
  }

  const int64_t total_rows = B * H;
  sycl::range<1> local_range(kBlockSize);
  sycl::range<1> global_range(total_rows * kBlockSize);

  switch (vec_size) {
    VEC_LAUNCH_ROPE(4);
    VEC_LAUNCH_ROPE(2);
    default:
      VEC_LAUNCH_ROPE(1);
  }
}
#undef VEC_LAUNCH_ROPE

}  // namespace

// .Internal 层：只负责从 Tensor 里取指针 + 拿 queue，不做 TORCH_CHECK，
// 对应 RMSNorm.cpp 里 RMSNormKernelImplInternal 的角色(区别是这里没有多
// dtype 分发的需求，所以不需要是模板函数)。
static void fused_q_norm_rope_kernel_internal(
    const torch::Tensor& q_input,
    torch::Tensor& q_output,
    const torch::Tensor& freqs_cis,
    const torch::Tensor& positions,
    int64_t B,
    int64_t H,
    int64_t D,
    int64_t rope_dim,
    int64_t nope_dim,
    double eps) {
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  launch_vectorized_fused_q_norm_rope_kernel(
      static_cast<const bf16_t*>(q_input.data_ptr()),
      static_cast<bf16_t*>(q_output.data_ptr()),
      static_cast<const float*>(freqs_cis.data_ptr()),
      static_cast<const int32_t*>(positions.data_ptr()),
      q_input.stride(0),
      q_output.stride(0),
      B,
      H,
      D,
      rope_dim,
      nope_dim,
      static_cast<float>(eps),
      queue);
}

// 命名不要加  前缀
void fused_q_norm_rope(
    const torch::Tensor& q_input,
    torch::Tensor& q_output,
    const torch::Tensor& freqs_cis,
    const torch::Tensor& positions,
    double eps) {
  TORCH_CHECK(q_input.is_xpu(), "q_input must be an XPU tensor");
  TORCH_CHECK(q_output.is_xpu(), "q_output must be an XPU tensor");
  TORCH_CHECK(freqs_cis.is_xpu(), "freqs_cis must be an XPU tensor");
  TORCH_CHECK(positions.is_xpu(), "positions must be an XPU tensor");

  TORCH_CHECK(
      q_input.scalar_type() == at::ScalarType::BFloat16, "q_input must be bfloat16");  // bf16 的类型限制还要跟 cao 讨论
  TORCH_CHECK(q_output.scalar_type() == at::ScalarType::BFloat16, "q_output must be bfloat16");
  TORCH_CHECK(freqs_cis.scalar_type() == at::ScalarType::Float, "freqs_cis must be float32");
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Int, "positions must be int32");

  TORCH_CHECK(q_input.dim() == 3, "q_input must be 3D: (B, H, D)");
  TORCH_CHECK(q_output.dim() == 3, "q_output must be 3D: (B, H, D)");
  TORCH_CHECK(freqs_cis.dim() == 2, "freqs_cis must be 2D: (max_pos, rope_dim)");
  TORCH_CHECK(positions.dim() == 1, "positions must be 1D: (B)");

  const int64_t B = q_input.size(0);
  const int64_t H = q_input.size(1);
  const int64_t D = q_input.size(2);

  TORCH_CHECK(
      q_output.size(0) == B && q_output.size(1) == H && q_output.size(2) == D, "q_output shape must match q_input");

  TORCH_CHECK(positions.size(0) == B, "positions size must equal batch size");
  TORCH_CHECK(q_input.stride(2) == 1 && q_output.stride(2) == 1, "last dim must be contiguous");
  TORCH_CHECK(q_input.stride(1) == D && q_output.stride(1) == D, "head dim must be contiguous");

  const int64_t rope_dim = freqs_cis.size(1);
  const int64_t max_pos = freqs_cis.size(0);
  TORCH_CHECK(rope_dim > 0 && rope_dim % 2 == 0, "rope_dim must be positive and even");
  TORCH_CHECK(rope_dim <= D, "rope_dim must be <= head_dim");
  TORCH_CHECK(max_pos > 0, "freqs_cis first dimension must be > 0");

  // Keep aligned with CUDA AOT behavior for this op.
  TORCH_CHECK(
      D == 128 || D == 192,
      "Unsupported head_dim for fused_q_norm_rope: ",
      D);  // 这点需要报告 Cao，在 CUDA 实现里这里也是要求 128 和 192，但是 ds 官方只给了 head_dim = 512

  const int64_t nope_dim = D - rope_dim;

  if (B == 0 || H == 0) {
    return;
  }

  CHECK_CONTIGUOUS(freqs_cis);
  CHECK_CONTIGUOUS(positions);

  auto pos_min = positions.min().item<int32_t>();
  auto pos_max = positions.max().item<int32_t>();
  TORCH_CHECK(pos_min >= 0, "positions must be non-negative");
  TORCH_CHECK(static_cast<int64_t>(pos_max) < max_pos, "positions contain index out of range for freqs_cis");

  fused_q_norm_rope_kernel_internal(q_input, q_output, freqs_cis, positions, B, H, D, rope_dim, nope_dim, eps);
  // 分发的时候，会出现有一部分有 rope，有一部分没有 rope 的情况吗，因为只有 B,H,D 的最后 rope_dim 个维度才会做
  // rope，其他的维度不做 rope，这个时候就需要在 kernel 里判断是否需要做 rope
}

}  // namespace at::native::xpu
