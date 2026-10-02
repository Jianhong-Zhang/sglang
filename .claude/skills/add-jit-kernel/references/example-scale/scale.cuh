// NOTE: Comments for headers are not common in practice.
// It is only shown here for tutorial purposes to highlight the key abstractions.
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST, div_ceil

#include <sgl_kernel/type.cuh>   // For DTypeTrait, fp16_t, bf16_t, fp32_t
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace sglang {

/**
 * \brief Element-wise scale using vectorized 128-bit loads/stores.
 *
 * \tparam T       Element type: fp16_t | bf16_t | fp32_t
 * \tparam kVecN   Elements per vector load (e.g. 8 for fp16)
 * \tparam kUsePDL Whether to emit the PDL wait/trigger pair
 * \param dst      Output buffer, `n_total` elements
 * \param src      Input buffer, `n_total` elements
 * \param factor   Runtime scale factor
 * \param n_total  Number of elements to scale
 */
template <typename T, int kVecN, bool kUsePDL>
__global__ void scale_kernel(T* __restrict__ dst, const T* __restrict__ src, float factor, uint32_t n_total) {
  using vec_t = device::AlignedVector<T, kVecN>;
  const uint32_t n_vecs = n_total / kVecN;

  // If using PDL, wait for primary kernel before any global memory load.
  // This is NOT a synchronization point, which means some threads can early exit before this.
  device::PDLWaitPrimary<kUsePDL>();

  // --- vectorised body ---
  const uint32_t vec_stride = blockDim.x * gridDim.x;
  for (uint32_t vi = blockIdx.x * blockDim.x + threadIdx.x; vi < n_vecs; vi += vec_stride) {
    vec_t v;
    v.load(src, vi);
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      v[i] = static_cast<T>(static_cast<float>(v[i]) * factor);
    }
    v.store(dst, vi);
  }

  // --- scalar tail ---
  const uint32_t base = n_vecs * kVecN;
  const uint32_t scalar_stride = blockDim.x * gridDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; base + i < n_total; i += scalar_stride) {
    dst[base + i] = static_cast<T>(static_cast<float>(src[base + i]) * factor);
  }

  // If using PDL, signal for the secondary kernel to start after all threads have finished
  // This is NOT a synchronization point, which means some threads can early exit before this.
  device::PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Validate the tensors, select the vector width, launch `scale_kernel`.
 *
 * \tparam T       Element type: fp16_t | bf16_t | fp32_t
 * \tparam kUsePDL Whether to launch with PDL enabled
 * \param dst      Output tensor; same shape / dtype / device as `src`
 * \param src      Input tensor on CUDA
 * \param factor   Runtime scale factor
 */
template <typename T, bool kUsePDL>
void scale(tvm::ffi::TensorView dst, tvm::ffi::TensorView src, float factor) {
  using namespace host;

  // 1. Validate input tensors with TensorMatcher
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N})  //
      .with_dtype<T>()
      .with_device<kDLCUDA>(device_)
      .verify(dst)
      .verify(src);  // same shape / dtype / device as dst

  const uint32_t n = static_cast<uint32_t>(N.unwrap());
  const DLDevice device = device_.unwrap();

  CHECK_HOST(n > 0) << "scale: num_elements must be > 0, got " << n;

  // 2. Choose vector width for 128-bit loads (16 bytes)
  //    fp16/bf16: 8 elements x 2 bytes = 16 bytes
  //    fp32:      4 elements x 4 bytes = 16 bytes
  // We encourage using `device::kMaxVecBytes`, which will change according to
  // the target architecture and can enable 256-bit vectorization on SM100+ if desired.
  // But 128-bit is more commonly adapted for better compatibility,
  // so it's still ok to hardcode 16 here just for simplicity.
  constexpr int kVecN = 16 / sizeof(T);
  const uint32_t n_work_items = div_ceil(n, static_cast<uint32_t>(kVecN));

  // 3. Launch
  constexpr uint32_t kBlockSize = 256;
  const uint32_t grid = div_ceil(n_work_items, kBlockSize);

  // PDL feature is 100% optional. Without `enable_pdl`, the code should still be correct.
  // Try to enable it if profiling shows that it can benefit the performance of this kernel.
  LaunchKernel(grid, kBlockSize, device)
      .enable_pdl(kUsePDL)(
          scale_kernel<T, kVecN, kUsePDL>,
          static_cast<T*>(dst.data_ptr()),
          static_cast<const T*>(src.data_ptr()),
          factor,
          n);
}

}  // namespace sglang
