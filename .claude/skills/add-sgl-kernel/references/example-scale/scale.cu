#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "utils.h"  // DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16

// scale_kernel: out[i] = input[i] * factor
// Supports float, half (__half), __nv_bfloat16 via template T
template <typename T>
__global__ void scale_kernel(T* __restrict__ out, const T* __restrict__ input, float factor, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = static_cast<T>(static_cast<float>(input[idx]) * factor);
  }
}

void scale(at::Tensor& out, const at::Tensor& input, double factor) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.sizes() == input.sizes(), "out and input must have the same shape");
  TORCH_CHECK(out.scalar_type() == input.scalar_type(), "out and input must have the same dtype");

  const int64_t n = input.numel();
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  // Dispatches over float, float16, bfloat16
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    scale_kernel<c_type><<<blocks, threads, 0, stream>>>(
        static_cast<c_type*>(out.data_ptr()),
        static_cast<const c_type*>(input.data_ptr()),
        static_cast<float>(factor),
        n);
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, "scale_kernel launch failed: ", cudaGetErrorString(status));
    return true;
  });
}
