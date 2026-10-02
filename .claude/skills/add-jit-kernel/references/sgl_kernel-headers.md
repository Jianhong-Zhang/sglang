# `sgl_kernel/` header API catalogue

The shared abstractions in `python/sglang/kernels/jit/include/sgl_kernel/`. Look up the header you need; the source is the authority when this drifts.

## Contents

- `utils.h` — host checks (`CHECK_HOST`), `div_ceil`, `irange`, pointer offset
- `utils.cuh` — type aliases, `SGL_DEVICE`, `LaunchKernel`, PDL wait/trigger
- `tensor.h` — `TensorMatcher`, `SymbolicSize` / `SymbolicDType` / `SymbolicDevice`
- `ffi.h` — `ffi::empty`, `ffi::empty_like`, `ffi::from_blob`
- `type.cuh` — `DTypeTrait`, `packed_t`, `device::cast`, reduction traits
- `vec.cuh` — `AlignedVector`
- `tile.cuh` — `tile::Memory`
- `math.cuh` — `device::math::`
- `warp.cuh` — `warp::reduce` and wrappers
- `cta.cuh` — `cta::reduce_max`
- `atomic.cuh` — `atomic::max`
- `runtime.cuh` — occupancy / SM count, persistent kernel pattern
- Real kernels that use them

### `utils.h` — Host-side utilities

```cpp
#include <sgl_kernel/utils.h>
```

- **`CHECK_HOST(cond) << "msg " << value`** — **Preferred** runtime check: stream-style, throws `PanicError` with file/line info on failure. Zero overhead on the true path — the message expressions are only evaluated when the check fails.
- **`host::RuntimeCheck(cond, args...)`** — Function-style alternative to `CHECK_HOST`. Note its message args are always evaluated (even when the check passes), so prefer `CHECK_HOST` — especially on hot paths.
- **`host::Panic(args...)`** — Unconditionally throw a `PanicError` with a descriptive message.
- **`host::div_ceil(a, b)`** — Integer ceiling division `(a + b - 1) / b`.
- **`host::irange(n)`** / **`host::irange(start, end)`** — Range views for cleaner loops.
- **`host::pointer::offset(ptr, offsets...)`** — Byte-safe pointer arithmetic on `void*`. Use this instead of raw casts.

### `utils.cuh` — Device-side utilities + `LaunchKernel`

```cpp
#include <sgl_kernel/utils.cuh>
```

- **Type aliases**: `fp16_t`, `bf16_t`, `fp32_t`, `fp8_e4m3_t`, `fp8_e5m2_t` and their packed variants `fp16x2_t`, `bf16x2_t`, `fp32x2_t`, etc.
- **`SGL_DEVICE`** — Expands to `__forceinline__ __device__`. Use on all device functions.
- **`device::kWarpThreads`** — Constant `32`.
- **`device::load_as<T>(ptr, offset)`** / **`device::store_as<T>(ptr, val, offset)`** — Type-safe loads/stores from `void*`.
- **`device::pointer::offset(ptr, offsets...)`** — Pointer arithmetic on device.
- **`host::LaunchKernel(grid, block, device_or_stream [, smem])`** — RAII kernel launcher that:
  - Resolves the CUDA stream from a `DLDevice` via TVM-FFI automatically.
  - Checks the CUDA error with file/line info after launch via `operator()(kernel, args...)`.
  - Supports `.enable_pdl(bool)` for PDL (Programmatic Dependent Launch, SM90+).
- **`device::PDLWaitPrimary<kUsePDL>()`** / **`device::PDLTriggerSecondary<kUsePDL>()`** — The two halves of PDL, on sm_90+ (no-ops on older archs and ROCm). Their guarantees are **not** symmetric:
  - `PDLTriggerSecondary` (`griddepcontrol.launch_dependents`) only lets the next kernel in the stream *start* early. It carries no memory ordering and publishes nothing — matching that, the header's asm has no `"memory"` clobber.
  - `PDLWaitPrimary` (`griddepcontrol.wait`) is the ordering point: it waits until the preceding kernel has fully finished and its writes are visible.

  So every read of data the preceding kernel produced must come after `PDLWaitPrimary()`. What overlaps with the primary's tail is whatever you put *before* the wait — loading parameters, computing indices, touching buffers the primary never wrote — so a kernel that waits on its first line gains nothing. Neither call is a barrier: threads may reach or skip them independently. See "Programmatic Dependent Launch and Synchronization" in the CUDA C++ Programming Guide.
- **`CHECK_CUDA(expr) << "context"`** — Stream-style CUDA error check; evaluates `expr` once and throws `PanicError` with `cudaGetErrorString` + file/line info if it is not `cudaSuccess`. Extra streamed context is optional.
- **`host::RuntimeDeviceCheck(cudaError_t)`** — Function-style alternative to `CHECK_CUDA`. It takes no context message, so prefer `CHECK_CUDA`, which builds its error object only on the failure path.

### `tensor.h` — Tensor validation (`TensorMatcher`, Symbolic types)

```cpp
#include <sgl_kernel/tensor.h>
```

This is the **primary validation API** for all kernel launchers. Use it to validate every `tvm::ffi::TensorView` argument.

- **`host::SymbolicSize{"name"}`** — A named symbolic dimension. Call `.set_value(n)` to pin it, `.unwrap()` to extract after verification.
- **`host::SymbolicDType`** — Symbolic dtype. Use `.set_options<Ts...>()` to restrict allowed types.
- **`host::SymbolicDevice`** — Symbolic device. Use `.set_options<kDLCUDA>()` to restrict to CUDA.
- **`host::TensorMatcher({dims...})`** — Fluent builder for tensor validation:
  - `.with_dtype<T>()` — require a specific C++ type (e.g. `fp16_t`)
  - `.with_dtype<T1, T2, ...>()` — allow a set of types
  - `.with_device<kDLCUDA>(device_sym)` — require CUDA and bind the checked device to a `SymbolicDevice`
  - `.with_strides({strides...})` — validate strides (omit to require contiguous)
  - `.verify(tensor_view)` — execute the check; throws `PanicError` with full context on failure; **chainable** (`verify(a).verify(b)` to check multiple tensors with the same shape)
- **`host::is_type<T>(dtype)`** — whether a `DLDataType` denotes the C++ type `T` (e.g. `fp16_t`).

**Typical pattern:**
```cpp
auto N = SymbolicSize{"num_elements"};
auto device = SymbolicDevice{};
device.set_options<kDLCUDA>();
TensorMatcher({N})  //
    .with_dtype<fp16_t>()
    .with_device<kDLCUDA>(device)
    .verify(dst)
    .verify(src);  // same shape, dtype, device as dst
const int64_t n = N.unwrap();
const DLDevice dev = device.unwrap();
const int64_t last_dim = 128;
TensorMatcher({N, last_dim})  // a fixed dimension can be a plain integer
    .with_dtype<fp16_t>()
    .with_device<kDLCUDA>(device)
    .verify(tensor_2d);
```

### `ffi.h` — Tensor allocation and blob wrapping (`host::ffi::`)

```cpp
#include <sgl_kernel/ffi.h>
```

The counterpart to `tensor.h`: that one validates what came in, this one produces new `tvm::ffi::Tensor` values. Allocation goes through the environment allocator (`TVMFFIEnvTensorAlloc`), so buffers come from PyTorch's caching allocator rather than a raw `cudaMalloc`.

- **`host::alloc_workspace_tensor(nbytes, device)`** (declared in `utils.cuh`) — **the way to get scratch memory**: a 1-D `uint8` tensor of `nbytes`, or an empty tensor when `nbytes == 0`. Hold the returned `Tensor` in a local across every launch that touches it — it frees on destruction.
- **`host::ffi::empty(shape, dtype, device)`** — Uninitialized tensor; `shape` accepts a braced list, so `ffi::empty({rows, sizeof(Plan)}, dtype, device)` works for a typed scratch array.
- **`host::ffi::empty_like(tensor_view)`** — Same shape, dtype, and device as an existing tensor.
- **`host::ffi::from_blob(data, shape, dtype, device[, deleter, stride, byte_offset])`** / **`from_blob_like(data, tensor_view, ...)`** — View memory you already own as a `Tensor`, no copy. The default deleter does nothing, so ownership stays with the caller; pass one only when the `Tensor` should own the block. Strides default to contiguous.

### `type.cuh` — `DTypeTrait<T>`, `packed_t<T>`, and reduction traits

```cpp
#include <sgl_kernel/type.cuh>
```

- **`DTypeTrait<T>`** — Static trait struct, specialized for integral types, `fp32_t`, `fp16_t`, `bf16_t`, `fp8_e4m3_t`, and their packed x2/x4 variants. Provides:
  - `DTypeTrait<T>::from(value)` — convert from another type via the right CUDA intrinsic (e.g. `fp32_t` → `fp16_t`)
  - `DTypeTrait<T>::abs/max/min` — type-dispatched math (fp32, fp16/bf16 scalar and x2, integrals)
  - `DTypeTrait<T>::sqrt/rsqrt/exp/sin/cos(x)` — `fp32_t` only
  - Metadata: `packed_t` / `unpacked_t` / `kVecSize` (packed layout), `kFloatMax` (dtype max as float, e.g. 448.0f for fp8-e4m3), `kZeroBits`
- **`packed_t<T>`** — Two-element packed alias: `packed_t<fp16_t>` = `fp16x2_t`, `packed_t<bf16_t>` = `bf16x2_t`, `packed_t<fp32_t>` = `fp32x2_t`. Use for vectorized loads/stores.
- **`device::cast<To, From>(value)`** — Type-safe cast using `DTypeTrait`, e.g. `cast<fp32x2_t, fp16x2_t>(v)`.
- **`device::unpack(value)`** — View a packed value as an `unpacked_t[kVecSize]` array reference (e.g. `fp32x2_t` → `fp32_t[2]`); element writes propagate back to the packed value.
- **`device::ReductionOp` (`SUM`/`MAX`/`MIN`) and `device::ReductionTrait<Op, T>::reduce(x, y)`** — One binary reduction step, dispatched through `DTypeTrait` (packed types reduce elementwise). This is the engine behind `warp::reduce`; use it directly when writing custom reductions.

### `vec.cuh` — Vectorized memory access (`AlignedVector`)

```cpp
#include <sgl_kernel/vec.cuh>
```

- **`device::AlignedVector<T, N>`** — Aligned storage for N elements of type T. N must be a power of two, `sizeof(T)*N <= 32`. Enables vectorized loads/stores for bandwidth efficiency. In terms of API/codegen constraints, the upper bound is 256-bit; in practice, 128-bit is the portable default, while 256-bit vectorization is typically only viable on `SM100+` and should be gated by an architecture check when needed.
  - `.load(ptr, offset)` — vectorized load from `ptr[offset]`
  - `.store(ptr, offset)` — vectorized store to `ptr[offset]`
  - `.fill(value)` — fill all N elements with `value`
  - `operator[](i)` — element access

### `tile.cuh` — `tile::Memory` (strided memory access pattern)

```cpp
#include <sgl_kernel/tile.cuh>
```

- `tile::Memory<T>` is fundamentally a **1D cooperative accessor** over a contiguous region.
- **`device::tile::Memory<T>::cta(blockDim.x)`** — Creates a tile accessor where each thread handles `tid = threadIdx.x` with stride `tsize` (for `cta(blockDim.x)`, this is `blockDim.x`). Common for loops over a 1D array.
- **`.load(ptr, offset)`** — loads `ptr[tid + offset * tsize]`
- **`.store(ptr, val, offset)`** — stores to `ptr[tid + offset * tsize]`
- **`.in_bound(n, offset)`** — boundary check

For a **2D tile**, either flatten `(row, col)` into a linear tile index first, or compute the address manually with `ptr[row * stride + col]` using your thread/block coordinates.

### `math.cuh` — Device math (`device::math::`)

```cpp
#include <sgl_kernel/math.cuh>
```

- `device::math::max/min<T>(a, b)` — type-dispatched binary math via `DTypeTrait`
- `device::math::abs/sqrt/rsqrt/exp/sin/cos<T>(x)` — type-dispatched unary math via `DTypeTrait`

### `warp.cuh` — Warp-level primitives

```cpp
#include <sgl_kernel/warp.cuh>
```

- `device::warp::reduce<Op, kNumThreads, kInner>(value, active_mask)` — generic warp reduction via `__shfl_xor_sync`. `Op` is a `device::ReductionOp` (`SUM`/`MAX`/`MIN`); `kNumThreads` is a power-of-two group size (default 32 = full warp); `kInner=true` (default) reduces within each `kNumThreads`-sized group, `kInner=false` reduces across groups (lanes at the same offset in different groups).
- `device::warp::reduce_sum/reduce_max/reduce_min<kNumThreads, kInner>(value)` — convenience wrappers over `reduce`. Work for any type with a `ReductionTrait`: floats, integers, and packed x2 types.

### `cta.cuh` — CTA-level primitives

```cpp
#include <sgl_kernel/cta.cuh>
```

- `device::cta::reduce_max<T>(value, smem, min_value)` — CTA-wide max using shared memory + warp reduction. Caller is responsible for a `__syncthreads()` after if the result in `smem[0]` is needed.

### `atomic.cuh` — Atomic operations

```cpp
#include <sgl_kernel/atomic.cuh>
```

- `device::atomic::max(float* addr, float value)` — float atomic max (handles negative values correctly via bit tricks).

### `runtime.cuh` — Occupancy and device info

```cpp
#include <sgl_kernel/runtime.cuh>
```

- `host::runtime::get_blocks_per_sm(kernel, block_dim)` — max active blocks per SM (occupancy)
- `host::runtime::get_sm_count(device_id)` — number of SMs on the device
- `host::runtime::get_cc_major(device_id)` — compute capability major version

**Persistent kernel pattern** (cap blocks to SM count × occupancy):
```cpp
static const uint32_t max_occ = runtime::get_blocks_per_sm(kernel, kBlockSize);
static const uint32_t num_sm  = runtime::get_sm_count(device.unwrap().device_id);
const auto num_blocks = std::min(num_sm * max_occ, div_ceil(n, kBlockSize));
LaunchKernel(num_blocks, kBlockSize, device.unwrap())(kernel, params);
```

### Real kernels that use them

- `python/sglang/kernels/jit/csrc/elementwise/add_constant.cuh` — minimal runnable reference
- `python/sglang/kernels/jit/csrc/elementwise/rmsnorm.cuh` — `TensorMatcher` + `LaunchKernel` + `tile::Memory`
- `python/sglang/kernels/jit/csrc/elementwise/qknorm.cuh` — `runtime::get_blocks_per_sm` + persistent kernel pattern
