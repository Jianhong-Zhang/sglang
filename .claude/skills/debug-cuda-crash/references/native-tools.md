# Native tools: compute-sanitizer, cuda-gdb, kernel printf

Pair each with level-3 logging to a file, so the tool names the kernel and the log names the inputs.

## Contents

- [compute-sanitizer](#compute-sanitizer)
- [cuda-gdb](#cuda-gdb)
- [Kernel printf()](#kernel-printf)
  - [Warp-Specialized Kernels: Choosing the Right Print Thread](#warp-specialized-kernels-choosing-the-right-print-thread)
  - [Quick Reference](#quick-reference)
  - [Other Kernel Debugging Tools](#other-kernel-debugging-tools)

## compute-sanitizer

For harder bugs, combine kernel API logging with CUDA memory checking:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
export SGLANG_KERNEL_API_LOGDEST=debug.log

compute-sanitizer --tool memcheck python3 my_script.py
```

Use `debug.log` to see the exact inputs that reached the crashing API boundary.

Typical `compute-sanitizer` output:

```text
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at 0x1234 in SomeKernel
=========     by thread (256,0,0) in block (10,0,0)
=========     Address 0x... is out of bounds
```

Use the sanitizer output to identify the failing kernel and use `debug.log` to identify the exact tensors that reached the API boundary right before it.

If you need more synchronous host-side error reporting, you can try `CUDA_LAUNCH_BLOCKING=1` as a separate follow-up experiment. It is not part of the default workflow because it changes execution timing and can hide concurrency-related behavior.

## cuda-gdb

For crashes that need a stack trace instead of only memory diagnostics:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
export SGLANG_KERNEL_API_LOGDEST=debug.log

cuda-gdb --args python3 my_script.py
```

Inside `cuda-gdb`:

```text
(cuda-gdb) run
(cuda-gdb) where
```

Then correlate the backtrace with `debug.log`.

## Kernel printf()

When you own the CUDA kernel, `printf()` is still useful for narrowing down bad indices, bad launch geometry, or broken state propagation.

Basic pattern:

```cpp
__global__ void MyKernel(const float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("n=%d input0=%f\n", n, input[0]);
  }

  if (idx < n) {
    output[idx] = input[idx] * 2.0f;
  }
}
```

After launch, force the output to flush:

```python
my_kernel(...)
torch.cuda.synchronize()
```

For warp-specialized kernels, do not blindly print only on `threadIdx.x == 0`. Pick one representative thread per warp or per specialization group instead.

### Warp-Specialized Kernels: Choosing the Right Print Thread

Problem:
- `threadIdx.x == 0` only prints from the first warp in the block
- for warp-specialized kernels, that often misses the warp or group that is actually wrong

Better pattern:

```cpp
__global__ void WarpSpecializedKernel(...) {
  // Example: first lane of each warp
  if ((threadIdx.x % 32) == 0) {
    printf("warp=%d\n", threadIdx.x / 32);
  }
}
```

Or, if the kernel is organized in larger specialization groups, print once per group instead of once per block.

Common mistake:

```cpp
// Only warp 0 prints
if (threadIdx.x == 0) {
  printf("warp=%d\n", threadIdx.x / 32);
}
```

### Quick Reference

| Kernel Type | Print Condition | Notes |
|----------|----------|-------------|
| Simple kernel | `threadIdx.x == 0` | One thread per block is usually enough |
| Warp-specialized kernel | one representative lane per warp | e.g. `threadIdx.x % 32 == 0` |
| Group-specialized kernel | one representative lane per group | choose based on the kernel's scheduling layout |

### Other Kernel Debugging Tools

```cpp
assert(value >= 0.0f && "value must be non-negative");
static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be warp aligned");
```
