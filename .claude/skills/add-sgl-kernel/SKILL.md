---
name: add-sgl-kernel
description: Adds a heavyweight AOT CUDA/C++ kernel to sgl-kernel (python/sglang/kernels/aot) — the .cu source, the sgl_kernel_ops.h declaration, the TORCH_LIBRARY schema in common_extension.cc, the CMakeLists.txt source entry, the Python wrapper shipped in the sgl_kernel wheel, plus pytest tests and a benchmark. Use when the kernel depends on CUTLASS or another large C++ project, or must be part of the AOT wheel and torch op registration; for lightweight kernels use add-jit-kernel instead.
---

# Adding a Kernel to `sgl-kernel` (AOT / Heavyweight)

Worked example: an element-wise `scale(x, factor) = x * factor` — `x` a CUDA tensor in FP16 / BF16 / FP32 (dispatched via `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16` from `python/sglang/kernels/aot/include/utils.h`), written in-place or into a pre-allocated `out`. The complete example files are in `references/example-scale/`.

**AOT or JIT:** use AOT only when the kernel depends on CUTLASS or another large C++ project, or must ship in the AOT wheel / torch op registration flow. Otherwise use the `add-jit-kernel` skill; kernels whose CUTLASS comes through `flashinfer` can also be JIT.

Every new kernel ships with tests (pytest) and a benchmark script (`triton.testing`, which the existing `python/sglang/kernels/aot/benchmark/` scripts use).

## Checklist of files

```
python/sglang/kernels/aot/csrc/elementwise/scale.cu          # NEW: CUDA kernel + launcher      (Step 1)
python/sglang/kernels/aot/include/sgl_kernel_ops.h           # MODIFIED: C++ declaration        (Step 2)
python/sglang/kernels/aot/csrc/common_extension.cc           # MODIFIED: schema + registration  (Step 3)
python/sglang/kernels/aot/CMakeLists.txt                     # MODIFIED: add source (sorted)    (Step 4)
python/sglang/kernels/aot/python/sgl_kernel/elementwise.py   # MODIFIED: Python wrapper         (Step 5)
python/sglang/kernels/aot/python/sgl_kernel/__init__.py      # MODIFIED: re-export              (Step 5)
python/sglang/kernels/aot/tests/test_scale.py                # NEW: tests                       (Step 6)
python/sglang/kernels/aot/benchmark/bench_scale.py           # NEW: benchmark                   (Step 7)
```

---

## Step 1: Implement the kernel in `csrc/`

Pick the subdirectory: `csrc/elementwise/` for element-wise ops (this example); `csrc/gemm/`, `csrc/attention/`, `csrc/moe/` for other categories. Create `python/sglang/kernels/aot/csrc/elementwise/scale.cu` — copy and adapt [references/example-scale/scale.cu](references/example-scale/scale.cu).

- Use `at::Tensor`, `TORCH_CHECK` for validation, `at::cuda::getCurrentCUDAStream()` for the stream.
- Keep Python wrappers thin; do shape/dtype/device validation in C++ right around the launch path.
- `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16` covers `float`, `half` (FP16), `__nv_bfloat16` (BF16).
- Check for a device error after every kernel launch.
- **Arch constraints:** if the kernel only works on certain architectures, enforce it with `TORCH_CHECK` in the launcher and add matching skip logic to the tests and benchmark (Steps 6–7).

---

## Step 2: Add a C++ declaration in `include/sgl_kernel_ops.h`

Edit `python/sglang/kernels/aot/include/sgl_kernel_ops.h`, add to the elementwise section:

```cpp
void scale(at::Tensor& out, const at::Tensor& input, double factor);
```

---

## Step 3: Register the op in `csrc/common_extension.cc`

Edit `python/sglang/kernels/aot/csrc/common_extension.cc`, inside `TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)`:

```cpp
// From csrc/elementwise
m.def("scale(Tensor! out, Tensor input, float factor) -> ()");
m.impl("scale", torch::kCUDA, &scale);
```

- `Tensor!` means in-place / mutable output argument.
- The schema is important for `torch.compile` and for consistent call signatures.
- Keep the torch schema in PyTorch scalar types (`float` here), but the C++ launcher signature still needs `double` for scalar arguments accepted by `torch::Library`.

---

## Step 4: Add the new source file to `CMakeLists.txt`

Edit `python/sglang/kernels/aot/CMakeLists.txt`, add to `set(SOURCES ...)`, keeping the list **alphabetically sorted** (the file explicitly requires this):

```cmake
csrc/elementwise/scale.cu
```

---

## Step 5: Expose a Python API under `python/sglang/kernels/aot/python/sgl_kernel/`

Follow the existing module organization. For elementwise kernels, implement the wrapper in `python/sglang/kernels/aot/python/sgl_kernel/elementwise.py`:

```python
```python
import torch

def scale(
    input: torch.Tensor,
    factor: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Element-wise scale: out = input * factor.

    Supported dtypes: torch.float16, torch.bfloat16, torch.float32.

    Parameters
    ----------
    input  : CUDA input tensor
    factor : scale factor (float)
    out    : optional pre-allocated CUDA output tensor (same shape/dtype as input)
    """
    if out is None:
        out = torch.empty_like(input)
    torch.ops.sgl_kernel.scale.default(out, input, factor)
    return out
```
```

Then re-export it from `python/sglang/kernels/aot/python/sgl_kernel/__init__.py` following the existing import style used by other kernels.

---

## Step 6: Write tests (required)

Create `python/sglang/kernels/aot/tests/test_scale.py` — copy and adapt [references/example-scale/test_scale.py](references/example-scale/test_scale.py). Cover every supported dtype, plus the shape-mismatch and CPU-input error paths.

These are plain pytest files: `pr-test-sgl-kernel.yml` runs `pytest tests/` in the package. They do not take a `register_*_ci(...)` call (the pre-commit hook rejects one under `python/sglang/`); registered `run_suite.py` tests are covered by the `write-sglang-test` skill.

---

## Step 7: Add a benchmark (required)

Create `python/sglang/kernels/aot/benchmark/bench_scale.py` — copy and adapt [references/example-scale/bench_scale.py](references/example-scale/bench_scale.py). It compares against a PyTorch reference with `triton.testing.do_bench_cudagraph` and shrinks the sweep under `is_in_ci()`.

---

## Step 8: Build

```bash
cd python/sglang/kernels/aot
make build -j16
```

If you need to limit host resource usage:

```bash
cd python/sglang/kernels/aot
make build -j1 MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

Done when the build finishes and `python -c "import sgl_kernel; sgl_kernel.scale"` succeeds.

---

## Step 9: Validate

```bash
pytest python/sglang/kernels/aot/tests/test_scale.py -q
python python/sglang/kernels/aot/benchmark/bench_scale.py
```

Done when pytest reports no failures (arch-skipped cases count as skipped, not failed) and the benchmark prints a row per (dtype, size) with an `SGL Kernel` and a `PyTorch` column.

PR CI also runs `pr-test-sgl-kernel.yml`, including the B200 job `sgl-kernel-b200-test` when kernel changes are detected. Use that job as the Blackwell coverage signal for AOT `sgl-kernel` changes.

---

## Troubleshooting

- **Async CUDA errors**: `CUDA_LAUNCH_BLOCKING=1`
- **Memory errors**: `compute-sanitizer --tool memcheck python ...`
- **Build is too slow / OOM**: reduce `MAX_JOBS` and `SGL_KERNEL_COMPILE_THREADS`
- **Binary bloat**: use `python/sglang/kernels/aot/analyze_whl_kernel_sizes.py`
- **CMake sources list**: if your `.cu` file is missing from `SOURCES`, the symbol will be undefined at link time

---

## References

- `python/sglang/kernels/aot/README.md`
- `python/sglang/kernels/aot/include/sgl_kernel_ops.h`
- `python/sglang/kernels/aot/csrc/common_extension.cc`
- `python/sglang/kernels/aot/CMakeLists.txt`
- `python/sglang/kernels/aot/include/utils.h` — `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16` macro and friends
- `python/sglang/kernels/aot/csrc/elementwise/activation.cu` — reference for the FP16/BF16/FP32 dispatch pattern
