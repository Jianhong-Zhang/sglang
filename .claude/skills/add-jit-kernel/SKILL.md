---
name: add-jit-kernel
description: Adds a lightweight JIT CUDA kernel to sglang (source in python/sglang/kernels/jit/csrc, compiled on first use by load_jit through TVM-FFI), with its Python wrapper in python/sglang/kernels/ops, a registered correctness test and a marker benchmark. Use when writing or changing a kernel that needs no CUTLASS or other large C++ dependency, or when wrapping, testing or benchmarking a jit_kernel op.
---

# Adding a JIT Kernel to SGLang

Worked example: an element-wise `scale(x, factor) = x * factor` — `x` a CUDA tensor in FP16 / BF16 / FP32, `factor` a runtime float, output allocated internally. The complete files are in `references/example-scale/`; each step names the file to copy and adapt.

**JIT or AOT:** use JIT unless the kernel depends on CUTLASS or another large C++ project (then use the `add-sgl-kernel` skill); kernels whose CUTLASS comes through `flashinfer` can still be JIT.

---

## Conventions

These hold for every step below.

These hold for every step below.

- **`namespace sglang` is where JIT code lives.** Open it after the include block and close it at the end of the file, with the device kernels, traits and host wrapper inside. The shared `host::` / `device::` helpers are nested in it too, so they resolve unqualified. `load_jit` emits the `TVM_FFI_DLL_EXPORT_TYPED_FUNC` wrapper inside `namespace sglang` as well, so the `kernel_name` you pass from Python needs no `sglang::` prefix.
- **Check where the check is cheapest: `static_assert` > C++ host check > cached Python > per-call Python.** Anything fixed at compile time is a `static_assert`. Anything about the tensors is a `TensorMatcher` / `CHECK_HOST` in the C++ launcher, free next to a kernel launch. A check Python cannot delegate goes inside the `@cache_once` module factory, where it runs once per specialisation. What remains in the per-call entry point costs interpreter time on *every* forward, so it should be nothing but picking the module and allocating `out`.
- **Fixed-width integer types.** Prefer `int32_t` / `int64_t` / `uint32_t` / `size_t` over `int`, `long`, or `long long`, so an index has the same width on both sides of the FFI boundary. Bare `int` is fine only where the width plainly cannot matter — an unrolled loop counter over a `constexpr` bound, a template `int` parameter. Shapes arrive as `int64_t` (`SymbolicSize::unwrap()`); narrowing to `uint32_t` for in-kernel indexing is a deliberate act, so write the `static_cast` explicitly and only where the range is known.
- **Doxygen comments in C++.** Document exported entities with `///` or `/** ... */` blocks using `\brief`, `\param`, `\tparam`, `\return`, the way `include/sgl_kernel/` does. `python -m sglang.kernels.jit` writes `CommentFormat: Doxygen` into `.clangd` when clangd is 21 or newer, so these render on hover in the editor. Plain `//` remains fine for implementation notes inside a function body.
- **ASCII only in C++ and CUDA sources.** Write `--`, `->`, `<=` instead of `—`, `→`, `≤`, including in comments. `grep -nP '[^\x00-\x7F]' <file>` before committing.
- **`const T* __restrict__` for read-only pointers.** This is what `csrc/` does throughout, and it lets the compiler emit non-coherent (`LDG`) loads.
- **Watch the register budget.** For memory-bound kernels, keep to roughly 64 registers per thread so occupancy does not become the limit. Build once with `extra_cuda_cflags=["-Xptxas", "-v"]` to see the actual count, and prefer recomputing a value over letting it spill.
- **Prefer the `sgl_kernel/` abstractions over raw CUDA primitives.** Drop to raw primitives only for performance the abstraction cannot reach, and justify it in a comment. The header API (`TensorMatcher`, `LaunchKernel`, `AlignedVector`, `tile::Memory`, `DTypeTrait`, warp/CTA reductions, PDL, occupancy helpers) is catalogued in [references/sgl_kernel-headers.md](references/sgl_kernel-headers.md).

---

## Step 0 (optional): Generate a `.clangd` config

```bash
python -m sglang.kernels.jit -h  # for verbose help info about clangd configuration
python -m sglang.kernels.jit
python -m sglang.kernels.jit --dep cutlass flashinfer  # with cutlass/flashinfer dependency
```

Done when clangd resolves `#include <sgl_kernel/...>` in the editor.

---

## Step 1: Implement the CUDA kernel in `kernels/jit/csrc/`

Create `python/sglang/kernels/jit/csrc/elementwise/scale.cuh` — copy and adapt [references/example-scale/scale.cuh](references/example-scale/scale.cuh).

- Validate tensors only with `TensorMatcher`; launch only with `LaunchKernel`; use `AlignedVector` for 128-bit loads/stores.
- Pass runtime scalars like `factor` as arguments unless compile-time specialisation is genuinely required.
- With PDL, place `PDLWaitPrimary()` right before the first read of upstream data, not at the top of the kernel.

Done when the file is inside `namespace sglang`, uses no raw CUDA primitive that a header already covers, and `grep -nP '[^\x00-\x7F]' <file>` prints nothing.

---

## Step 2: Add the Python wrapper in `kernels/ops/`

The wrapper lives next to its functional group under `python/sglang/kernels/ops/`, not beside the CUDA source — `kernels/jit/` holds only the JIT infrastructure (`csrc/`, `include/`, `utils/`, `benchmark/`). Create `python/sglang/kernels/ops/elementwise/scale.py` — copy and adapt [references/example-scale/scale.py](references/example-scale/scale.py).

- Use `cache_once` — **not** `functools.lru_cache` (incompatible with `torch.compile`).
- `load_jit` first arg(s) form the unique build marker; same marker = same cached binary. Put only compile-time specialisation knobs in it; runtime values like `factor` stay runtime.
- `cuda_wrappers`: `(export_name, kernel_symbol)` — `export_name` is called from Python.
- `make_cpp_args(dtype, ...)` converts `torch.dtype` to the C++ alias; `is_arch_support_pdl()` gives the PDL template argument.
- The entry point stays thin (see Conventions); the supported-dtype guard goes in the `@cache_once` module factory.

| `torch.dtype`      | C++ type   |
|--------------------|------------|
| `torch.float16`    | `fp16_t`   |
| `torch.bfloat16`   | `bf16_t`   |
| `torch.float32`    | `fp32_t`   |

Done when `from sglang.kernels.ops.elementwise.scale import scale` compiles on first call and returns `x * factor` for each supported dtype.

---

## Step 3 (optional): Tune JIT build flags

If the kernel uses math functions like `expf` or `sinf`, consider `--use_fast_math` (with a potential precision tradeoff):

```python
return load_jit(
    "scale",
    *args,
    cuda_files=["elementwise/scale.cuh"],
    cuda_wrappers=[("scale", f"scale<{args}>")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)
```

If the kernel requires SM90+, raise a clear Python error before calling `load_jit`. Arch gating has to live in Python — it decides whether to compile at all, so the C++ launcher never gets to run:

```python
if torch.cuda.get_device_capability()[0] < 9:
    raise RuntimeError("This kernel requires SM90 (Hopper) or later")
```

Done when the Step 4 tests still pass with the new flags.

---

## Step 4: Write tests (required)

Create `test/registered/kernels/ops/elementwise/test_scale.py` — copy and adapt [references/example-scale/test_scale.py](references/example-scale/test_scale.py).

- Tests live under `test/registered/kernels/ops/<group>/`, mirroring the wrapper's group — never under `python/sglang/` (the `check-no-registered-tests-in-package` pre-commit hook rejects a `register_*_ci(...)` there).
- CI does not run `pytest` there: `test/run_suite.py` collects module-level `register_*_ci(...)` calls by parsing each file's AST. Every file needs at least one CUDA entry with **literal** arguments, e.g. `register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")`.
- Suite choice (B200 / multi-GPU / nightly), the `{stage}-test-{runner_config}` naming and `disabled=` are covered once in the `write-sglang-test` skill.
- Cover the tail remainder (sizes that are not a multiple of the vector width) and every supported dtype.

Run like CI (from repo root); `pytest` on the single file is fine for iteration:

```bash
(cd test && python3 run_suite.py --hw cuda --suite base-b-kernel-unit-test-1-gpu-large)
```

Done when the suite run passes and lists `test_scale.py`.

---

## Step 5: Add a benchmark (required)

Create `test/registered/kernels/benchmark/elementwise/bench_scale.py` — copy and adapt [references/example-scale/bench_scale.py](references/example-scale/bench_scale.py). Register it for `base-b-kernel-benchmark-test-1-gpu-large`.

Use the project's `marker` framework, not `triton.testing`. Its decorators, `do_bench` knobs (`memory_args`, `memory_output`, `graph_clone_args`, `use_cuda_graph`, bandwidth column) and helpers are in [references/benchmark-marker.md](references/benchmark-marker.md).

```bash
python test/registered/kernels/benchmark/elementwise/bench_scale.py
(cd test && python3 run_suite.py --hw cuda --suite base-b-kernel-benchmark-test-1-gpu-large)
```

Done when the table prints a latency column for every implementation and, for a memory-bound kernel, a GB/s column whose bytes match what the kernel actually reads and writes.

---

## Troubleshooting

- **`No CI registry found in ...` from `run_suite.py`**: add a module-level `register_cuda_ci(...)` with literal `est_time`, `stage`, and `runner_config`; starred args and non-literal values break AST collection
- **JIT compilation fails**: ensure the `.cuh` file is under `python/sglang/kernels/jit/csrc/`; reduce template argument combinations
- **CUDA crash / illegal memory access**: `CUDA_LAUNCH_BLOCKING=1`; `compute-sanitizer --tool memcheck python ...`
- **Unstable benchmark results or missing GB/s column**: see `graph_clone_args` and `disable_log_bandwidth` in [references/benchmark-marker.md](references/benchmark-marker.md)

---

## References

- `docs/docs/developer_guide/development_jit_kernel_guide.mdx`
- `test/run_suite.py` — suite names, discovery of `test/registered/`, execution entrypoint for CI
- `python/sglang/test/ci/ci_register.py` — `register_cuda_ci` and AST registration rules
- `python/sglang/kernels/jit/utils/compile/` — `load_jit`, `make_cpp_args`
- `python/sglang/kernels/jit/utils/common.py` — `cache_once`, `should_run_full_tests`, `get_ci_test_range`
- `python/sglang/kernels/jit/include/sgl_kernel/` — headers; see [references/sgl_kernel-headers.md](references/sgl_kernel-headers.md)
- `python/sglang/kernels/jit/csrc/elementwise/add_constant.cuh` — minimal runnable reference

## Summary of Files Created

```
python/sglang/kernels/jit/csrc/elementwise/scale.cuh              # NEW: CUDA kernel
python/sglang/kernels/ops/elementwise/scale.py                    # NEW: Python wrapper
test/registered/kernels/ops/elementwise/test_scale.py             # NEW: Tests
test/registered/kernels/benchmark/elementwise/bench_scale.py      # NEW: Benchmark
```
