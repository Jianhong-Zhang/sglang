# Benchmark `marker` framework

`python/sglang/kernels/jit/benchmark/marker.py` (public names: `benchmark`, `parametrize`, `do_bench`, `skip`, `BenchResult`, `BenchSkip`) and `python/sglang/kernels/jit/benchmark/utils.py`. Do **not** use `triton.testing.perf_report` / `triton.testing.do_bench` directly.

## Decorators

- **`@marker.benchmark(line_arg, line_vals, *, unit="us")`** — the **innermost** decorator (bottom of the stack, directly above `def benchmark`). Declares the column axis: each value in `line_vals` becomes a result column, and `line_arg` is the parameter name passed into the benchmark function. `unit` is one of `"us" | "ms" | "s"`.
- **`@marker.parametrize(names, vals, ci_vals=None)`** — stackable, one per swept row axis, above `@marker.benchmark` (Cartesian product across all `parametrize` decorators). `names` may be a single name (`"size"`) or a comma-separated correlated tuple axis (`"h,d"`, with `vals` then a list of tuples like `[(1, 64), (2, 128)]`). The optional `ci_vals` is a smaller sweep auto-selected under `is_in_ci()` — the built-in CI-shrinking mechanism.
- The `line_arg` name and every `parametrize` name must match a parameter of the benchmark function.
- Call `benchmark.run()` (no `print_data=` kwarg — the framework prints directly).

## `marker.do_bench(fn, *, input_args=(), input_kwargs={}, ...)`

Runs `fn` and returns a `BenchResult`.

- `use_cuda_graph` — CUDA-graph timing by default; set `False` only for kernels that can't be captured.
- `memory_args` — defaults to `"all"` (footprint of all input args/kwargs). Pass an explicit tuple of tensors (e.g. `(k, v, indices)`) to count only the inputs the kernel touches.
- `memory_output` — defaults to `"out"`: re-runs `fn` once to capture its **returned** tensor and counts it. For in-place kernels (return `None`) the default counts nothing, so pass the written tensors explicitly (e.g. `memory_output=(k, v)`); the re-run is then skipped. `None` counts no output.
- Together `memory_args` + `memory_output` give the GB/s column; with both defaults `out = f(src)` reports `bytes(src) + bytes(out)`. For memory-bound kernels it is the most informative number.
- `disable_log_bandwidth` — defaults from `SGLANG_KERNEL_DISABLE_LOG_BANDWIDTH=1`; skips the bandwidth column. Use it for compute-bound kernels where bandwidth is misleading. A missing GB/s column means one of these is set.
- `graph_clone_args` / `graph_clone_kwargs` — which inputs to clone per CUDA-graph iteration to defeat L2 reuse. Defaults to `"all"`. If you narrow it, it must still cover every tensor the kernel *reads*, including in-place modified ones. Keep write-only tensors in it too: they set the rotation count, and a shared output buffer stays L2-hot the same way.
- `metrics=(0.5, "avg")` — reported quantiles; the first becomes the table latency column.

## Helpers (`benchmark/utils.py`)

- **`create_random(*shape)` / `create_empty(*shape)`** — `torch.randn` / `torch.empty` with `DEFAULT_DTYPE` (`bfloat16`) and `DEFAULT_DEVICE` (`"cuda"`); override via `dtype=` / `device=`. Prefer them over open-coded `torch.randn(...)`.
- **`get_benchmark_range(full_range, ci_range)`** — `ci_range` under CI, `full_range` locally. For the `benchmark(...)` column axis (which has no `ci_vals`); row axes use `ci_vals`.

## Real examples

- `test/registered/kernels/benchmark/layernorm/bench_qknorm.py` — multi-axis `parametrize` (with `ci_vals`) + in-place `memory_output`
- `test/registered/kernels/benchmark/kvcache/bench_store_cache.py` — scoped `memory_args` / `memory_output` + selective `graph_clone_args`
