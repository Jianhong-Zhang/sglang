---
name: debug-cuda-crash
description: Debugs CUDA crashes and numerical faults in SGLang with the @debug_kernel_api logging decorator, which logs (and at level 10 dumps) each kernel call's inputs before it runs, so the failing call is visible even after the process aborts. Use on "an illegal memory access was encountered", "device-side assert triggered", out-of-bounds, NaN/Inf outputs, CUDA out of memory, or when pairing logs with compute-sanitizer or cuda-gdb.
---

# Debugging CUDA crashes with kernel API logging

Logging covers the highest-value kernel boundaries: ops registered through `register_custom_op(...)` /
`register_custom_op_from_extern(...)`, LLM attention/linear/quantization/multi-platform wrapper entry points, diffusion
attention/linear/rotary/custom-op wrappers, and selected direct `torch.ops.sglang.*` hotspots. It does not cover every
pure PyTorch call. If the failing path goes through none of these, the log stays silent.

## Levels and variables

| Variable | Value | Effect |
|---|---|---|
| `SGLANG_KERNEL_API_LOGLEVEL` | `0` | No logging (default; the decorator returns the original callable, zero overhead) |
| | `1` | Function names only |
| | `3` | Inputs and outputs with shape, dtype, device, contiguity |
| | `5` | Level 3 plus `min`/`max`/`mean`/`nan_count`/`inf_count` |
| | `10` | Level 5 plus crash-safe dumps: `inputs.pt` and `metadata.json` written **before** the call |
| `SGLANG_KERNEL_API_LOGDEST` | `stdout` / `stderr` / `<path>` | Where the log goes; `%i` in a path expands to the process ID |
| `SGLANG_KERNEL_API_DUMP_DIR` | `<path>` | Level-10 dump root (`%i` works here too) |
| `SGLANG_KERNEL_API_DUMP_INCLUDE` / `_EXCLUDE` | shell wildcards | Only dump / skip matching API names, e.g. `'sglang.custom_op.*'`, `'*.fake_impl'` |

## Procedure

1. **Level 3, to a file.** A crash can abort before stdout flushes, so log to a file:

       export SGLANG_KERNEL_API_LOGLEVEL=3
       export SGLANG_KERNEL_API_LOGDEST=debug.log

   Re-run. The last entry with inputs logged but no output logged is the failing call. Check its shapes, dtypes,
   device placement and contiguity against what the kernel expects.
2. **Level 5 for NaN/Inf.** Read `nan_count`/`inf_count` and suspicious `min`/`max` on the inputs of each call leading
   up to the failure. Bad inputs mean the fault is upstream of the crashing kernel.
3. **Level 10 when the inputs must survive the crash.** Set `SGLANG_KERNEL_API_DUMP_DIR`, and narrow it with
   `SGLANG_KERNEL_API_DUMP_INCLUDE`/`_EXCLUDE` when every call is dumped. A crashed call leaves `inputs.pt` and
   `metadata.json` with `execution_status: "exception"`, and no `outputs.pt`. A dump is a snapshot of the call
   boundary, not a guaranteed one-click replay: some methods depend on module state that is not serialized.
4. **Multi-rank or multi-process:** put `%i` in both `SGLANG_KERNEL_API_LOGDEST` (e.g. `debug_rank_%i.log`) and
   `SGLANG_KERNEL_API_DUMP_DIR`, so the ranks do not write into one file or one dump tree.
5. **Still unexplained:** pair level 3 with compute-sanitizer (which kernel faulted, on which address) or cuda-gdb
   (backtrace); see [references/native-tools.md](references/native-tools.md).

**Done when** the failing API boundary is named, together with its inputs (shape, dtype, device and, for numerical
faults, the statistics), and you can say which input is wrong or that all of them are valid.

## CUDA graph capture

- Level-5 statistics and level-10 dumps are skipped while a graph is being captured (they would sync or copy to CPU).
  `statistics=[skipped: CUDA graph capture in progress]` and `Tensor dump skipped: CUDA graph capture in progress` are
  expected, not errors.
- To get statistics or dumps from a real model run, disable CUDA graph and piecewise CUDA graph for the debug run.

## Troubleshooting

- **No logs:** check `$SGLANG_KERNEL_API_LOGLEVEL` and `$SGLANG_KERNEL_API_LOGDEST`, and that the failing path goes
  through a covered boundary (see above).
- **Too much output:** drop to level 3, or keep level 10 and filter the dumps with `_INCLUDE`/`_EXCLUDE`.
- **Synchronous host-side errors:** `CUDA_LAUNCH_BLOCKING=1` is a separate follow-up experiment, not the default. It
  changes timing and can hide concurrency bugs.
- Unset `SGLANG_KERNEL_API_LOGLEVEL` when done.

## References

- [references/log-formats.md](references/log-formats.md): sample output at each level, and the level-10 dump layout
  and `metadata.json`.
- [references/error-patterns.md](references/error-patterns.md): illegal memory access / device-side assert, NaN/Inf
  and OOM: what to look for in the log, with worked examples.
- [references/native-tools.md](references/native-tools.md): compute-sanitizer, cuda-gdb, and kernel `printf()`
  (including how to choose a print thread in warp-specialized kernels).
- [references/reproducers.md](references/reproducers.md): an LLM and a diffusion reproducer. Run one to confirm logging
  and dumps work on this box before debugging the real failure.
