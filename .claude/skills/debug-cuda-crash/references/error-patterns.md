# Common CUDA errors: what to log and what to check

## Contents

- [Illegal Memory Access or Device-Side Assert](#illegal-memory-access-or-device-side-assert)
- [NaN or Inf](#nan-or-inf)
- [Out of Memory](#out-of-memory)
- [Example: Spot a Shape Bug from the Log](#example-spot-a-shape-bug-from-the-log)

## Illegal Memory Access or Device-Side Assert

**Typical errors**:
```
RuntimeError: CUDA error: an illegal memory access was encountered
torch.AcceleratorError: CUDA error: device-side assert triggered
```

Use:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
```

Check in the logs:
- ✅ Tensor shapes
- ✅ Tensor dtypes
- ✅ CUDA vs CPU device placement
- ✅ Tensor stride / contiguity
- ✅ Whether the failing call has inputs logged but no outputs logged

Typical shape-mismatch pattern:

```text
SGLang Kernel API Call: ...
arg[0]=Tensor(shape=(..., 128), ...)   # ✅ expected dimension
arg[1]=Tensor(shape=(..., 64), ...)    # ❌ mismatch
```

This often points to head-dim, hidden-dim, or cache-layout mismatch rather than a random CUDA failure.

## NaN or Inf

Use:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=5
```

Check:
- `min`
- `max`
- `mean`
- `nan_count`
- `inf_count`

Typical bad pattern:

```text
Tensor(
  ...
  min=-1234567.000000   # ❌ suspiciously large
  max=9876543.000000    # ❌ suspiciously large
  mean=nan              # ❌ bad
  nan_count=128         # ❌ found NaNs
  inf_count=0           # ✅ no Infs here
)
```

This usually means the bad values were already present before the crashing kernel.

## Out of Memory

Use:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
```

Check:
- Unexpectedly large tensor shapes
- Batch size
- Sequence length
- Frame count or image resolution in diffusion workloads

Also check whether a supposedly per-token or per-frame tensor accidentally became full-sequence or full-image sized.

Typical bad pattern:

```text
Tensor(
  shape=(1024, 8192, 128, 128)   # ❌ way too large
  ...
)
```

## Example: Spot a Shape Bug from the Log

Suppose the failing API log looks like this:

```text
[<timestamp>] SGLang Kernel API Call: RotaryEmbedding.forward
Positional input arguments:
  arg[0]=Tensor(shape=(1, 8), dtype=torch.int64, ...)
  arg[1]=Tensor(shape=(1, 8, 8, 256), dtype=torch.bfloat16, ...)    # ✅ query
  arg[2]=Tensor(shape=(1, 8, 4, 64), dtype=torch.bfloat16, ...)     # ❌ key head_dim mismatch
```

What this tells you:
- ✅ positions look reasonable
- ✅ query looks plausible
- ❌ key last dimension is inconsistent with the expected rotary/head dimension

That usually means the bug is in projection layout, head packing, or cache format rather than in the rotary kernel itself.
