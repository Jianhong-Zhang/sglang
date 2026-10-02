# Kernel API log and dump formats

What each `SGLANG_KERNEL_API_LOGLEVEL` writes, with sample output.

## Contents

- [Basic Logging (Function Names Only)](#basic-logging-function-names-only)
- [Detailed Logging (Inputs with Metadata)](#detailed-logging-inputs-with-metadata)
- [Full Logging (With Tensor Statistics)](#full-logging-with-tensor-statistics)
- [Crash-Safe Dumps (Inputs Saved Before Execution)](#crash-safe-dumps-inputs-saved-before-execution)

## Basic Logging (Function Names Only)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=1
export SGLANG_KERNEL_API_LOGDEST=stdout

python my_script.py
```

Output:
```
================================================================================
[<timestamp>] SGLang Kernel API Call: RMSNorm.forward
================================================================================
[<timestamp>] SGLang Kernel API Call: sglang.quant_method.UnquantizedLinearMethod.apply
================================================================================
[<timestamp>] SGLang Kernel API Call: sglang.custom_op.fused_inplace_qknorm
```

## Detailed Logging (Inputs with Metadata)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
export SGLANG_KERNEL_API_LOGDEST=debug.log

python my_script.py
```

Output in `debug.log`:
```
================================================================================
[<timestamp>] SGLang Kernel API Call: sglang.quant_method.UnquantizedLinearMethod.apply
Positional input arguments:
  arg[0]=QKVParallelLinear(
      repr=QKVParallelLinear(in_features=1024, output_features=4096, bias=False, tp_size=1, gather_output=False)
    )
  arg[1]=Tensor(
      shape=(1, 1024)
      dtype=torch.bfloat16
      device=cuda:0
      requires_grad=False
      is_contiguous=True
    )
  arg[2]=None
Output:
  return=Tensor(
      shape=(1, 4096)
      dtype=torch.bfloat16
      device=cuda:0
      requires_grad=False
      is_contiguous=True
    )
```

## Full Logging (With Tensor Statistics)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=5
export SGLANG_KERNEL_API_LOGDEST=debug.log

python my_script.py
```

Additional output:
```
================================================================================
[<timestamp>] SGLang Kernel API Call: diffusion.quant_method.UnquantizedLinearMethod.apply
Positional input arguments:
  arg[1]=Tensor(
      shape=(1, 77, 768)
      dtype=torch.bfloat16
      device=cuda:0
      requires_grad=False
      is_contiguous=True
      min=-27.250000
      max=28.500000
      mean=0.011723
      nan_count=0
      inf_count=0
    )
Output:
  return=Tensor(
      shape=(1, 77, 2304)
      dtype=torch.bfloat16
      device=cuda:0
      requires_grad=False
      is_contiguous=True
      min=-8.937500
      max=9.375000
      mean=0.009460
      nan_count=0
      inf_count=0
    )
```

## Crash-Safe Dumps (Inputs Saved Before Execution)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=10
export SGLANG_KERNEL_API_LOGDEST=debug.log
export SGLANG_KERNEL_API_DUMP_DIR=/tmp/sglang_kernel_api_dumps

python my_script.py
```

At level 10, SGLang saves the inputs before execution. If the kernel crashes, the dump directory still contains the inputs and exception metadata.

If CUDA graph capture is active, tensor dumps are skipped automatically to avoid capture-time CUDA errors. In that case, you still get the kernel API call log, but not `inputs.pt` / `outputs.pt`.

Level-10 dumps are best understood as crash-safe call snapshots. They always preserve the observed call boundary. They do not guarantee one-click replay for every method, because some methods depend on module state that is not serialized into the dump.

Level-10 dump layout: one directory per call under `SGLANG_KERNEL_API_DUMP_DIR`:

```text
<dump_dir>/<YYYYMMDD_HHMMSS_mmm>_pid<pid>_RotaryEmbedding.forward_call0001/inputs.pt
<dump_dir>/<YYYYMMDD_HHMMSS_mmm>_pid<pid>_RotaryEmbedding.forward_call0001/metadata.json
<dump_dir>/<YYYYMMDD_HHMMSS_mmm>_pid<pid>_RotaryEmbedding.forward_call0001/outputs.pt
```

`metadata.json` excerpt:

```json
{
  "function_name": "RotaryEmbedding.forward",
  "timestamp": "<YYYYMMDD_HHMMSS_mmm>",
  "process_id": <pid>,
  "execution_status": "completed",
  "input_tensor_keys": ["arg_0", "arg_1", "arg_2"],
  "output_tensor_keys": ["result_0", "result_1"]
}
```
