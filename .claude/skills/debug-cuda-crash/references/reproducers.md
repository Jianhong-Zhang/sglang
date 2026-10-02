# Reproducers: confirm kernel API logging works on this box

Each writes a small script that indexes out of range inside a registered custom op, then runs it at levels 1, 3 and 10.

## Contents

- [LLM crash](#llm-crash)
- [Diffusion crash](#diffusion-crash)

## LLM crash

Create a temporary reproducer:

```bash
python3 - <<'PY'
from pathlib import Path
Path("/tmp/sglang_llm_crash.py").write_text(
    "import torch\\n"
    "import torch.nn.functional as F\\n"
    "from sglang.srt.utils.custom_op import register_custom_op\\n\\n"
    "def _fake_embedding(indices, table):\\n"
    "    return torch.empty((*indices.shape, table.shape[-1]), device=table.device, dtype=table.dtype)\\n\\n"
    "@register_custom_op(op_name='mock_llm_cuda_crash', fake_impl=_fake_embedding)\\n"
    "def mock_llm_cuda_crash(indices, table):\\n"
    "    out = F.embedding(indices, table)\\n"
    "    torch.cuda.synchronize()\\n"
    "    return out\\n\\n"
    "table = torch.randn(4, 8, device='cuda', dtype=torch.float16)\\n"
    "indices = torch.tensor([0, 7], device='cuda', dtype=torch.long)\\n"
    "mock_llm_cuda_crash(indices, table)\\n"
)
PY

SGLANG_KERNEL_API_LOGLEVEL=1 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_llm_level1.log \
python3 /tmp/sglang_llm_crash.py
```

What to expect:
- The script exits with a CUDA `device-side assert`
- The log still contains the last API boundary before the crash

Try the same example at level 3:

```bash
SGLANG_KERNEL_API_LOGLEVEL=3 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_llm_level3.log \
python3 /tmp/sglang_llm_crash.py
```

Now the log shows tensor metadata before the crash.

Try level 10:

```bash
SGLANG_KERNEL_API_LOGLEVEL=10 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_llm_level10.log \
SGLANG_KERNEL_API_DUMP_DIR=/tmp/sglang_llm_level10_dumps \
python3 /tmp/sglang_llm_crash.py
```

Now you should see:
- A log entry for `sglang.custom_op.mock_llm_cuda_crash`
- A dump directory with `inputs.pt`
- `metadata.json` showing `execution_status: "exception"`
- No `outputs.pt`, because the kernel crashed before producing output

For real-model success-path level-10 dumps, it is often easier to temporarily disable CUDA graph and piecewise CUDA graph for the debug run.

## Diffusion crash

Create a temporary diffusion-side reproducer:

```bash
python3 - <<'PY'
from pathlib import Path
Path("/tmp/sglang_diffusion_crash.py").write_text(
    "import torch\\n"
    "import torch.nn.functional as F\\n"
    "from sglang.multimodal_gen.runtime.layers.utils import register_custom_op\\n\\n"
    "def _fake_embedding(positions, cache):\\n"
    "    return torch.empty((*positions.shape, cache.shape[-1]), device=cache.device, dtype=cache.dtype)\\n\\n"
    "@register_custom_op(op_name='mock_diffusion_cuda_crash', fake_impl=_fake_embedding)\\n"
    "def mock_diffusion_cuda_crash(positions, cache):\\n"
    "    out = F.embedding(positions, cache)\\n"
    "    torch.cuda.synchronize()\\n"
    "    return out\\n\\n"
    "cache = torch.randn(4, 64, device='cuda', dtype=torch.float16)\\n"
    "positions = torch.tensor([0, 9], device='cuda', dtype=torch.long)\\n"
    "mock_diffusion_cuda_crash(positions, cache)\\n"
)
PY

SGLANG_KERNEL_API_LOGLEVEL=1 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_diffusion_level1.log \
python3 /tmp/sglang_diffusion_crash.py
```

Try level 3:

```bash
SGLANG_KERNEL_API_LOGLEVEL=3 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_diffusion_level3.log \
python3 /tmp/sglang_diffusion_crash.py
```

Try level 10:

```bash
SGLANG_KERNEL_API_LOGLEVEL=10 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_diffusion_level10.log \
SGLANG_KERNEL_API_DUMP_DIR=/tmp/sglang_diffusion_level10_dumps \
python3 /tmp/sglang_diffusion_crash.py
```

If your local environment has unrelated FlashInfer import issues, resolve them in the shell before running the example. The example itself does not set any `FLASHINFER_*` environment variable.
