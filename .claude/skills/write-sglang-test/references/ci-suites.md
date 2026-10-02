# SGLang CI suites

Suite names a test can register for. The authority is `PER_COMMIT_SUITES` / `NIGHTLY_SUITES` in `test/run_suite.py`; pick one with "Choosing a Suite" in SKILL.md.

## Contents

- Per-commit (CUDA)
- Per-commit (AMD)
- Per-commit (Ascend NPU)
- Per-commit (Intel XPU)
- Nightly

## Per-commit (CUDA)

| Suite | Runner (label) | Description |
|-------|----------------|-------------|
| `base-a-test-1-gpu-small` | `1-gpu-5090` | Quick checks on a small NVIDIA GPU before heavier stages |
| `base-a-test-cpu` | `ubuntu-latest` | CPU-only unit tests |
| `base-b-test-1-gpu-small` | `1-gpu-5090` | Core engine tests that fit a 5090-class card |
| `base-b-test-1-gpu-large` | `1-gpu-h100` | Tests that need H100-class memory or kernels (e.g. FA3) |
| `base-b-test-2-gpu-large` | `2-gpu-h100` | Two-GPU correctness and parallelism (TP/PP) on H100 |
| `base-b-test-4-gpu-b200` | `4-gpu-b200` | Early Blackwell coverage (SM100+ paths) on four GPUs |
| `base-b-kernel-unit-test-1-gpu-large` | `1-gpu-h100` | JIT kernel correctness tests (`test/registered/kernels/ops/`) |
| `base-b-kernel-unit-test-4-gpu-b200` | `4-gpu-b200` | JIT kernel correctness tests for Blackwell / SM100-specific paths |
| `base-b-kernel-unit-test-8-gpu-h200` | `8-gpu-h200` | Multi-GPU JIT kernel correctness tests (`test/registered/kernels/ops/`) |
| `base-b-kernel-benchmark-test-1-gpu-large` | `1-gpu-h100` | JIT kernel benchmarks (`test/registered/kernels/benchmark/`) |
| `base-c-test-4-gpu-h100` | `4-gpu-h100` | Large 4-GPU H100 integration and scaling tests |
| `base-c-test-8-gpu-h200` | `8-gpu-h200` | Large 8-GPU H200 runs for big models and parallelism |
| `base-c-test-8-gpu-h20` | `8-gpu-h20` | Large 8-GPU H20 runs for big models |
| `base-c-test-deepep-4-gpu-h100` | `4-gpu-h100` | DeepEP expert-parallel and networking on four H100s |
| `base-c-test-8-gpu-b200` | `8-gpu-b200` | 8-GPU B200 suite |
| `base-c-test-4-gpu-b200` | `4-gpu-b200` | 4-GPU B200 suite for large models on Blackwell |
| `base-c-test-4-gpu-b200-small` | `4-gpu-b200` | Smaller 4-GPU B200 suite split onto low-disk B200 runners |
| `base-c-test-4-gpu-gb200` | `4-gpu-gb200` | 4-GPU GB200 suite for Grace Blackwell |

## Per-commit (AMD)

| Suite | Runner (label) | Description |
|-------|----------------|-------------|
| `stage-a-test-1-gpu-small-amd` | `linux-mi325-1gpu-sglang` | Quick checks on one MI325-class GPU |
| `stage-b-test-1-gpu-small-amd` | `linux-mi325-1gpu-sglang` | Core 1-GPU AMD tests |
| `stage-b-test-1-gpu-small-amd-nondeterministic` | `linux-mi325-1gpu-sglang` | Non-deterministic 1-GPU AMD tests |
| `stage-b-test-1-gpu-small-amd-mi35x` | `linux-mi35x-gpu-1` | 1-GPU tests on MI35x hardware |
| `stage-b-test-1-gpu-large-amd` | `linux-mi325-1gpu-sglang` | Large 1-GPU AMD tests |
| `stage-b-test-2-gpu-large-amd` | `linux-mi325-2gpu-sglang` | 2-GPU ROCm correctness and parallel setups |
| `stage-b-test-large-8-gpu-mi35x-disaggregation-amd` | `linux-mi35x-gpu-8.fabric` | PD disaggregation and RDMA on 8×MI35x fabric |
| `stage-c-test-4-gpu-amd` | `linux-mi325-4gpu-sglang` | 4-GPU AMD integration |
| `stage-c-test-large-8-gpu-amd` | `linux-mi325-8gpu-sglang` | 8-GPU MI325 scaling and integration |
| `stage-c-test-large-8-gpu-amd-mi35x` | `linux-mi35x-gpu-8` | 8-GPU MI35x scaling |


## Per-commit (Ascend NPU)

| Suite | Runner (label) | Description |
| --- | --- | --- |
| `per-commit-1-npu-a2` | `linux-aarch64-a2-1` | 1-NPU LLM CI machine |
| `per-commit-2-npu-a2` | `linux-aarch64-a2-2` | 2-NPU LLM CI machine |
| `per-commit-4-npu-a3` | `linux-aarch64-a3-4` | 4-NPU LLM CI machine |
| `per-commit-16-npu-a3` | `linux-aarch64-a3-16` | 16-NPU LLM CI machine  |
| `multimodal-gen-test-1-npu-a3` | `linux-aarch64-a3-2` | 1-NPU multimodal CI machine |
| `multimodal-gen-test-2-npu-a3` | `linux-aarch64-a3-16` | 2-NPU multimodal CI machine |
| `multimodal-gen-test-8-npu-a3` | `linux-aarch64-a3-16` | 8-NPU multimodal CI machine |

## Per-commit (Intel XPU)

Registered with `register_xpu_ci` (same signature as `register_cuda_ci`); run by `pr-test-xpu.yml` with `run_suite.py --hw xpu`.

- `stage-a-test-1-gpu-xpu`
- `stage-b-test-1-gpu-xpu`

## Nightly

Nightly suites are listed in `NIGHTLY_SUITES` in `test/run_suite.py`. They run via `nightly-test-nvidia.yml`, `nightly-test-amd.yml`, `nightly-test-npu.yml` and `nightly-test-intel.yml`, not `pr-test.yml`.

CUDA nightly suites are named `nightly-test-{runner_config}` — one per machine type, holding everything that runs nightly on it. There is no per-purpose split (kernel / eval / perf / precision all share their machine's suite); `auto_partition` splits the work. Examples:

- `nightly-test-1-gpu-large` (CUDA)
- `nightly-test-2-gpu-large` (CUDA)
- `nightly-test-8-gpu-h200` (CUDA)
- `nightly-test-4-gpu-gb300` (CUDA)
- `nightly-amd` (AMD)
- `nightly-amd-8-gpu-mi35x` (AMD)
- `nightly-1-npu-a3` (NPU)
- `nightly-2-npu-a3` (NPU)
- `nightly-4-npu-a3` (NPU)
- `nightly-8-npu-a3` (NPU)
- `nightly-16-npu-a3` (NPU)
- `nightly-xpu-1-gpu`, `nightly-xpu-2-gpu`, `nightly-xpu-4-gpu` (XPU)

> **Note**: Multimodal diffusion uses `python/sglang/multimodal_gen/test/run_suite.py`, not `test/run_suite.py`.
