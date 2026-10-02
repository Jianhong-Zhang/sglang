# Validation History

Maintainer record of the host runs this skill was validated on. Not needed for
a triage: the rules these runs established are in SKILL.md and the references it
routes to.

## Real H100 Validation

The current reference run is the `4x H100` matrix captured on `2026-04-23` on
`h100_sglang` under:

- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3`

Rendered markdown bundle:

- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3/h100_large_model_matrix_v3_bundle.md`

Validated model directories:

- `mixtral_8x7b_instruct`
- `qwen2_5_32b_instruct`
- `qwen3_32b`

Each model directory contains:

- `analysis_sglang.txt`
- `analysis_vllm.txt`
- `analysis_trtllm.txt`
- framework-specific trace roots and probe artifacts

Validated matrix:

| Model | SGLang | vLLM | TensorRT-LLM | Result |
| --- | --- | --- | --- | --- |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | `4x H100` | `4x H100` | `4x H100` | three tables rendered correctly on all three frameworks; benchmark probes returned direct, non-empty text |
| `Qwen/Qwen2.5-32B-Instruct` | `4x H100` | `4x H100` | `4x H100` | three tables rendered correctly on all three frameworks; benchmark probes returned direct, non-empty text |
| `Qwen/Qwen3-32B` | `4x H100` | `4x H100` | `4x H100` | three tables rendered correctly on all three frameworks; vLLM and TensorRT-LLM chat probes often emitted `<think>` prefixes |

Use this run as the main H100 reference.
The older `2026-04-22` single-card Qwen3 matrix is still useful for bring-up, but it is
not the default reference anymore.
TokenSpeed support was added later and is covered by existing-trace triage and
HTTP profiler-control support, but it is not part of this older H100 validation
matrix yet.

Stage-separated workload validation captured on `2026-05-01` on `h100_sglang`:

- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260501_stage_split_validation`
- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260501_stage_split_validation_large`

Validated models:

| Model | GPU | Workloads | Result |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | `1x H100` | prefill `4090->1`, decode `1->2048` | generated separate `prefill/*.trace.json.gz` and `decode/*.trace.json.gz`; kernel, overlap, and fuse tables rendered with separate `extend/prefill` and `decode` sections |
| `Qwen/Qwen2.5-1.5B-Instruct` | `1x H100` | prefill `4090->1`, decode `1->2048` | generated separate `prefill/*.trace.json.gz` and `decode/*.trace.json.gz`; kernel, overlap, and fuse tables rendered with separate `extend/prefill` and `decode` sections |
| `Qwen/Qwen2.5-7B-Instruct` | `1x H100` | prefill `4090->1`, decode `1->2048` | generated separate traces; prefill kernel table captured 28-layer GEMM/FA3/RMSNorm work, decode captured 5-step graph launches, and fuse rows were split by stage |
| `Qwen/Qwen2.5-14B-Instruct` | `1x H100` | prefill `4090->1`, decode `1->2048` | generated separate traces; prefill kernel table captured 48-layer GEMM/FA3/RMSNorm work, decode captured 5-step graph launches, and fuse rows were split by stage |
| `Qwen/Qwen3-8B` | `2x H100`, TP=2 | prefill `4090->1`, decode `1->2048`, warmup 10/capture 5 | generated separate prefill/decode traces and all three tables; unique probe prompts avoided prefix-cache pollution in the prefill table |
| `mistralai/Mistral-7B-Instruct-v0.3` | `2x H100`, TP=2 | prefill `4090->1`, decode `1->2048`, warmup 10/capture 5 | generated separate prefill/decode traces and all three tables; server logs showed no repeated-prompt prefix-cache shortcut during the active prefill window |

This validation also covers the compatibility fix for older SGLang profiler
state machines (the internal scheduler guard step).
The 2026-05-01 two-card validation artifacts for the additional models are:

- `/data/bbuf/validate/core_skill_validation_20260501/qwen3_8b/profiler`
- `/data/bbuf/validate/core_skill_validation_20260501/mistral_7b_instruct_v03/profiler`

To render a validated run into one markdown document:

```bash
python3 scripts/render_triage_markdown_bundle.py \
  --analysis-root /data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3 \
  --output /data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3/h100_large_model_matrix_v3_bundle.md
```

The bundle groups by model and keeps the three tables for each framework.

H100 notes:

- all three frameworks now render kernel, overlap, and fuse tables with separate `extend/prefill` and `decode` sections when the trace contains a clean stage split
- SGLang live capture is validated and calls the server profiler API directly instead of shelling out to `sglang.profiler`
- SGLang trace flush can lag well beyond a few seconds, so the runner waits longer for artifacts than the earlier implementation
- SGLang kernel-site reconstruction keeps sampling disabled in the mapping path so the optimized parser does not perturb SGLang table output; equality rechecks matched for `Mixtral-8x7B-Instruct-v0.1`, `Qwen3-32B`, and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`
- TensorRT-LLM validation stays on `--backend pytorch`; the H100 flow writes the trace with `TLLM_TORCH_PROFILE_TRACE` and then analyzes the saved trace
- TensorRT-LLM current mainline was rechecked at `0722c5f47d2cae69ac1a237da51e550dd214532c` on 2026-06-26; the latest delta affects KV eviction / block-offset staging rather than profiler trace controls, so the `b9e1945` profiler evidence still applies
- TokenSpeed trace analysis has first-class registry rows for native TokenSpeed CuTe DSL MLA, MLA KV pack + FP8 quantize, fused top-k/top-p sampling, persistent lm_head GEMM, and NVFP4 GEMM + SwiGLU + quant; live capture still requires an existing torch-profiler trace until the target TokenSpeed image exposes a supported profiler API
- on this host, keep all trace roots under `/data/...`, not `/home/...`
