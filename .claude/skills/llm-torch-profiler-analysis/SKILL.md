---
name: llm-torch-profiler-analysis
description: "Triages torch.profiler traces from sglang, vLLM, TensorRT-LLM and TokenSpeed (CUDA or Intel XPU) into one three-table report: kernel, overlap-opportunity and fuse-pattern tables, mapped back to Python code paths. Use when asked which kernels dominate prefill or decode, to analyze a trace.json(.gz) or profile directory, to profile a running server, or to check whether a known fusion or overlap applied."
---

# Unified LLM Torch Profiler Analysis

## Overview

There is one public workflow, `triage`, and it always prints the same three tables:

- kernel table
- overlap-opportunity table
- fuse-pattern table

By default, all three tables only render rows at or above `1.0%` cumulative GPU-time share.
Rows below that are hidden by default unless the user asks for a lower cutoff.

Keep the fuse-pattern table source-backed and deterministic.
Do not turn it into a fuzzy matcher.

On Intel XPU a trace has no device-execution timeline: the rows are
`urEnqueueKernelLaunch` runtime events, so the GPU-time column is host
launch-dispatch time, not device time. The script prints this note under the
tables; carry it into the summary.

## Scripts

Run every script from the skill directory (`python3 scripts/<name> --help` for flags).

| Script | Run or read | Purpose |
| --- | --- | --- |
| `scripts/analyze_llm_torch_profile.py` | run | The entrypoint: triage an existing trace or drive live capture; stdout is the report |
| `scripts/analyze_sglang_torch_profile.py` | run | Backwards-compatibility shim for older `analyze_sglang_torch_profile.py` calls; forwards to the entrypoint |
| `scripts/render_triage_markdown_bundle.py` | run | Bundle several runs' tables into one markdown document, grouped by model and framework |
| `scripts/probe_llm_server.py` | run | Small correctness and latency probe against an OpenAI-compatible server |
| `scripts/make_trtllm_py_executor_override.py` | run | Generate a TensorRT-LLM `py_executor.py` with `with_stack=True` |
| `scripts/run_{sglang,vllm,trtllm}_*_profile_host.sh`, `scripts/run_llm_single_model_matrix_host.sh` | run | Host runners that launch a server and capture; they `docker exec` into a container named `sglang_bbuf`, so read them and adapt that before use elsewhere |
| `scripts/triage_kernel_helpers.py`, `scripts/triage_overlap_helpers.py`, `scripts/profile_common.py` | read | Modules the entrypoint imports; read only to debug a table |

## Capability Matrix

| Capability | SGLang | vLLM | TensorRT-LLM | TokenSpeed |
| --- | --- | --- | --- | --- |
| Existing trace triage | yes | yes | yes | yes |
| Single-trace live capture | yes | yes, if torch profiler is enabled on server | requires profiler control endpoints | yes, if `/start_profile` and `/stop_profile` are exposed |
| Two-trace mapping+formal triage | yes | yes | yes | yes |
| Stage-separated live workload | yes | yes | yes, with a writable shared trace dir or per-stage host runner | yes, via workload-separated HTTP capture |
| `--profile-by-stage` capture | yes | no | no | no |
| `--profile-prefix` control | yes | usually ignored on HTTP profiler route | usually ignored on HTTP profiler route | yes, mapped to `profile_id` |

For TensorRT-LLM, live capture only works when the server exposes `/start_profile` and
`/stop_profile`, and when the deployment already provides a shared trace path plus the
required env vars.

## Diffusion Backend Gate

For diffusion benchmark or profiling work, only analyze traces produced by the native
SGLang diffusion backend. If the run that generated the trace logs any of
`Falling back to diffusers backend`, `Using diffusers backend` or
`Loaded diffusers pipeline`, stop instead of analyzing the trace and handle it
as a backend-selection issue, not as native-kernel profiler evidence.

## Choose The Triage Shape

### Single-trace triage (default)

Use when you want the lowest-friction report:

- one trace is already available
- you mainly want kernel share and fusion clues
- you are comparing two runs side by side by running triage once per trace

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --input /path/to/profile_dir_or_trace.json.gz
```

The overlap table stays conservative in single-trace mode and will tell you when a
mapping/formal pair is needed.

### Two-trace triage

Use when you need:

- a stronger overlap answer
- graph-off source mapping plus graph-on final behavior
- more trustworthy overlap recommendations in the middle table

1. mapping trace with graph disabled or with the lower-fusion / more-readable config
2. formal trace with the real serving optimizations enabled

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --mapping-input /path/to/graph_off_profile_dir \
  --formal-input /path/to/graph_on_profile_dir
```

Do not call the mapping pass a "fast profile".
It exists to recover `kernel -> cpu_op -> python scope`.

### Live capture

To capture from a running server instead of an existing trace (single or
two-trace), read [references/live-capture.md](references/live-capture.md) for the
stage-separated capture contract and the per-framework flow.

## Workflow

1. If the user only wants a diagnosis, one trace is enough.
2. Prefer one-rank traces over merged traces whenever the profiler emitted both.
3. For a live server, let the script drive the profiler only when the framework-specific prerequisites are already met.
4. Prefer `--profile-workload both`; use `legacy` only when reproducing an old trace contract.
5. Prefer workload-separated SGLang capture; use internal `--profile-by-stage`
   mainly for `legacy` or manually collected traces.
6. Run `triage` and read the results in this order: kernel table, overlap-opportunity table, fuse-pattern table.
7. Before calling something a "new" optimization idea, compare the top rows against
   [references/fuse-overlap-catalog.md](references/fuse-overlap-catalog.md) (fuse rows) and
   [references/overlap-catalog.md](references/overlap-catalog.md) (overlap rows), and for split
   kernels on vLLM also [references/vllm-torch-compile-fusions.md](references/vllm-torch-compile-fusions.md).
   Check mainline rows first, then the `PR-backed / in-flight` sections. Prefer reporting:
   - an existing fused or overlap path that should already apply here
   - an existing path that appears disabled, unsupported, or regressed in this trace
   - an upstream pattern that is mainline elsewhere but missing locally, or still open upstream
   - a truly new opportunity only when no catalog entry fits
8. If no exact pattern fully matches but the trace is still close to a known family, add one flat
   similarity note after the tables: `high`, `medium`, or `low` only.
   Base that note on the full pattern shape, not on one kernel name alone.
   Prefer semantic cues such as producer-consumer chain, source locations, CPU op names, TP context, and model-specific structure.
   Do not rewrite the script table itself to include these heuristic judgments.

Done when the report below is returned with all three tables, and every
"new" claim has been checked against both catalogs.

## References

Load these only when needed:

- [references/live-capture.md](references/live-capture.md)
  - the stage-separated capture contract and per-framework live-capture flows (SGLang, vLLM, TensorRT-LLM, TokenSpeed, two-trace)
- [references/heuristics.md](references/heuristics.md)
  - overlap labels, dependency-risk interpretation, and limits
- [references/fuse-overlap-catalog.md](references/fuse-overlap-catalog.md)
  - source-backed catalog of existing fuse patterns, mainline plus PR-backed / in-flight rows
- [references/overlap-catalog.md](references/overlap-catalog.md)
  - source-backed catalog of existing kernel-overlap patterns, mainline plus PR-backed / in-flight rows
- [references/vllm-torch-compile-fusions.md](references/vllm-torch-compile-fusions.md)
  - current vLLM torch.compile fusion passes and the source patterns they target
- [references/source-map.md](references/source-map.md)
  - upstream SGLang profiler entrypoints and trace-writing paths; still most useful for SGLang-specific source follow-up
- [references/maintaining-catalogs.md](references/maintaining-catalogs.md)
  - only when refreshing the catalogs: last refresh heads and rescan commands
- [references/validation-history.md](references/validation-history.md)
  - only when re-validating the skill: the host runs it was validated on

## Output Contract

Return:

- trace path or generated profile path
- framework
- model/server args when available
- kernel table
- overlap-opportunity table
- fuse-pattern table
- optional similarity note with `high` / `medium` / `low` when exact matching is inconclusive
- one short summary of what dominates the run
- whether the overlap read came from single-trace triage or mapping/formal two-trace triage
