# Live Capture

How to drive `scripts/analyze_llm_torch_profile.py` against a running server, per
framework. Run every command from the skill directory.

## Contents

- Stage-separated capture contract
- SGLang
- vLLM
- TensorRT-LLM
- TokenSpeed
- Two-trace capture from running servers
- `profile_by_stage`

## Stage-separated capture contract

Live capture must not use one mixed prompt as the default.
By default, `analyze_llm_torch_profile.py --url ...` captures two labeled
workloads and then renders the same three tables with separate stage sections:

- prefill: synthetic input length `4090`, output length `1`
- decode: synthetic input length `1`, output length `2048`

Every live profiler path warms up `10` steps before arming the profiler and then
captures `5` active steps by default. Keep this warmup/active split aligned
across SGLang, vLLM, and TensorRT-LLM before comparing kernel tables.

Use these options to override the contract when the benchmark workload is known:

```bash
--profile-workload both \
--warmup-steps 10 --num-steps 5 \
--prefill-input-len 4090 --prefill-output-len 1 \
--decode-input-len 1 --decode-output-len 2048
```

Allowed `--profile-workload` values:

- `both`: default; capture prefill and decode separately
- `prefill`: capture only the long-input / one-token workload
- `decode`: capture only the one-input / long-output workload
- `legacy`: keep the old `--probe-prompt` / `--probe-max-new-tokens` behavior

When the benchmark scenario's input/output lengths are known, do not use the
defaults; set the profiler lengths from that scenario. Prefill uses the
scenario's input length with output `1`, and decode uses input `1` with the
scenario's output length. For a mixed dataset, profile the slowest
representative bucket such as the p50 or p95 input/output pair used in the
benchmark report, and record the bucket in the artifact notes. If only a mixed
trace is available, label that limitation before comparing it to separated
prefill/decode traces.

For every framework, `--output-dir` must be a path both the server and this
machine can read.

## SGLang

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework sglang \
  --url http://127.0.0.1:30000 \
  --output-dir /path/to/sglang_profile_live \
  --num-steps 5 \
  --warmup-steps 10 \
  --profile-by-stage \
  --profile-workload both
```

The script sends `POST /start_profile` to the SGLang server directly.
The script writes `server_args.json`, warms up with the same workload shape,
sends the active probe requests after profiling is armed, captures separate
`prefill/` and `decode/` profile roots by default, and waits longer for trace
flush than the earlier implementation (SGLang trace flush can lag well beyond a
few seconds).
For the default workload-separated capture, the directory name labels the stage
and the SGLang internal `profile_by_stage` mode is not used inside each
workload. This avoids mixing a one-token prefill probe with a separate decode
profile. The helper still adds one internal guard step because older SGLang
profilers increment `forward_ct` before checking whether the profiler should
stop; without that guard, a `num_steps=1` prefill capture can stop just before
the actual prefill forward.

## vLLM

Launch vLLM with torch profiler enabled, for example:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/path/to/vllm_profile"}'
```

Then run:

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework vllm \
  --url http://127.0.0.1:8000 \
  --output-dir /path/to/vllm_profile \
  --num-steps 5 \
  --warmup-steps 10 \
  --no-profile-by-stage \
  --profile-workload both
```

For vLLM, `--output-dir` must point to the same `torch_profiler_dir` the server uses.
The current vLLM profiler config already defaults `torch_profiler_with_stack=true`,
so the runner only needs to set `torch_profiler_dir`.
A vLLM server in its own container must mount the Hugging Face cache and the
trace output directory at the same paths this machine uses.

## TensorRT-LLM

Use this only when the server exposes `POST /start_profile` and `POST /stop_profile`,
and the trace path is shared with the current machine. Stay on `--backend pytorch`.

Typical env expectations are:

- `TLLM_PROFILE_START_STOP=<start>-<stop>` such as `10-20`
- `TLLM_TORCH_PROFILE_TRACE=/shared/path/trace.json` or `.json.gz`

Then run:

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework trtllm \
  --url http://127.0.0.1:8000 \
  --output-dir /shared/path \
  --num-steps 5 \
  --no-profile-by-stage \
  --profile-workload both
```

If the deployment does not expose the profiler control endpoints, fall back to analyzing
an existing trace instead of trying live capture:

1. launch `trtllm-serve` with `TLLM_PROFILE_START_STOP=<start>-<stop>` and `TLLM_TORCH_PROFILE_TRACE=/shared/path/trace.json`
2. run a few benchmark requests
3. analyze the emitted trace with `--input /shared/path/trace.json`

If the TensorRT-LLM trace output is configured as one fixed file path, use
`scripts/run_trtllm_pytorch_profile_host.sh --stage prefill` and `--stage decode`
instead of direct `--profile-workload both`, so each stage gets its own trace file.

On the current TensorRT-LLM mainline path, `py_executor.py` creates the torch profiler
with `record_shapes=True` and `with_modules=True` but not `with_stack=True`, so the
tables lose Python locations. Keep the override path unless the target image proves
otherwise:

```bash
python3 scripts/make_trtllm_py_executor_override.py \
  --source /path/to/original/py_executor.py \
  --output /path/to/overrides/trtllm/py_executor_with_stack.py
```

The matrix runner does this automatically before TensorRT-LLM capture starts.

## TokenSpeed

For a running TokenSpeed server that exposes the profiler routes, the unified
helper can drive live capture:

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework tokenspeed \
  --url http://127.0.0.1:8000 \
  --output-dir /path/to/tokenspeed_profile \
  --num-steps 5 \
  --warmup-steps 10 \
  --no-profile-by-stage \
  --profile-workload both \
  --profile-prefix ts-triage
```

The helper sends `POST /start_profile` with:

- `output_dir`: the `--output-dir` path
- `activities`: `["CPU", "GPU"]`
- `with_stack`: `true`
- `record_shapes`: `false`
- `profile_id`: `--profile-prefix`, with `-prefill` or `-decode` appended during workload-separated capture

It then sends OpenAI-compatible probe requests and calls `POST /stop_profile`.
TokenSpeed writes files such as `ts-triage-prefill-TP-0.trace.json.gz` under the
output directory. If the server was launched with multiple TP ranks, expect one
trace per rank.

TokenSpeed's own manual profiler control surface can also be used:

```bash
curl -X POST http://127.0.0.1:8000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/path/to/tokenspeed","activities":["CPU","GPU"],"with_stack":true,"record_shapes":false,"profile_id":"ts-manual"}'

# send representative workload here

curl -X POST http://127.0.0.1:8000/stop_profile
```

For server-side automatic stop, pass `num_steps`. For TokenSpeed-native
EXTEND/DECODE split, pass `profile_by_stage: true`; this produces files with
stage suffixes such as `-EXTEND` and `-DECODE`.

TokenSpeed's benchmark driver can capture traces too:

```bash
tokenspeed bench serve \
  --base-url http://127.0.0.1:8000 \
  --model <model> \
  --dataset-name random \
  --random-input-len 4090 \
  --random-output-len 1 \
  --num-prompts 64 \
  --profile \
  --profile-num-steps 5 \
  --extra-body '{"output_dir":"/path/to/tokenspeed","activities":["CPU","GPU"],"with_stack":true,"profile_id":"ts-bench"}'
```

If `output_dir` is omitted, TokenSpeed falls back to `TOKENSPEED_PROFILER_DIR`
and then `/tmp`.

Use `scripts/probe_llm_server.py` with `--framework tokenspeed` for a small
OpenAI-compatible endpoint probe before or after trace collection:

```bash
python3 scripts/probe_llm_server.py \
  --framework tokenspeed \
  --url http://127.0.0.1:8000 \
  --requests 6 \
  --max-tokens 48
```

## Two-trace capture from running servers

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework sglang \
  --mapping-url http://127.0.0.1:31025 \
  --formal-url http://127.0.0.1:31026 \
  --num-steps 5 \
  --profile-by-stage
```

For `vllm` or `TensorRT-LLM`, use the same shape but pass:

- `--framework vllm` or `--framework trtllm`
- `--mapping-output-dir ...`
- `--formal-output-dir ...`
- `--no-profile-by-stage`

For TokenSpeed, either use `--mapping-url` and `--formal-url` against servers
that expose `/start_profile` and `/stop_profile`, or pass two existing trace
directories with `--mapping-input` and `--formal-input`.

## `profile_by_stage`

`--profile-by-stage` is only meaningful on the SGLang live-capture path.

- With `--profile-workload both` / `prefill` / `decode`, workload directories
  are the stage labels (see the SGLang section).
- On legacy or hand-captured SGLang serving, internal `profile_by_stage` is
  still useful because prefill and decode usually have very different
  bottlenecks.
- On the current profile-v2 path inside SGLang, stage-based profiling is effectively the normal path.
- PD-disaggregated serving adds one extra rule: prefill workers and decode workers must be profiled separately. That is stricter than ordinary `profile_by_stage`.
- For `vllm`, `TensorRT-LLM`, and `TokenSpeed`, disable it with
  `--no-profile-by-stage`.
