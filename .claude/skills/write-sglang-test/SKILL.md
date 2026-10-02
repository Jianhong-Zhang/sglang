---
name: write-sglang-test
description: Writes and registers SGLang CI and unit tests — CustomTestCase, register_*_ci registration and suite choice, server fixtures, model selection, mock-based unit tests, and where each kind of test lives under test/registered/. Use when creating new tests, adding CI test cases, writing unit tests or kernel tests, or when the user asks to add tests for SGLang features.
---

# Writing SGLang CI / UT Tests

This skill covers **how to write and register tests**. Read `test/README.md` first for the full CI layout, how to run tests, and extra tips. For CI pipeline internals (stage ordering, fast-fail, gating, partitioning, debugging CI failures), see the `ci-workflow-guide` skill.

## Core Rules

1. **Always use `CustomTestCase`** — never raw `unittest.TestCase`. It ensures `tearDownClass` runs even when `setUpClass` fails, preventing resource leaks in CI.
2. **`tearDownClass` must be defensive** — use `hasattr`/null checks before accessing resources (e.g. `cls.process`) that `setUpClass` may not have finished allocating.
3. **Place tests under `test/registered/`** — the directory for each kind of test is in [Test Placement](#test-placement); nested subfolders are allowed.
4. **Reuse server fixtures** — inherit from `DefaultServerBase` or write `setUpClass`/`tearDownClass` with `popen_launch_server`
5. **Prefer mock over real server** — when testing logic that doesn't need a server / engine launch (middleware, request routing, config validation, argument parsing), use `unittest.mock.patch` / `MagicMock` and place tests in `test/registered/unit/`. Only launch a real server when the test genuinely needs inference results or server lifecycle behavior.

For adding or updating a kernel under `python/sglang/kernels/jit/`, use the `add-jit-kernel` skill first.

---

## Model & Backend Selection

| Scenario | Model | CI Registration | Suite |
|----------|-------|-----------------|-------|
| **Unit tests** (no server / engine launch) | None | `register_cpu_ci` (prefer) or `register_cuda_ci` | `base-a-test-cpu` or `base-b-test-1-gpu-small` |
| **Common / backend-independent** (middleware, abort, routing, config, arg parsing) | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (1B) | `register_cuda_ci` only | `base-b-test-1-gpu-small` |
| **Model-agnostic functionality** (sampling, session, OpenAI API features) | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (1B) | `register_cuda_ci` (+ AMD if relevant) | `base-b-test-1-gpu-small` |
| **General performance** (single node, no spec/DP/parallelism) | `DEFAULT_MODEL_NAME_FOR_TEST` (8B) | `register_cuda_ci` | `base-b-test-1-gpu-large` |
| **Bigger features** (spec, DP, TP, disaggregation) | Case by case | Case by case | See [references/ci-suites.md](references/ci-suites.md) |

**Key principle for E2E tests**: Do NOT add `register_amd_ci` / `register_npu_ci` / `register_xpu_ci` unless the test specifically exercises that backend's code paths. Common E2E tests just need any GPU to run — duplicating across backends wastes CI time with no extra coverage.

### All model constants

Defined in `python/sglang/test/test_utils.py`:

| Constant | Model | When to use |
|----------|-------|-------------|
| `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` | Llama-3.2-1B-Instruct | Common features, model-agnostic tests |
| `DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE` | Llama-3.2-1B | Base (non-instruct) model tests |
| `DEFAULT_MODEL_NAME_FOR_TEST` | Llama-3.1-8B-Instruct | General performance (single node) |
| `DEFAULT_MOE_MODEL_NAME_FOR_TEST` | Mixtral-8x7B-Instruct | MoE-specific tests |
| `DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST` | — | Embedding tests |
| `DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST` | — | Vision-language tests |

### Naming Conventions

A per-commit suite name is **generated** from registration metadata as `{stage}-test-{runner_config}` — you don't hand-write it:

- **`stage`** — the CI stage (e.g. `base-b`, `base-b-kernel-unit`, `base-c`).
- **`runner_config`** — a runner-pool key from `scripts/ci/runner_configs.yml`, which maps it to the physical runner label (so `1-gpu-large` runs on `1-gpu-h100`). AMD/NPU use their own keys (e.g. `amd`).
- **Suite** — `register_cuda_ci(stage="base-b", runner_config="1-gpu-small")` → `base-b-test-1-gpu-small`, the name you pass to `run_suite.py --suite`. The `-test-` is just the connector; never put it in `register_*_ci`.

> CUDA nightly uses the same shape with `stage="nightly"` (e.g. `stage="nightly", runner_config="1-gpu-large"` → `nightly-test-1-gpu-large`) and **no** `nightly=True` — the stage name carries the cadence, and setting the flag makes the test silently never run. Legacy single-string `suite=` is left only for `stress` and some AMD/CPU/NPU pools.

### All CI Suites

The full suite list per backend (CUDA, AMD, NPU, XPU, nightly) is in [references/ci-suites.md](references/ci-suites.md).

### Choosing a Suite

Use the lightest suite that meets your test's needs:

- **No GPU required** → `base-a-test-cpu`
- **Most small GPU tests** → `base-b-test-1-gpu-small` (default choice)
- **Need H100 memory or Hopper features** → `base-b-test-1-gpu-large`
- **JIT kernel correctness** → `base-b-kernel-unit-test-1-gpu-large`
- **JIT kernel correctness for B200 / SM100 paths** → `base-b-kernel-unit-test-4-gpu-b200`
- **JIT kernel benchmarks** → `base-b-kernel-benchmark-test-1-gpu-large`
- **Multi-GPU** → only when the test actually needs multiple GPUs

---

## Test File Templates

### Unit Tests (no server / engine launch)

See `test/registered/unit/README.md` for quick-start and rules. Unit tests live in `test/registered/unit/`, mirroring `python/sglang/srt/`:

```python
"""Unit tests for srt/<module>"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.<module> import TargetClass
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
# Prefer CPU. Only use register_cuda_ci when the test truly needs a GPU.

class TestTargetClass(CustomTestCase):
    def test_basic_behavior(self):
        obj = TargetClass(...)
        self.assertEqual(obj.method(), expected)

    @patch("sglang.srt.<module>.some_dependency")
    def test_with_mock(self, mock_dep):
        mock_dep.return_value = MagicMock()
        # test logic with dependency mocked
        ...


if __name__ == "__main__":
    unittest.main()
```

Use `unittest.mock.patch` / `MagicMock` to mock dependencies and isolate the logic under test. If the module transitively imports GPU-only packages (e.g. `sgl_kernel`), they can be stubbed so the test runs on CPU CI. Do not modify `sys.modules` at module level — use `patch.dict` (as a class decorator or with `start`/`stop`) to ensure cleanup and avoid cross-test pollution. See `test/registered/unit/README.md` for details and examples.

**Quality bar** — test real logic (validation boundaries, state transitions, error paths, branching, etc.). Skip tests that just verify Python itself works (e.g., "does calling an abstract method raise `NotImplementedError`?", "does a dataclass store the field I assigned?"). Consolidate repetitive patterns into parameterized tests. No production code changes in test PRs.

E2E templates (small-model and 8B performance) and the eval-accuracy mixins are in [references/templates.md](references/templates.md).

---

## Server Fixture Reuse

For tests that only need a standard server, inherit from `DefaultServerBase` and override class attributes:

```python
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

class TestMyFeature(DefaultServerBase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    other_args = ["--enable-my-feature"]

    def test_something(self):
        ...
```

Available fixtures in `python/sglang/test/server_fixtures/`:

| Fixture | Use case |
|---------|----------|
| `DefaultServerBase` | Standard single-server tests |
| `EagleServerBase` | EAGLE speculative decoding |
| `PDDisaggregationServerBase` | Disaggregated prefill/decode |
| `MMMUServerBase` | Multimodal VLM tests |

---

## CI Registration

Every CI-discovered test file must call a registration function at module level:

```python
from sglang.test.ci.ci_register import (
    register_cuda_ci,
    register_amd_ci,
    register_cpu_ci,
    register_npu_ci,
    register_xpu_ci,
)

# Per-commit test (small 1-gpu, runs on 5090)
register_cuda_ci(est_time=80, suite="base-b-test-1-gpu-small")

# Per-commit test (large 1-gpu, runs on H100)
register_cuda_ci(est_time=120, suite="base-b-test-1-gpu-large")

# Nightly-only test (same shape as per-commit, stage is just "nightly")
register_cuda_ci(est_time=200, stage="nightly", runner_config="1-gpu-large")

# Multi-backend test (only when testing backend-specific code paths)
register_cuda_ci(est_time=80, suite="base-a-test-1-gpu-small")
register_amd_ci(est_time=120, suite="stage-a-test-1-gpu-small-amd")
register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)

# Temporarily disabled test
register_cuda_ci(est_time=80, suite="base-b-test-1-gpu-small", disabled="flaky - see #12345")
```

Parameters:
- `est_time`: estimated runtime in seconds (used for CI partitioning)
- `suite`: which CI suite to run in (see [references/ci-suites.md](references/ci-suites.md))
- `nightly=True`: legacy cadence flag, for non-CUDA nightly suites only. CUDA nightly uses `stage="nightly"` and must leave this unset
- `disabled="reason"`: temporarily disable with explanation

`register_xpu_ci` (Intel XPU) takes the same parameters.

### Kernel Test Registration

Kernel tests and benchmarks are ordinary files under `test/registered/` and register the same way, with the kernel stages:

```python
from sglang.test.ci.ci_register import register_cuda_ci

# Correctness tests in test/registered/kernels/ops/<group>/
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="8-gpu-h200")

# Benchmarks in test/registered/kernels/benchmark/<group>/
register_cuda_ci(est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")

# Optional nightly registration — same form, stage is just "nightly"
register_cuda_ci(est_time=120, stage="nightly", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, stage="nightly", runner_config="8-gpu-h200")
```

Suite names follow the Naming Conventions above. Keep `est_time`, `stage`, `runner_config`, and `suite` as **literal values** — `run_suite.py` collects them by AST parsing. A `register_*_ci(...)` call under `python/sglang/` is rejected by the `check-no-registered-tests-in-package` pre-commit hook.

---

## Test Placement

```
test/
├── registered/          # CI tests (auto-discovered by run_suite.py)
│   ├── unit/            # No server / engine launch (see test/registered/unit/README.md)
│   ├── kernels/
│   │   ├── ops/<group>/       # Kernel correctness (no server, GPU required), incl. JIT kernels
│   │   └── benchmark/<group>/ # Kernel benchmarks, incl. JIT kernels
│   ├── sampling/        # test_penalty.py, test_sampling_params.py ...
│   ├── sessions/        # test_session_control.py ...
│   ├── openai_server/   # basic/, features/, validation/ ...
│   ├── spec/            # eagle/, utils/ ...
│   ├── models/          # model-specific accuracy tests
│   ├── perf/            # performance benchmarks
│   └── <category>/      # create new category if needed
├── manual/              # Non-CI: debugging, one-off, manual verification
└── run_suite.py         # CI runner (scans registered/)
```

**Decision rule** (see also `test/registered/README.md`):
- Component logic, no server → `registered/unit/`
- Kernel correctness → `registered/kernels/ops/<group>/`; kernel benchmarks → `registered/kernels/benchmark/<group>/`, `<group>` mirroring the op's group under `python/sglang/kernels/ops/`. Only test-only helpers (e.g. `benchmark/marker.py`) stay under `python/sglang/kernels/jit/`. AOT `sgl-kernel` tests are the exception: plain pytest under `python/sglang/kernels/aot/tests/`, unregistered (see `add-sgl-kernel`).
- Server needed → `registered/<category>/`
- Local debugging → `manual/`

---

## Key Utilities

```python
from sglang.test.test_utils import (
    CustomTestCase,              # base class with retry logic
    popen_launch_server,         # launch server subprocess
    DEFAULT_URL_FOR_TEST,        # auto-configured base URL
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,  # 600s default
    run_bench_serving,           # benchmark helper (launch + bench)
)
from sglang.srt.utils import kill_process_tree  # cleanup server
```

---

## Checklist

Before submitting a test:

- [ ] Inherits from `CustomTestCase` (not `unittest.TestCase`)
- [ ] Has `register_*_ci(...)` call at module level
- [ ] Placed per the [Test Placement](#test-placement) decision rule
- [ ] Backend-independent tests: `register_cuda_ci` only + smallest model
- [ ] Logic that doesn't need a server / engine launch → unit test in `registered/unit/` (see Unit Tests section)
- [ ] `setUpClass` launches server, `tearDownClass` kills it (if server-based)
- [ ] `tearDownClass` is defensive — uses `hasattr`/null checks before accessing resources that may not have been allocated
- [ ] Has `if __name__ == "__main__": unittest.main()`
- [ ] `est_time` is reasonable (measure locally)
