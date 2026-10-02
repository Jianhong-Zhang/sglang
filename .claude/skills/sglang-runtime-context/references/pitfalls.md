# Pitfalls to check before and while refactoring

- **Moving code drops first-line guards**: early returns (`if self.is_draft_worker: return`)
  are the easiest thing to lose when relocating a method body. A draft is built from
  the target's published config — there is no draft config copy and no nested publish —
  so a body moved out of a draft-aware call site keeps reading the target's
  bags, and only that guard tells the two apart. What the draft build *does* scope is
  narrower and named: `draft_model_build_scope()` for the MoE fusion gates,
  `speculative_moe_backend_context()` for the runner backends.
- **Late function-scope imports shadow module names** for the WHOLE function
  (UnboundLocalError at earlier lines). Audit moves with AST, not grep.
- **Storage matrix for state read inside torch.compile-traced model code**
  (piecewise cuda graph compiles the whole model forward): contextvars are
  untraceable (hard error); dict-slot values are guarded per value — for a
  per-forward int that is one recompile per distinct size, straight into the
  recompile limit; **class/instance attributes are the only compile-friendly
  form** (attribute-source ints get automatic-dynamic after the first size
  change). Bools (≤2 values) are tolerable in any form — see
  `ForwardFlags._GRAPH_VISIBLE`. Config-bag leaves are real instance attributes for
  exactly this reason. Parallel leaves are the exception, measured rather
  than assumed: they come through `ParallelContext.__getattr__`, which traces
  under `torch.compile(fullgraph=True)` (`object.__getattribute__` is the form
  that graph-breaks, and it is not on this path). Before moving such state,
  prove its readers sit outside compile coverage; a piecewise-prefill boot of a small
  model is the fast check (recompile storms show as `torch._dynamo hit
  config.recompile_limit` during the compile pass).
- **Engine-booting e2e tests are the only coverage for launcher-path code**; a child crash
  kills the process tree and pytest dies silently — run with `PYTHONUNBUFFERED=1` and read
  child logs.
- CI arms `SGLANG_ENABLE_ASYNC_ASSERT=1` (device-side `torch._assert_async` probes, e.g.
  KV-cache OOB): a fired device assert kills the tree with no Python traceback, and the
  same bug is *silent corruption* locally with the flag off. Arm it when reproducing CI
  crashes.
- CI startup logs print the full `server_args=ServerArgs(...)`; diffing that dump between
  runs is the fastest config-divergence check.
