---
name: sglang-runtime-context
description: Explains how SGLang's runtime configuration and process-global state are organized (RuntimeContext tiers, publish + namespace config bags, the pristine ServerArgs seed, override entry points, resource/stream/buffer leases, per-forward flags), the CI guardrails that enforce the design, and the idioms for developing and testing against it. Use before touching server_args, model overrides, module-level state, or per-forward state in sglang, or to tell whether a launch failure is a config-resolution error or a runtime one.
---

# SGLang runtime-context architecture

One container owns process-static runtime state: `sglang.srt.runtime_context.RuntimeContext`
(a process singleton reached via `get_context()`). Everything below is a tier on it.

| Tier | Accessor | Holds | Lifecycle |
|------|----------|-------|-----------|
| raw config seed | `get_server_args()` | the published `ServerArgs` — the startup record, for debugging, dumps and provenance. Business code does not read fields off it (see "Reading config") | published at process entry; re-publish is **last-publish-wins** (the tokenizer publish in the launcher process; sequential engine rebuild in one process, e.g. unit tests) and re-projects the bags; read-only |
| resolved config | `get_exec()` `get_memory()` `get_schedule()` `get_model()` `get_spec()` `get_serving()` `get_observability()` `get_disagg()` `get_lora()` `get_mm()` `get_device()` | namespace **config bags** — the single source of truth for resolved config; leaves are real attributes (dynamo-traceable). Each is a **module function of no arguments**, bound once per module (failure shapes: references/guardrails.md) | projected at `publish` from the declarations over `server_args`' raw fields; mutated only via `get_context().override` |
| runtime flags | `get_flags()` | state that is *not* a pure function of config: `capture` (cuda-graph lifecycle), `moe` (ACTIVE backends, swappable), `dp` (DP-attention runtime flags) | materialized at subsystem init; groups offer `override()` for tests |
| resources | `get_resources()`, `get_stream(name)`, `get_buffer(name, factory)` | process-level handles: graph pools, EPLB state, EP dispatcher state, named side streams, workspace buffers | lazy; cleared by `reset_context()` |
| per-forward | `get_forward()` | forward-scoped flags (multi-stream switch, MoE output buffer, attn-TP inputs, extend-in-batch) | contextvar-backed; `scoped(**kw)` restores on exit; new threads see defaults |
| parallel | `get_parallel()` | one spelling per name: ranks and group handles are the live topology (`@property`, read-through); every other name, sizes included, is a leaf of the parallel config bag | ranks/groups: after dist init; leaves: after publish |

`reset_context()` (unit-test teardown) drops the published config and installs fresh
flags/resources/forward tiers.

## Writing config: one rule

`ServerArgs` holds the raw input and is never written after resolution (the strict
`__setattr__` guard raises). Pick the write by when the value is known:

- **During resolution** (`__post_init__`) → declare (`self._declare` /
  `declare_resolution`); model code declares through `arg_groups/overrides.py`.
- **Launcher stage, before publish** (LoRA normalization, chat-template parser
  detection) → `declare_late_resolution(server_args, source, **fields)`.
- **After publish** → `get_context().override(source, **fields)`, the only post-publish
  entry point. It writes the bag leaves in place and logs provenance; the record stays
  pristine. Control-plane changes go through its wrapper
  `TokenizerManager.record_config_updates`.
- **A value one runner or worker owns** → a constructor argument or a runner
  attribute, never a config copy or a bag leaf.

Details, and why an override cannot replace a declaration:
[references/config-resolution.md](references/config-resolution.md).

## Reading config

`get_server_args().field` in business code is a ratchet failure: a field read there
answers with what the operator typed, not with what resolution decided. Read:

- **a resolved leaf** → its namespace bag (`get_exec().moe.moe_runner_backend`,
  `get_schedule().chunked_prefill_size`, …). Bag-backed reads — a leaf directly, or
  a bag-derived accessor below — are what see post-publish overrides. Only the
  instance-derived accessors (the ones with no leaf to read) answer from the
  startup record and therefore do not.
- **a leaf the caller names at runtime** (a readback endpoint, a control-plane handler
  reporting a list of fields) → `get_context().config_leaf(name)`, the read side of
  `override`; it resolves the name through `NS` and raises on a non-leaf. A call site
  that knows its field reads the bag leaf; `config_leaf` is not a way around the ratchet.
- **the live topology or a parallel size** → `get_parallel()` (bare names). The MoE
  communicator's width (`get_moe_cp_size()`) and the effective DCP topology have their
  own spelling: [references/parallel.md](references/parallel.md).
- **a value derived from published leaves** → an accessor in `runtime_context` that
  derives it *from the bags*: `mamba_extra_buffer_enabled()` /
  `mamba_extra_buffer_lazy_enabled()` read `get_memory()` and `get_exec()`, so
  they see post-publish overrides. Prefer this shape whenever the inputs are
  leaves; the same-named `ServerArgs` members are the pre-publish equivalents the
  resolution pipeline uses, and wrapping one of those instead would quietly cost
  you override visibility. `is_ep_joiner()` / `is_ep_scale_joiner()` are the same
  shape over `exec.moe.ep_join_mode`, `attention_backends()` derives the
  `(prefill, decode)` pair from the three `exec.kernel` leaves, and
  `max_speculative_num_draft_tokens()` / `cutedsl_moe_max_num_tokens()` derive
  theirs from `spec` / `schedule` / `exec.graph`.
- **a value only the instance can compute** → the named accessor in
  `runtime_context`, which is the one module allowed to read the slot:
  `mamba_cache_chunk_size()`, `mamba_state_chunk_size()`, `uses_mla_backend()`,
  `process_model_config()`. These have no leaf to read — they combine several fields,
  the HF config, or a property with no bag of its own. A new derived member gets an
  accessor here rather than call sites reaching for the record, and only when the
  bag-derived shape above cannot express it.
- **this runner's resolved value** → the runner
  (`prefill_attention_backend_str`, `kv_cache_dtype_str`,
  `draft_attention_backend`, `num_fused_shared_experts` on the model).
- **a `ServerArgs` you were handed** (whole-object passes, `GrammarManager`,
  `MMEncoder`) → the supplied-instance rules in
  [references/config-resolution.md](references/config-resolution.md#reads-that-legitimately-stay-on-a-serverargs-instance).
- **inside the resolution pipeline** → a view (`resolving_view` / `resolved_view`),
  never a direct field read; see
  [references/config-resolution.md](references/config-resolution.md#mid-resolution-reads-inside-the-pipeline-only).

## References

- [references/config-resolution.md](references/config-resolution.md): publish and roles,
  `NS` metadata, post-publish and late changes, per-runner values, supplied-instance
  reads, mid-resolution views, model-specific adjustments, load-time vs resolution-time.
- [references/parallel.md](references/parallel.md): `get_parallel()` names, the
  `_MOE_DP` alias, scoped TP groups, elastic EP, the DCP pair, `ParallelState`.
- [references/runtime-state.md](references/runtime-state.md): `get_flags()` groups,
  stream/buffer leases, per-forward contextvars.
- [references/testing.md](references/testing.md): overriding causes not effects, test
  doubles that publish, mocks, per-file runs.
- [references/guardrails.md](references/guardrails.md): the CI ratchets and what to do
  when one fires, the five failures no default test catches, how to write a new guard.
- [references/pitfalls.md](references/pitfalls.md): what breaks when moving code
  (dropped guards, import shadowing, torch.compile storage, e2e-only coverage, CI-only
  asserts).

## Where to read the code

Key source files: `python/sglang/srt/runtime_context.py` (the container, every tier,
`publish`, `_ConfigBag`, `override_server_args`),
`python/sglang/srt/arg_groups/overrides.py` (override registry, passes,
`declare_late_resolution`), `python/sglang/srt/server_args.py` (`NS` metadata,
`Arg(..., resolvable=True)`, `__setattr__` strict guard), and the guardrail tests under
`test/registered/unit/` (`test_server_args_mutation_ratchet.py`,
`test_global_config_read_ratchet.py`, `test_legacy_global_ratchet.py`,
`test_module_state_ratchet.py`, `test_server_args_namespaces.py`,
`test_runtime_context.py` — the last one doubles
as executable documentation of every tier's semantics).
