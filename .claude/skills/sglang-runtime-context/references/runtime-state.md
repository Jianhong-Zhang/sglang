# Runtime flags, resources and per-forward state

## Runtime flags (`get_flags()`)

For state that init-time code *derives* and runtime code reads — parsed enums, platform
probes, swappable ACTIVE values. Not for config mirrors (read the bag leaf instead).

- Groups are typed dataclasses on `Flags` (`capture` / `moe` / `dp`): typo-safe writes,
  transactional test-only `override(**kw)` context manager.
- `flags.moe` is materialized by `initialize_moe_config()` at scheduler init (it
  reads `exec.moe` / `spec` / `model`, and takes no record);
  accessors (`get_moe_a2a_backend` etc.) are thin shims with lazy defaults. The speculative
  contexts (`speculative_moe_backend_context`) swap the ACTIVE leaves around draft forwards.
- `flags.dp` is materialized by `initialize_dp_attention`; `is_dp_attention_enabled()` is a
  shim over `flags.dp.enabled`.
- Adding a leaf: declare the dataclass field with a default equal to the pre-init behavior,
  materialize it at the owning subsystem's init, keep any public accessor as a shim.

## Resources (`get_resources()`)

Named slots + two keyed-lazy registries:

- `get_stream(name)` — get-or-create a named CUDA side stream; `set_stream(name, stream)`
  installs explicitly. **Name leases by subsystem ROLE**: all model alternate streams share
  `"alt"`; the offloader's copy stream is `"offload"`; DP-TBO comm is `"dp_tbo_comm"`; LoRA
  side stream is `"lora_side"`. Two call sites may share a name only if their work belongs
  on one stream — sharing across roles serializes intended overlap.
- `get_buffer(name, factory)` — get-or-create a named persistent buffer. Grow-only or
  per-device semantics manage their `resources.buffers` entries directly (see tokenspeed /
  SM120 split / Marlin workspace). Buffer names are per-backend; do not silently
  share.
- Singletons with manager semantics (EP dispatcher buffers, EPLB recorder/metadata, graph
  memory pool) keep their owning accessors/classes as facades; only the *state* lives in a
  resources entry. Preserve exact semantics in the shim: lazy defaults (the EPLB recorder
  defaults to a Noop instance, not None), publish-once asserts, event-reuse contracts.
- Stream/buffer creation is a driver call — it must happen outside cuda-graph capture;
  keep lease points at init/warmup time.

## Per-forward flags (`get_forward()`)

Contextvar-backed; a new thread sees the defaults; `scoped(**kw)` is the regular write path
(transactional, restores on exit and on exception); `set(name, value)` exists for legacy
sticky setters (`is_extend_in_batch` is intentionally sticky within a thread). Use this
tier for anything set-per-forward and read-within-forward. Before adding cross-thread
state here, prove the readers' thread affinity: contextvars do NOT propagate to already-
running or newly spawned threads. Note TBO ("two-batch overlap") interleaves ubatches on
ONE thread — do not design for TBO threads that don't exist.
