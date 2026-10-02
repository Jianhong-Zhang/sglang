# Testing against the runtime context

- **Force a code path by overriding causes, not effects**: compose
  `get_context().override_server_args(**fields)` (publishes a fresh dummy-boundary
  `ServerArgs` carrying the overrides AND projects the bags — `with`-scoped, or
  `install()`/`restore()` + `addCleanup` for fixture-lifetime use) +
  `get_<ns>().override(...)` (scoped override of one bag's own leaves) +
  `get_parallel().override(...)` (live topology) + `get_flags().<group>.override(...)` +
  `get_forward().scoped(...)`. All are scoped and transactional. Tests control execution
  through the context — do not hand-build and publish config objects.
- **Test doubles publish, they do not inject.** A stand-in that carries
  `server_args=SimpleNamespace(field=...)` stops working the moment production reads
  the bag. Never monkeypatch import bindings (`module.get_x = lambda: ...`) either: a
  faked accessor silently stops intercepting after any reader migration. Seed the value
  with `override_server_args`, which publishes only once it is entered or installed —
  the bare call just builds the override:

  ```python
  override = get_context().override_server_args(field=...)
  override.install()
  self.addCleanup(override.restore)      # or: with get_context().override_server_args(...):
  ```

  Then adjust bag leaves with the scoped bag `override` where the constructed
  `ServerArgs` cannot carry the value (e.g. `get_device().override(device="meta")`).
  Prefer this even where a single-accessor stub would work: it expresses the *cause*
  (the configuration) rather than pinning one helper's answer, and it keeps working
  when a reader migrates between the accessor and the leaf. For example,
  `test_attention_patching.py` publishes the non-lazy strategy, and
  `test_kimi_k3_vision.py` publishes `tp_size` and forces the live topology through
  `get_parallel().override`. Stubbing one *named accessor* is a last resort for a case
  that isolates one branch of one helper where no published config can reach it — if
  you do it, say so in the test.
- Mocked runners/managers still need the **per-runner instance attributes** the code
  under test reads (`kv_cache_dtype_str`, `server_args` for whole-object passes) — set
  them explicitly on the mock; `MagicMock(spec=...)` raises on attributes that only
  exist post-`__init__`, which is the fastest way to find a missed stub.
- `reset_context()` in teardown when a test publishes outside a scoped override.
- `ServerArgs(model_path="dummy")` early-returns the pipeline (few declarations, no
  strict guard) — fine for lightweight fixtures.
- **Asserting what resolution decided reads `resolution_result(sa, "field")`**, not
  `sa.field`: the field is the raw input. Assert the field only when the point of
  the case *is* that the record stayed pristine (the FA4 page-size and waterfill
  cases do exactly that, and say so).
- **Run changed test files per-file** (own process), the way CI does: a monolithic local
  pytest run lets a context published by an earlier file mask a missing-publish bug in a
  later one.
- Never module-skip a test "until the migration settles" — seed the context instead.
