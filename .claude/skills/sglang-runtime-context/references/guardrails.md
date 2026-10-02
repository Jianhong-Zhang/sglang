# Guardrails: what fails CI, and what no test catches

## Contents

- [CI guardrails (what to do when they fire)](#ci-guardrails-what-to-do-when-they-fire)
- [Failures no default test catches](#failures-no-default-test-catches)
- [Writing a new guard](#writing-a-new-guard)

## CI guardrails (what to do when they fire)

1. **Strict mutation guard** (always on): bare `server_args.x = ...` after resolution
   raises unconditionally in `ServerArgs.__setattr__` — this *is* the guarantee that
   no writer can desync the bags, so there is no writer ratchet. Projected bags are
   sealed the same way (leaf assignment raises). Fix with the write rule in SKILL.md.
2. **Mutation ratchet** (`test_server_args_mutation_ratchet.py`, exact pin 0 over the whole
   package minus the pipeline / multimodal_gen): textual scan for assignment forms. Never
   raise the baseline.
3. **No-copy contract** (`test_server_args_no_instance_mutation_entry.py`): neither
   `ServerArgs.override` nor `ServerArgs.derive` exists, and nothing in the package
   calls either form. Rerouting a writer to the bags means flipping **all its readers
   in the same commit** (no transitional dual-write).
4. **Legacy-accessor ratchet** (`test_legacy_global_ratchet.py`): `get_global_server_args`
   call sites must not grow. The replacement for a *decision* is a bag leaf, a named
   accessor, or the owning runner's stamp — not `get_server_args().field` (item 5).
   `runtime_context.get_server_args()` is only for the whole-object shapes (dumps,
   provenance, a hand-off to a callee that takes a config).
5. **Global config read ratchet** (`test_global_config_read_ratchet.py`): baselines are
   **0** for both the direct `get_server_args().field` and the alias form (function-local
   — including local copies of an alias, `cfg = sa` — module-level, or parked on an
   instance attribute, plus the `getattr(..., "field")` spelling of each; a name
   computed at runtime or indirection deeper than a local name copy is census-tool
   territory, per the test's docstring). The scanner matches `get_server_args` by its
   literal name — bare or module-qualified (`ctx.get_server_args()`) — and
   `TestNoRenamedAccessorImports` in the same file *bans* `import ... as` renames of it,
   which is what makes literal-name matching sound. Exempt by owner
   module only (`runtime_context.py`, `server_args.py`, `arg_groups/`). Two classes,
   no more: `TestGlobalConfigReadRatchet` holds the two baselines and
   `TestNoRenamedAccessorImports` holds the ban. There is no configured-size registry:
   `get_parallel()` has one spelling per name, so a size read is not a choice between
   two answers and nothing needs registering.
6. **Module-state ratchet** (`test_module_state_ratchet.py`): `global` statements in the
   flag-owning layers are pinned by name. A new module-level runtime global belongs on a
   flags group / resources slot instead; migrating a pinned survivor must shrink the pin.
7. **Namespace coverage** (`test_server_args_namespaces.py`,
   `test_runtime_context_config_bags.py`): every `ServerArgs` field carries `NS(...)`
   metadata and the projected bags must cover the fields exactly (two-way).

## Failures no default test catches

Check these by hand when a change touches config reads, signatures or readbacks.

1. **The other implementations of an interface.** Dropping a parameter means
   auditing implementers, not just callers: `CustomSpecAlgo` is the plugin
   base for speculative algorithms, and the dispatch calls it with the
   built-in's argument list. Nothing in the tree implements it, so only a
   plugin user hits the `TypeError`.
   Guard: `test_plugin_hook_signatures.py`.
2. **Publish order inside a process entry, not per file.** A file containing a
   `publish` says nothing about whether a given read runs before it. Spawned
   workers (`MMEncoder` for encoder DP/TP, the Ray scheduler actor) start with
   an empty context, so a bag read above the publish raises only there.
   Guard: `test_publish_precedes_bag_reads.py`.
3. **The role namespace a process publishes under.** `ROLE_NAMESPACE_SETS`
   narrows what each role may read; the DP controller is audited for `exec`
   alone. A helper that reaches for another namespace passes every default-mode
   test and aborts startup under `SGLANG_ROLE_NAMESPACES=enforce`. Prefer
   answering from the caller's own namespaces over widening the set.
4. **Sibling surfaces of a readback.** Changing what one entry point reports
   means enumerating the others: HTTP, gRPC and in-process `Engine` each have
   their own server-info and model-info, and each passes its own tests while
   its users lose the field.
   Guard: `test_effective_state_surfaces.py`.
5. **The accessor name itself.** A bag accessor is a module function of no arguments,
   bound once per module. Called as an object member (`manager.get_disagg()`,
   `self.get_disagg = get_disagg`), or shadowed by a same-named import
   (`from model_loader import get_model` next to the model bag, where the later
   import silently wins and the loader call gets a zero-argument bag), it imports
   fine and fails only when that path runs. `ruff --select F811` catches the import
   collision; the member-call shapes are an `AttributeError` at call time only because
   `RuntimeContext` has no bag-named member and no `__getattr__` — a delegating
   `__getattr__` would make them silent, and then this needs a guard rather than a rule.

## Writing a new guard

Write guards over a **derived** set, never a hand-kept list: an entry naming a function
that no longer exists (an entry-point row for a method the class does not have), or a
field list missing the one field nobody migrated, passes green forever. Both stay
invisible when the assertion has slack (`>= len(...) - 1`) or compares key names
instead of value sources.
