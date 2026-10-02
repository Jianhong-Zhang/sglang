# Config resolution: publish, bags, post-publish changes and instance reads

## Contents

- [Publish and the namespace bags](#publish-and-the-namespace-bags)
- [Post-publish and late changes](#post-publish-and-late-changes)
- [Per-runner values are constructor arguments](#per-runner-values-are-constructor-arguments)
- [Reads that legitimately stay on a ServerArgs instance](#reads-that-legitimately-stay-on-a-serverargs-instance)
- [Mid-resolution reads (inside the pipeline only)](#mid-resolution-reads-inside-the-pipeline-only)
- [Adding a model-specific config adjustment](#adding-a-model-specific-config-adjustment)
- [Load-time vs resolution-time](#load-time-vs-resolution-time)

## Publish and the namespace bags

**`ServerArgs` holds the raw input and nothing else. Resolution writes no field:
it declares, and the declarations are what the namespace bags are projected from.**

- Every publishing process entry calls `publish(server_args, role=...)`
  (`run_scheduler_process`, the Ray `SchedulerActor`, the DP controller, tokenizer,
  detokenizer, encoder, weight-cache daemon, the multi-tokenizer worker, the
  spawned encoder TP/DP workers, the benchmark work functions, ...); constructors
  do not publish — `ModelRunner`, `TokenizerManager` and `MMEncoder` call
  `assert_published` and fail loudly if an entry forgot. The roles are enumerated once,
  as the keys of `ROLE_NAMESPACE_SETS` — there is no `launcher` role, the launch
  path publishes as `tokenizer`. The remaining non-publisher is
  `run_multi_detokenizer_router_process`: it *is* handed a `ServerArgs`, and uses
  it only for `configure_logger(server_args)`, so it has nothing to publish
  for — a bag read added under that entry needs a `publish` at the entry first.
- `publish` projects the config bags from the declarations over the record's raw
  fields; the accessors (`get_exec()` etc.) fail closed before it runs.
- `role` records which process type published, and keys per-role namespace
  enforcement: `SGLANG_ROLE_NAMESPACES=record` audits which namespaces each role's
  process actually reads (per-pair persisted via `SGLANG_ROLE_NAMESPACES_OUT`;
  reads inside torch.compile-traced code are NOT observed — audit with
  compilation disabled before restricting a role), and
  `=enforce` fails closed on bag reads outside the role's `ROLE_NAMESPACE_SETS` entry
  (`None` = full tree; only audited roles are restricted).
- Bag membership is metadata on the dataclass: every `ServerArgs` field carries
  `NS("path")` (e.g. `NS("exec.moe")`); coverage is linted two-way
  (`test_server_args_namespaces.py`, `test_runtime_context_config_bags.py`).
- There is **no write-through** from an override to the `ServerArgs` instance, and no
  in-place mutation entry on the instance at all: it is read-only after resolution.

## Post-publish and late changes

- **Post-startup control-plane changes** — a weight update, a HiCache mirror
  attach, a parser resolved from the chat template — go through
  `TokenizerManager.record_config_updates(source, **fields)`, a named wrapper
  over `get_context().override`. One process keeps one log: the request dumps
  ship `get_context().overrides_log()`, and `config_value(name)` /
  `resolved_config_dict(base)` answer from the bags. The exposure ratchet
  resolves the wrapper, so a field recorded through it joins the post-publish
  override surface exactly like a direct `override` and needs the same ordering
  judgment against any supplied-instance read of it
  (`test_supplied_instance_exposure_ratchet.py`).
- **`model_path` and `served_model_name` are answered off the manager.** Both are
  `NS` leaves and `override` accepts them, but the tokenizer-side weight reload
  records only `load_format` and writes the two path fields as `TokenizerManager`
  attributes (`_MANAGER_OWNED_FIELDS`); `config_value` and `resolved_config_dict`
  overlay them on top of the bags. Bags do not cross a process boundary,
  so recording those two in the tokenizer process would leave every other
  process's bag on the old path while the log claimed a process-wide change. The
  scheduler rewrites its own copy where the reload happens —
  `ModelRunner.update_model_fields` overrides `model_path` / `load_format` for
  the target runner.
- **Late launcher-stage resolution (pre-publish)**: a few rules cannot run inside
  `__post_init__` — LoRA normalization, and the auto-parser detection that needs a
  tokenizer/chat-template load. They are resolution, not mutation, and they
  **declare** via `arg_groups.overrides.declare_late_resolution(server_args,
  source, **fields)`, which refuses the published instance. The declaration lands
  in the stash on that very object, so every holder of it carries the decision —
  the HTTP server, the multi-tokenizer workers it is serialized for, the
  schedulers it forks — and each of them publishes bags projected from it. The
  fields stay the operator's input; `resolution_result(sa, field)` and the bags
  are what answer for the decision. Returning a variant here is a bug: the
  launcher rebinds its local and everyone else keeps the unresolved object.

## Per-runner values are constructor arguments

A value another runner / worker owns is a constructor argument, not a config
copy. The draft worker's `context_length`, load format and attention backend
travel as arguments to `TpModelWorker` / `ModelRunner` and live on the runner
(`ModelRunner.draft_attention_backend`, `kv_cache_dtype_str`, …); the encoder
DP worker's device is `MMEncoder(gpu_id=...)`. There is no `ServerArgs.derive`:
a config object is never copied-and-edited; test doubles that need a
modified copy use `sglang.test.test_utils.server_args_variant`.

**Why a bag override cannot stand in for late resolution or per-runner
construction.** The bags are projected at publish *from the declarations over the
instance's raw fields*, so anything the runtime must read has to be declared before
publish — an override afterwards puts instance and bags back out of agreement, and
whole-object readers (`ModelConfig.from_server_args`, `build_load_config`,
`MMEncoder`'s own `self.server_args.X`) never see it. And bags do not cross a process
boundary: a child publishes from the object it receives and re-projects its own bags,
so a parent-side override is lost. Values that feed construction before any bag exists
(group init reads `server_args.tp_size`) have no bag to override at all.

## Reads that legitimately stay on a ServerArgs instance

`self.server_args.field` is right only for handed per-instance config. The allow-list
is `GrammarManager` and `MMEncoder`.

- **Per-runner values** — there is no per-runner `ServerArgs`. Every worker
  (`TpModelWorker`, the draft workers in `speculative/`) is handed the *same*
  instance the process published, so a bag leaf is the decision and
  `self.server_args.X` is the operator's input — a post-publish `override` moves only
  the bag, which is why a field that is process-wide config (`attention_backend`,
  `skip_tokenizer_init`, `kv_cache_dtype`) reads from the bags like any other, and why
  a residual instance read on this path is stale the moment someone overrides that leaf.
  What is genuinely per-runner travels two ways, neither of them a config
  instance: **constructor arguments** (`ModelRunner(draft_attention_backend=...)`,
  `MMEncoder(gpu_id=...)`) and **runner attributes holding the resolved value**
  (`model_runner.kv_cache_dtype_str`, `prefill_attention_backend_str`,
  `num_fused_shared_experts`, `linear_attn_backends`) — threaded to consumers as
  arguments, never backfilled onto a shared object. A per-runner choice also stays
  *out* of the bags: recording it there is how a second runner inherits the first
  one's answer. The one sanctioned bend in that rule is *scoped*:
  `ModelRunner._load_format_scope` exposes the draft's
  `--speculative-draft-load-format` through `get_model().override(load_format=...)`
  for exactly the duration of the draft build, because model construction
  reads that bag leaf — the override restores on exit, so nothing outlives
  the scope. When there is a runner in hand, read its stamp; that is a different rule
  from "read the instance".
- **Per-instance boundaries** — the tokenizer-manager family, everything under
  `entrypoints/`, and the tokenizer-process multimodal processors read the bags. A
  process holds at most one live config at a time (concurrent multi-Engine is
  unsupported; sequential rebuild stays legal, and unit tests rely on it), so "several
  `Engine`s share one process" is no reason to read the instance. The exposure
  ratchet's pin set is empty, so the next such read is a new entry that has to argue
  for itself; read the ratchet for the current set rather than assuming a directory is
  off-limits. What genuinely stays per-instance is what differs per *worker* within one
  engine: `base_gpu_id` travels as a constructor argument (`MMEncoder(gpu_id=...)`;
  `BaseMultimodalProcessor._fast_image_processor_device` is the shape to copy).
- **Whole-object passes** (`f(server_args)` handing the instance along) keep the
  supplied-instance contract; don't rewrite the parameter reads unless the
  field is runtime-mutated (see the elastic-EP `ep_size` case in
  `eplb/expert_location.py`) — **or the field is one that resolution fills in
  and the callee runs in a process that has published.** That second case is a
  decision, not a style question: the record carries the user's raw input, so a
  resolution-filled field read off it inside a runner-owned constructor answers
  with the pre-resolution value instead of the effective one. The answer is not
  automatically a bag read: pick where the value should come from — usually the
  `get_*()` bag, sometimes a runner stamp or a constructor argument (the per-mode
  attention pair and the encode-server `gpu_id` are both this). The per-instance
  boundaries above are **not** exempt from this unless-clause; each one gets its own
  disposition. `test_supplied_instance_exposure_ratchet.py` pins that set (empty) —
  three spellings of the read: `server_args.field`, literal-name
  `getattr(server_args, "field", default)`, and the parked form
  (`self.x = server_args` in a method that takes the parameter, read as
  `self.x.field` anywhere in the class) — and fails on a new one, so the
  disposition gets picked when the read is written. Two shapes stay parameter-form on
  purpose: a helper the *resolution pipeline* calls with a `resolved_view` (its
  parameter happens to be named `server_args`), and a factory whose contract is "build
  X from the record you are handed" (`create_kt_config_from_server_args`,
  `DllmConfig.from_server_args`).

The two allow-listed classes are residue, each for its own reason:

- `GrammarManager` is a handed instance for its residual `self.server_args`
  reads, but backend selection is **not** on the instance:
  `create_grammar_backend` reads `get_exec().kernel.grammar_backend`, and
  `__init__` calls that factory whenever `skip_tokenizer_init` is false. In
  production the scheduler process has published; a test that constructs one
  without publishing gets "config namespace not published" unless it patches the
  factory or publishes itself.
- `MMEncoder` publishes the very instance it is handed (`publish(server_args,
  role="encoder")`) and takes its per-worker device as a separate `gpu_id`
  argument, so its tests need no such arrangement. Its `self.server_args` reads are on
  the list as a construction-path convention, and the residual is real: they answer
  with the raw input, so a leaf resolution decided and a post-publish `override` both
  pass them by.

## Mid-resolution reads (inside the pipeline only)

Resolution runs in `__post_init__` and **writes nothing onto the record**: a
handler declares (`self._declare` / `declare_resolution`), the declaration goes
into the stash, and the fields keep what the caller passed. So a mid-resolution
read of a field answers with the *raw input* — every reader in the pipeline goes
through a view instead:

- `resolving_view(server_args)` / `self._resolved()` — the live view (walks the
  stash per read). This is what handlers and hooks bind, conventionally as
  `cfg = resolving_view(self)` at the top of the handler.
- `resolved_view(server_args)` — snapshots the overlay when built, which is what
  a post-process pass wants: it reads the state at *its* slot.

`test_resolution_reads_the_declarations` pins direct field reads at zero over the
two scopes it can derive exactly (every `arg_groups` function taking a config,
every `ServerArgs` handler the dispatcher reaches). Readers the pipeline calls
from elsewhere (`ModelConfig`, the platform defaults, the spec-algo hook) use the
view as well — a field read there is the same bug, just one the derivation cannot
enumerate.

Because the fields are the raw input, resolving a bare `dataclasses.replace` copy lands
in the same place as the parent — the pipeline reads only its own input.
`replace_resolved` is the way to copy a resolved record (it carries the declarations
and the `model_config` memo, so the copy does not re-resolve at all).

## Adding a model-specific config adjustment

Never assign `server_args` fields from model code. Declare instead
(`sglang/srt/arg_groups/overrides.py`):

- Constant per-arch values → `MODEL_OVERRIDES["MyArchForCausalLM"] = {...}`.
- Derived values → `@register_model_override("MyArchForCausalLM")` returning a dict; the
  callable receives *pristine* `server_args` + `hf_config` and must not write.
- Normalization that must see earlier declarations → a post-process pass invoked via
  `run_post_process_pass` at its slot (reads a view, returns a declaration dict).
- Values only knowable at load time are **per-runner state**, not declarations:
  there is no `declare_load_time_override`. A model-family decision that
  its checkpoint drives (shared-experts fusion) is a question the *loader* asks
  the model class — `shared_experts_fusion_disable_reason(hf_config,
  quant_config)`, a classmethod answering without an instance — at the single
  model-instantiation point, and
  `install_shared_experts_fusion_decision` writes the answer to the ACTIVE moe
  flag before that model's layers build and read it
  (`is_shared_experts_fusion_disabled`, config-intent fallback).
  `draft_model_build_scope` brackets every draft build and routes the draft's
  answer to the speculative leaf, so a draft's decision never overwrites the
  target's. A process-level load-time fact (the sm80 dtype fallback —
  device-driven, identical for every runner) records directly via
  `get_context().override`.

Declarable fields form a whitelist: `Arg(..., resolvable=True)` in the `ServerArgs`
dataclass. A declaration against a non-whitelisted field fails at its slot.

## Load-time vs resolution-time

`__post_init__` runs in the launcher process before any model/platform import. Logic that
consults an **extensible registry** (e.g. out-of-tree platforms registering attention
backends in `init_backend()`, which runs at `model_runner` import) is only correct after
the registrars ran, so it must stay at load time (ModelRunner init), writing through
`get_context().override()`. Before moving any load-time logic into resolution, verify
everything it reads is already complete at construction time.
