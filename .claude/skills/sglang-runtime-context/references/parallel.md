# `get_parallel()`: one spelling per name

**There is no `.config` hop.** Ranks and group handles are `@property`
read-through over the canonical getters, so they answer with the live process
groups. Everything else — `tp_size`, `pp_size`, `attn_cp_size`, `dcp_size`,
`moe_dp_size` included, alongside config-only leaves like `nccl_port`,
`enable_dp_attention`, `dp_size`, `ep_size`, `dwdp_size` — is answered from the
published `parallel` bag, and follows a post-publish override. Reading a leaf before
publish raises a `ValueError` naming the namespace; an unknown name is an
`AttributeError`.

A size reads from the configuration because the groups are built at exactly the
configured widths — checked at every assignment to `_TP` / `_PP` / `_ATTN_CP` /
`_DCP` / `_MOE_DP` in `parallel_state.py`. These do not follow that rule:

- `initialize_model_parallel` aliases `_MOE_DP` to `_ATTN_CP` when `attn_cp_size >
  moe_dp_size`, so a reader that means **the width of the MoE communicator it is about
  to operate on** calls `get_moe_cp_size()`, not `get_parallel().moe_dp_size`.
- `patch_tensor_parallel_group` runs a scope under a different TP group (draft
  workers), and declares it by overriding `tp_size`, `tp_rank` and `tp_group`
  for the scope's duration. Readers inside need no special spelling.
- Elastic EP scales `ep_size` / `dp_size` on the published bag while the group
  coordinators keep their construction width. Those are different names, not two
  answers to one name.
- DCP keeps its own pair: `get_parallel().attn_dcp_size` / `.dcp_enabled` answer the
  *effective* topology, while `dcp_size` is what the launch requested. The pair does
  not need dist init: with no group installed it degrades to `1` / `False`
  (`test_attn_dcp_defaults_when_group_is_uninitialized` pins this).

A process-global seed field-read of one of these sizes
(`get_server_args().tp_size`, or an alias of it) is a read-ratchet failure. A
`server_args` the object was *handed* is a different thing and not a ratchet
matter (see the supplied-instance rules with the config-resolution reference).

Fail-loud is narrower: before dist init, a live size/group read raises — except
the DCP pair above. After init, only the DCP group is optional (`_DCP` exists only when
`dcp_size > 1`; attn-CP and moe-DP always install, as size-1 aliases if unused). The
`config` hop is deliberately dynamo-traceable (a plain property over a slot, no
`object.__getattribute__`); gate helpers like `enable_moe_dense_fully_dp()` run inside
compiled model forwards (`test_parallel_config_leaves_trace_under_torch_compile` pins
this).

A third surface carries the same names: `ParallelState` (`self.ps` / `mr.ps`), the
frozen per-process snapshot built once in `Scheduler.__init__` from these configured
sizes plus this process's ranks, and handed down (draft runners included). Prefer it
where an object was handed one; it is not a global accessor.
