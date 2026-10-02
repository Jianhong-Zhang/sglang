# Maintaining The Catalogs

For maintainers refreshing `fuse-overlap-catalog.md`, `overlap-catalog.md` and
`vllm-torch-compile-fusions.md` by rescanning local source trees. Nothing here
is used by the triage scripts at runtime or during a triage.

## Contents

- Last refresh
- Fuse catalog refresh commands
- Overlap catalog refresh commands

## Last refresh

Fuse catalog: Refresh note `2026-06-26`: rechecked official main heads for SGLang
`8524678889485801e7a4a12d62015be0c68f7a90`, vLLM
`abc71548ef029132c3316b902207f254a246d593`, TensorRT-LLM
`0722c5f47d2cae69ac1a237da51e550dd214532c`, and TokenSpeed
`5aedf69d6b476baa65571011de6ea60fd5a238a8`. The vLLM torch.compile pass
inventory is split out in
`vllm-torch-compile-fusions.md`. Stable
current-code families remain folded into the mainline rows below. This refresh
adds first-class TokenSpeed-origin rows for CuTe DSL MLA, MLA KV pack+FP8
quantize, sampling, lm_head GEMM, and NVFP4 GEMM+SwiGLU+quant, plus the latest
SGLang LTX2 Ada-value diffusion fusion. Recheck PR state before treating an
in-flight row as shipped.

Overlap catalog: Refresh note `2026-06-26`: rechecked official main heads for SGLang
`8524678889485801e7a4a12d62015be0c68f7a90`, vLLM
`abc71548ef029132c3316b902207f254a246d593`, TensorRT-LLM
`0722c5f47d2cae69ac1a237da51e550dd214532c`, and TokenSpeed
`5aedf69d6b476baa65571011de6ea60fd5a238a8`, then added the first
TokenSpeed-origin communication-fusion row. Closed-unmerged SGLang
[#22410](https://github.com/sgl-project/sglang/pull/22410) and FlashInfer
[#2840](https://github.com/flashinfer-ai/flashinfer/pull/2840) were removed
from the PR-backed sections. SGLang
[#21877](https://github.com/sgl-project/sglang/pull/21877), FlashInfer
[#2720](https://github.com/flashinfer-ai/flashinfer/pull/2720), and vLLM
[#35968](https://github.com/vllm-project/vllm/pull/35968) /
[#39301](https://github.com/vllm-project/vllm/pull/39301) remain useful
upstream overlap references as of this refresh.

vLLM torch.compile inventory: Refresh: `2026-06-26`.
Source tree: vLLM `origin/main` at
`abc71548ef029132c3316b902207f254a246d593`; no new LLM compile-fusion pass was
added after `2317682f9` in this refresh. The mainline `#40392` MLA RoPE +
KV-cache cat fusion is already included below. Recent post-`#46735` vLLM
changes include runtime / frontend work such as `#44800` and `#46799`, but they
do not add a new LLM compile-fusion pass to this inventory.

## Fuse catalog refresh commands

```bash
# Optional sibling checkouts used for comparative scanning:
FLASHINFER_REPO=${FLASHINFER_REPO:-../flashinfer}
TRTLLM_REPO=${TRTLLM_REPO:-../TensorRT-LLM}
VLLM_REPO=${VLLM_REPO:-../vllm}

rg -n "fused_add_rmsnorm|gemma_fused_add_rmsnorm|silu_and_mul|gelu_and_mul|fused_qk_rope_reshape_and_cache|fused_set_kv_buffer|fused_metadata_copy|normal_decode_set_metadata|_append_shared_to_topk_output|fused_append_shared_experts_with_weights" python/sglang
rg -n "MiniMaxM2RMSNormTP|fused_qknorm_rope|fused_qk_rope_cat_and_cache_mla|fused_qk_norm_mrope_3d_cache_pts_quant_shuffle|split_qkv_rmsnorm_rope|trtllm_fp8_kv_kernel|set_mla_kv_buffer_fp8_quant" python/sglang
rg -n "FusedMoeRouter|fused_topk_deepseek|moe_fused_gate|aiter_fused_topk|fused_rms_fp8_group_quant|fast_topk_transform_fused|fused_store_index_k_cache|fused_temperature_softmax|fused_softcap" python/sglang
rg -n "fused_qkvzba_split_reshape_cat|fused_gdn_gating|rms_norm_gated|layer_norm_gated|chunk_gated_delta_rule_fwd_kkt_solve_kernel|fused_recurrent_gated_delta_rule_update|fused_mamba_state_scatter_with_mask|_fused_gather_to_staging_kernel|_fused_scatter_from_staging_kernel" python/sglang
rg -n "single_batch_overlap|alt_stream|shared_expert|_comm_stream|scatter_stream|triton_mrope_fused|ring_attn|all_to_all_single|reorder_for_compute_comm_overlap|use_dual_stream" python/sglang
git log --all --format='%h %s' | rg -i 'fused|fusion|overlap|cutedsl|triton|cuda|rope|topk|quant|combine|allreduce|all_to_all'
rg -n "silu_and_mul|gelu_tanh_and_mul|gelu_and_mul|silu_and_mul_scaled_nvfp4_experts_quantize|rmsnorm_quant|fused_add_rmsnorm|fused_add_rmsnorm_quant|fused_rmsnorm_silu" "$FLASHINFER_REPO/flashinfer"
rg -n "AllReduceFusionPattern|allreduce_fusion|trigger_completion_at_end|rope_quantize_fp8|rope_quantize_fp8_append_paged_kv_cache|fused_topk_deepseek|cutlass_fused_moe|trtllm_.*_moe" "$FLASHINFER_REPO/flashinfer"
rg -n "aux_stream|use_async_memset|split_device_green_ctx|split_device_green_ctx_by_sm_count|enable_pdl|launch_with_pdl" "$FLASHINFER_REPO/flashinfer" "$FLASHINFER_REPO/include"
git -C "$FLASHINFER_REPO" log --all --format='%h %s' | rg -i 'fused|fusion|overlap|pdl|stream|rope|kv|quant|topk|moe'
rg -n "flashinfer_silu_and_mul|flashinfer_gelu_tanh_and_mul|flashinfer_rmsnorm|flashinfer_gemma_rmsnorm|flashinfer_fused_add_rmsnorm|flashinfer_apply_rope_with_cos_sin_cache_inplace|triton_fused_add_rms_norm_quant_fp8|fuse_rmsnorm_quant_fp8" "$TRTLLM_REPO/tensorrt_llm/_torch"
rg -n "flashinfer_attention_mha_with_cache|append_paged_kv_cache|flashinfer_mla|append_paged_mla_kv_cache|flashinfer_cached_ssm|selective_state_update|flashinfer.fused_moe" "$TRTLLM_REPO/tensorrt_llm/_torch" "$TRTLLM_REPO/docs/source"
rg -n "multi_stream_moe|multi_stream_mla_attn|multi_stream_gemm|record_event_passthrough|begin_aux_stream_passthrough|end_aux_stream_passthrough|wait_aux_stream_passthrough" "$TRTLLM_REPO/tensorrt_llm/_torch"
git -C "$TRTLLM_REPO" log --all --format='%h %s' | rg -i 'fused|fusion|overlap|flashinfer|mla|kv cache|multi-stream|stream|rope|rmsnorm|moe'
rg -n "fused_add_rms_norm|merge_attn_states|fused_qk_norm_rope|grouped_topk|topk_softmax|topk_sigmoid|dsv3_router_gemm|dsv3_fused_a_gemm|concat_and_cache_mla_rope_fused|gpt_oss_router_gemm|cutlass_scaled_mm|cpu_fused_moe|fused_moe_lora|triton_pos_embed_interpolate" "$VLLM_REPO/vllm" "$VLLM_REPO/csrc"
rg -n "fuse_allreduce_rms|fuse_norm_quant|fuse_act_quant|fuse_attn_quant|enable_qk_norm_rope_fusion|fuse_rope_kvcache|enable_sp|fuse_gemm_comms|RocmAiter|dcp_alltoall|shared_experts_stream|TRTLLM_ENABLE_PDL|wk_weights_proj" "$VLLM_REPO/vllm" "$VLLM_REPO/docs/design/fusions.md" "$VLLM_REPO/csrc"
git -C "$VLLM_REPO" log --all --format='%h %s' | rg -i 'fused|fusion|overlap|triton|cuda|rope|kv cache|topk|router|allreduce|reduce-scatter|all-gather|all_to_all|quant'
# GitHub PR scan terms for the connector or web UI:
#   "fused OR overlap repo:sgl-project/sglang"
#   "triton OR cutedsl OR cuda fused repo:sgl-project/sglang"
#   "fused OR overlap repo:flashinfer-ai/flashinfer"
#   "pdl OR aux_stream OR green_ctx repo:flashinfer-ai/flashinfer"
#   "fused OR overlap repo:NVIDIA/TensorRT-LLM"
#   "flashinfer OR mla OR moe OR rmsnorm repo:NVIDIA/TensorRT-LLM"
#   "multi-stream OR aux_stream OR cudagraph repo:NVIDIA/TensorRT-LLM"
#   "fused OR overlap repo:vllm-project/vllm"
#   "triton OR cuda fused repo:vllm-project/vllm"
```

## Overlap catalog refresh commands

```bash
# Optional sibling checkouts used for comparative scanning:
FLASHINFER_REPO=${FLASHINFER_REPO:-../flashinfer}
TRTLLM_REPO=${TRTLLM_REPO:-../TensorRT-LLM}
VLLM_REPO=${VLLM_REPO:-../vllm}
TOKENSPEED_REPO=${TOKENSPEED_REPO:-../tokenspeed}

rg -n "single_batch_overlap|alt_stream|shared_expert|scatter_stream|_fused_gather_to_staging_kernel|_fused_scatter_from_staging_kernel|async_op=True" python/sglang
rg -n "apply_qk_norm|vision.py|ring_attn|all_to_all_single|reorder_for_compute_comm_overlap|use_dual_stream" python/sglang/multimodal_gen python/sglang/srt
git log --all --format='%h %s' | rg -i 'fused|fusion|overlap|combine|all_to_all|ring attn|stream|triton|cutedsl|cuda'
rg -n "enable_pdl|launch_with_pdl|trigger_completion_at_end|aux_stream|use_async_memset|split_device_green_ctx|split_device_green_ctx_by_sm_count" "$FLASHINFER_REPO/flashinfer" "$FLASHINFER_REPO/include"
git -C "$FLASHINFER_REPO" log --all --format='%h %s' | rg -i 'fused|fusion|overlap|pdl|stream|rope|kv|quant|topk|moe'
rg -n "multi_stream_moe|multi_stream_mla_attn|multi_stream_gemm|record_event_passthrough|begin_aux_stream_passthrough|end_aux_stream_passthrough|wait_aux_stream_passthrough" "$TRTLLM_REPO/tensorrt_llm/_torch"
rg -n "mlir_elementwise_fusion|piecewise|cudagraph|caller_stream.synchronize" "$TRTLLM_REPO/tensorrt_llm/_torch"
git -C "$TRTLLM_REPO" log --all --format='%h %s' | rg -i 'overlap|multi-stream|aux stream|cudagraph|mlir|stream|flashinfer|moe|mla'
rg -n "fuse_gemm_comms|enable_sp|fused_matmul_reduce_scatter|fused_all_gather_matmul|shared_experts_stream|maybe_sync_shared_experts_stream|dcp_alltoall|async_op=True|aux_stream|maybe_execute_in_parallel" "$VLLM_REPO/vllm" "$VLLM_REPO/docs/design/fusions.md"
git -C "$VLLM_REPO" log --all --format='%h %s' | rg -i 'fused|fusion|overlap|allreduce|reduce-scatter|all-gather|all_to_all|stream|multi-stream|triton|cuda|router'
rg -n "enable_allreduce_fusion|comm_fusion|comm_fusion_max_num_tokens|allreduce|reduce_scatter" "$TOKENSPEED_REPO/python" "$TOKENSPEED_REPO/docs"
git -C "$TOKENSPEED_REPO" log --all --format='%h %s' | rg -i 'fused|fusion|overlap|allreduce|stream|comm|mla|tokenspeed_mla'
# GitHub PR scan terms for the connector or web UI:
#   "fused OR overlap repo:sgl-project/sglang"
#   "triton OR cutedsl OR cuda overlap repo:sgl-project/sglang"
#   "fused OR overlap repo:flashinfer-ai/flashinfer"
#   "pdl OR aux_stream OR green_ctx repo:flashinfer-ai/flashinfer"
#   "fused OR overlap repo:NVIDIA/TensorRT-LLM"
#   "multi-stream OR aux_stream OR cudagraph repo:NVIDIA/TensorRT-LLM"
#   "mlir OR piecewise OR flashinfer repo:NVIDIA/TensorRT-LLM"
#   "fused OR overlap repo:vllm-project/vllm"
#   "triton OR cuda overlap repo:vllm-project/vllm"
#   "multi-stream OR aux_stream overlap repo:vllm-project/vllm"
#   "fused OR overlap OR comm_fusion repo:lightseekorg/tokenspeed"
```
