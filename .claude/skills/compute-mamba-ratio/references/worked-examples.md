# Worked examples

One measured hybrid model (TP8, B300, fp32 ssm + fp8 KV), validated against measured clamps.
Recompute with your own `token_equiv` for a different model.

## The constants, read off one boot log

```
Mamba Cache is allocated. max_mamba_cache_size: 257, conv_state size: 0.46GB, ssm_state size: 13.04GB
KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 1167552, KV size: 15.03 GB
```

`state_bytes_per_slot = (0.46 + 13.04) GiB / 257 ≈ 56.4 MB`, `kv_bytes_per_token = 15.03 GiB / 1167552 ≈ 13.8 KB` → `token_equiv ≈ **4080**`.

## Balance ratio

Cache strategy `extra_buffer_lazy` → `S = 4`. Workload `L = 9216` (8192 in + 1024 out). NOSPEC → `D = 0`.

- **TP (dcp_size=1)**: `r = 4 · 4080 · 1 / 9216 ≈ 1.8`. Measured: clamp 88, KV cap 87 → balanced. ✅
- **DCP8 (dcp_size=8)**: `r = 4 · 4080 · 8 / 9216 ≈ 14`. Measured at r=14: clamp 125, KV cap 129 → balanced. ✅ (At the naive r=1.8 the DCP KV pool is ~8× over-provisioned — cap 683 vs clamp 86 — wasting budget that should go to the state pool.)

## Predicted clamp

`predict_clamp` on the same model at the **default** `extra_buffer`, overlap on and the decode-lock skip off (`S = 5`), three boot logs:

| config | `max_mamba_cache_size` | `mmcs // S` | measured `max_running_requests` |
|---|---|---|---|
| TP8, no DCP | 257 | 51 | 51 |
| DCP8, `r = 5.97` | 451 | 90 | 90 |
| DCP8, `r = 9` | 474 | 94 | 94 |

## Reference `r` by context length

For this model (`token_equiv ≈ 4080`), S=5 default; multiply by `(S+D)/S` for spec, by `dcp_size` for DCP:

| L | 2K | 4K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| r | 10.0 | 5.0 | 2.5 | 0.62 | 0.31 | 0.16 |
