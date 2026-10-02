---
name: debug-distributed-hang
description: Debugs hanging issues in SGLang distributed inference (TP/PP/DP/EP). Covers identifying hang locations via py-spy/watchdog/cuda coredump, per-rank logging to find state divergence, binary-search methodology for locating the first diverge point, and fix patterns. Use when a multi-GPU SGLang run hangs, freezes, hits a collective timeout, or the scheduler watchdog fires.
---

# Debugging Distributed Hangs in SGLang

Ranks whose state diverged deadlock in a collective (AllGather, AllReduce, Broadcast, Barrier): a size mismatch, one rank skipping a collective, small non-determinism drifting into different batches, or one rank dying while the others wait.

## Step 1: Confirm and Locate the Hang

### 1a. Watchdog / py-spy

SGLang's watchdog automatically dumps py-spy traces on timeout. Look for:

```
Scheduler watchdog timeout (self.watchdog_timeout=300, self.soft=False)
```

The py-spy dump shows the stack trace of each thread. The hanging thread is typically blocked in a CUDA synchronize or NCCL collective:

```
Thread (active): "MainThread"
    cuStreamSynchronize (libcuda.so)
    ...
    forward_extend (model_runner.py)
```

SGLang has two watchdog modes (see `python/sglang/srt/utils/watchdog.py`):
- **Hard watchdog** (`soft=False`, default): dumps py-spy traces then sends `SIGQUIT` to kill the parent process.
- **Soft watchdog** (`soft=True`): only logs the timeout without killing the process, giving you more time to manually attach debuggers or collect coredumps.

If the watchdog doesn't trigger, manually dump (py-spy needs root or `CAP_SYS_PTRACE` to attach):

```bash
py-spy dump --pid <scheduler_pid>
```

### 1b. NCCL Debug Logging

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
```

Look for the last collective logged before the hang. Mismatched sizes show up as one rank waiting and another never entering.

On XPU use the oneCCL equivalents (`CCL_LOG_LEVEL`).

### 1c. CUDA Coredump

When py-spy shows a rank blocked in a CUDA synchronize and you need the GPU kernel it waits on, take a GPU coredump:
see [references/cuda-coredump.md](references/cuda-coredump.md).

### 1d. Identify the Collective

From the stack traces and logs, identify:
- Which collective hangs (AllGather, AllReduce, Broadcast)
- Which code path invokes it (e.g., `LogitsProcessor`, `tensor_model_parallel_all_gather`)
- Whether it's a size mismatch or a missing participant

## Step 2: Per-Rank Logging

The key technique: each rank writes its own log file so you can diff them.

### Setup Pattern

```python
import os

_debug_files = {}

def get_debug_file(rank):
    key = f"rank{rank}"
    if key not in _debug_files:
        _debug_files[key] = open(f"/tmp/debug_rank{rank}.log", "w")
    return _debug_files[key]
```

Gate logging behind an env var to avoid overhead in production. `SGLANG_DEBUG_HANG` is not a built-in SGLang env var — you need to add this check yourself in the code you're instrumenting:

```python
if os.environ.get("SGLANG_DEBUG_HANG"):
    f = get_debug_file(rank)
    f.write(f"EVENT_NAME key1={val1} key2={val2}\n")
    f.flush()
```

### What to Log

Log structured events at key state-mutation points:

```python
f.write(f"SCHED_BATCH step={step} num_reqs={n} extend_lens={lens}\n")
f.write(f"VERIFY predict_hash={hash} accept_len={alen}\n")
f.write(f"CACHE_INSERT rid={rid} num_tokens={n}\n")
```

Use consistent event names (uppercase prefix) for easy grep/diff.

### Hash Large Tensors

For tensor values, compute a hash instead of dumping raw data:

```python
import hashlib
h = hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()[:8]
f.write(f"LOGITS logits_hash={h}\n")
```

For token ID lists, `str(list).encode()` works:

```python
h = hashlib.md5(str(tensor.tolist()).encode()).hexdigest()[:8]
```

### Avoid Implicit Synchronization

`tensor.cpu()`, `tensor.tolist()`, and `tensor.numpy()` all trigger CUDA synchronization. This can:
- Change timing and mask or move the hang
- Deadlock if the log point is between two collectives that must run back-to-back

Prefer logging values that are already on CPU (e.g., Python ints, list lengths, request IDs). When you must hash a GPU tensor, do it at a point where the GPU is already idle (e.g., between scheduler steps, not inside a model forward pass).

## Step 3: Diff to Find the Diverge Point

### Basic Diff

```bash
# Extract specific event type
grep "^VERIFY" /tmp/debug_rank0.log > /tmp/v_r0.txt
grep "^VERIFY" /tmp/debug_rank1.log > /tmp/v_r1.txt
diff /tmp/v_r0.txt /tmp/v_r1.txt | head -20
```

### Count Events

```bash
grep -c "^VERIFY" /tmp/debug_rank*.log
```

If counts differ, one rank executed more iterations — that's already a diverge signal.

### Find First Diverge

The first diff line tells you the exact step where ranks diverge. All lines before it are identical — the root cause is at or before this step.

## Step 4: Binary-Search the Root Cause

Once you find the diverging event, trace backwards:

### 4a. Identify Inputs

For the diverging operation, list all its inputs. Add hash logging for each:

```python
f.write(
    f"OP_INPUTS input_a_hash={h_a} input_b_hash={h_b} "
    f"input_c_hash={h_c} input_d_hash={h_d}\n"
)
```

### 4b. Diff Inputs Across Ranks

Compare the hashes. Some inputs will match, some won't. The non-matching input is where divergence entered.

### 4c. Recurse

For the non-matching input, trace where it was produced and repeat: hash its inputs, diff across ranks, find the divergent one. Continue until you reach the root cause.

Done when the first input whose hash differs across ranks was produced from inputs that match.

## Step 5: Common Root Causes and Fixes

### Floating-Point Non-Determinism

**Symptom**: All "logical" inputs are identical (same logits after all-gather), but derived floating-point values (softmax, probabilities) differ across GPUs.

**Example**: in speculative-decoding verification, a softmax and top-k/top-p renormalization yield slightly different probabilities per GPU, so sampling picks different tokens. Those tokens reach the radix cache, prefix-match depths differ, extend lengths differ, and the next AllGather gets mismatched sizes and hangs.

**Fix**: compute the value on rank 0 and `broadcast(result, src=0)`, so every rank uses the same one.

### Random Number Divergence

**Symptom**: Operations using `torch.rand` produce different values on each rank.

**Fix**: Generate on rank 0 and broadcast, or use a shared seed.

### Conditional Code Paths

**Symptom**: A condition (e.g., memory check, queue length) evaluates differently on different ranks, causing one rank to enter a collective while another skips it.

**Fix**: Synchronize the condition value before branching, or restructure to ensure all ranks take the same path.

### Pipeline Parallel (PP) Send/Recv Mismatch

**Symptom**: In PP setups, one stage issues a `send` that the next stage never `recv`s (or vice versa), causing both to block indefinitely. Unlike TP hangs (collective mismatches), PP hangs typically involve point-to-point operations.

**Fix**: Ensure all stages agree on the number of microbatches and the sequence of send/recv calls for each microbatch.

## Step 6: Verify the Fix

Run the failing test multiple times to confirm the fix is stable. Intermittent hangs require many runs. A test that hung ~30% of the time needs at least 10 clean passes to be confident.
