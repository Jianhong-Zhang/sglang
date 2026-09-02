"""Bandwidth microbenchmark for the HiCache XPU H2D load kernel.

Measures ``xpu_kvcacheio.transfer_kv_per_layer`` -- the MHA host layer_first ->
device layer_first path that ``HiCacheController.start_loading`` drives. Unlike a
backup, which moves every layer in one launch, a load issues **one launch per
layer** so each attention layer can start as soon as its own slab arrives. This
script prices that choice, at a fixed byte count per row group:

    per-layer loop      layers launches, exactly what start_loading does
    one launch          the same bytes in a single launch, i.e. what a page-first
                        host layout could achieve -- the batching headroom
    single layer        one launch of 1/layers of the bytes, i.e. per-launch cost

and, per row, three index patterns (contiguous, page-shuffled, random tokens) to
confirm fragmentation is as free here as it is on the backup path.

Timing mirrors start_loading: start_event on the current stream, launches inside a
load-stream context, finish_event recorded there -- so "per-layer loop" is what
the ``[H2D]`` bandwidth log reports for the same batch.

The kernels are JIT-built on first use, so the oneAPI environment must be active.
The SYCL sources trace every launch to stderr unconditionally, which at one launch
per layer is loud and costs host time, so redirect it:

    source /opt/intel/oneapi/setvars.sh
    python benchmark/hicache/bench_xpu_transfer_h2d.py 2>/dev/null

Defaults match the 50-layer sliding-window pool of a gemma-4-31B-it run with tp=2
and page_size=64: 8 KV heads x 256 head_dim in bf16, i.e. item_size 4096 B.
"""

import argparse
import statistics
import time

import torch

from sglang.srt.mem_cache import xpu_kvcacheio


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 384, 512])
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument(
        "--item-size",
        type=int,
        default=4096,
        help="bytes per token per layer per K/V buffer (head_num * head_dim * dtype)",
    )
    parser.add_argument("--pool-tokens", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", type=str, default="xpu:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--with-forward",
        action="store_true",
        help="keep a second stream busy with matmuls, since loads overlap a prefill",
    )
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def make_indices(pattern, num_tokens, pool_tokens, page_size, device):
    """Return (host_indices, device_indices) for one pattern; both unique."""

    def contiguous(offset):
        return torch.arange(offset, offset + num_tokens, dtype=torch.int64)

    def shuffled_pages():
        num_pages = (num_tokens + page_size - 1) // page_size
        pool_pages = pool_tokens // page_size
        assert num_pages <= pool_pages, "pool too small for this batch"
        starts = torch.randperm(pool_pages)[:num_pages] * page_size
        idx = torch.cat([torch.arange(s, s + page_size) for s in starts.tolist()])
        return idx[:num_tokens].to(torch.int64)

    if pattern == "contiguous":
        src, dst = contiguous(pool_tokens - num_tokens), contiguous(0)
    elif pattern == "pages_both":
        src, dst = shuffled_pages(), shuffled_pages()
    elif pattern == "random_both":
        src = torch.randperm(pool_tokens)[:num_tokens].to(torch.int64)
        dst = torch.randperm(pool_tokens)[:num_tokens].to(torch.int64)
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return src.to(device), dst.to(device)


def measure(fn, iters, warmup, stream, busy_stream=None, busy_fn=None):
    """Time fn() with start_loading's event pattern. Returns (best, median) ms."""
    samples = []
    for i in range(warmup + iters):
        if busy_fn is not None:
            with torch.xpu.stream(busy_stream):
                for _ in range(8):
                    busy_fn()
        start = torch.xpu.Event(enable_timing=True)
        finish = torch.xpu.Event(enable_timing=True)
        start.record()  # current stream, as in start_loading
        with torch.xpu.stream(stream):
            start.wait(stream)
            fn()
            finish.record()
        finish.synchronize()
        if i >= warmup:
            samples.append(start.elapsed_time(finish))
        torch.xpu.synchronize()
    return min(samples), statistics.median(samples)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    assert args.item_size % 2 == 0, "item_size must be even for the int16 container"
    elems = args.item_size // 2

    xpu_kvcacheio.load()

    # Host is the source for a load. int16 is just a 2-byte container; values equal
    # the slot index so verify() can tell a real copy from a coincidence.
    def host_pool():
        buf = torch.empty(
            args.layers, args.pool_tokens, elems, dtype=torch.int16, pin_memory=True
        )
        buf[:] = torch.arange(args.pool_tokens, dtype=torch.int16).view(1, -1, 1)
        return buf

    def device_pool():
        return torch.zeros(
            args.layers, args.pool_tokens, elems, dtype=torch.int16, device=args.device
        )

    hk, hv, dk, dv = host_pool(), host_pool(), device_pool(), device_pool()
    # Same allocations reinterpreted as one contiguous slab per token, which is what
    # a page-first host layout gives the kernel. No extra memory.
    hk_all = hk.view(args.pool_tokens, args.layers * elems)
    hv_all = hv.view(args.pool_tokens, args.layers * elems)
    dk_all = dk.view(args.pool_tokens, args.layers * elems)
    dv_all = dv.view(args.pool_tokens, args.layers * elems)

    per_token = args.layers * 2 * args.item_size
    print(
        f"pool={args.pool_tokens} tokens x {args.layers} layers x 2 x {args.item_size} B "
        f"= {dk.nbytes * 2 / 2**30:.2f} GiB device / {hk.nbytes * 2 / 2**30:.2f} GiB pinned host"
        f" | {per_token / 1024:.0f} KiB per token"
        + ("  [concurrent forward]" if args.with_forward else "")
    )

    load_stream = torch.xpu.Stream(device=args.device)
    fwd_stream = torch.xpu.Stream(device=args.device) if args.with_forward else None
    if args.with_forward:
        a = torch.randn(4096, 4096, device=args.device)
        out = torch.empty(4096, 4096, device=args.device)
        busy_fn = lambda: torch.matmul(a, a, out=out)
    else:
        busy_fn = None

    def per_layer_loop(src, dst):
        for i in range(args.layers):
            xpu_kvcacheio.transfer_kv_per_layer(
                src_k=hk[i],
                dst_k=dk[i],
                src_v=hv[i],
                dst_v=dv[i],
                src_indices=src,
                dst_indices=dst,
                item_size=args.item_size,
            )

    def one_launch(src, dst):
        xpu_kvcacheio.transfer_kv_per_layer(
            src_k=hk_all,
            dst_k=dk_all,
            src_v=hv_all,
            dst_v=dv_all,
            src_indices=src,
            dst_indices=dst,
            item_size=args.layers * args.item_size,
        )

    def single_layer(src, dst):
        xpu_kvcacheio.transfer_kv_per_layer(
            src_k=hk[0],
            dst_k=dk[0],
            src_v=hv[0],
            dst_v=dv[0],
            src_indices=src,
            dst_indices=dst,
            item_size=args.item_size,
        )

    def verify(src, dst, num_tokens):
        """The per-layer path must land the right slabs and touch nothing else."""
        for host_buf, device_buf, name in ((hk, dk, "K"), (hv, dv, "V")):
            for i in (0, num_tokens // 2, num_tokens - 1):
                s, d = int(src[i]), int(dst[i])
                for layer in (0, args.layers - 1):
                    if not torch.equal(host_buf[layer, s], device_buf[layer, d].cpu()):
                        raise AssertionError(
                            f"{name} mismatch at layer={layer} host={s} device={d}"
                        )
        untouched = set(range(args.pool_tokens)) - set(dst.tolist())
        if untouched:
            idx = next(iter(untouched))
            if dk[0, idx].abs().sum().item() != 0:
                raise AssertionError(f"device slot {idx} written but not in dst")

    cases = (
        ("per-layer loop", per_layer_loop, args.layers),
        ("one launch", one_launch, args.layers),
        ("single layer", single_layer, 1),
    )
    for num_tokens in args.tokens:
        print(
            f"\ntokens={num_tokens}  all-layer bytes={num_tokens * per_token / 2**20:.2f} MiB"
            f"  page_size={args.page_size}  iters={args.iters}"
        )
        print(
            f"  {'case':<16} {'pattern':<13} {'launches':>8} {'MiB':>8} "
            f"{'best ms':>9} {'GiB/s':>8}"
        )
        for label, fn, launches in cases:
            num_bytes = num_tokens * launches * 2 * args.item_size
            for pattern in ("contiguous", "pages_both", "random_both"):
                src, dst = make_indices(
                    pattern, num_tokens, args.pool_tokens, args.page_size, args.device
                )
                if not args.no_verify and label == "per-layer loop":
                    dk.zero_(), dv.zero_()
                    per_layer_loop(src, dst)
                    torch.xpu.synchronize()
                    verify(src, dst, num_tokens)
                best, _ = measure(
                    lambda: fn(src, dst),
                    args.iters,
                    args.warmup,
                    load_stream,
                    fwd_stream,
                    busy_fn,
                )
                print(
                    f"  {label:<16} {pattern:<13} {launches:8d} {num_bytes / 2**20:8.1f} "
                    f"{best:9.3f} {num_bytes / 2**30 / (best / 1e3):8.2f}"
                )


if __name__ == "__main__":
    main()
