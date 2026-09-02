"""Bandwidth microbenchmark for the HiCache XPU D2H transfer kernel.

Measures ``xpu_kvcacheio.transfer_kv_all_layer`` (the MHA layer_first -> layer_first
backup path) at a fixed byte count while varying only how fragmented the index
sets are, so the index pattern is the sole variable:

    contiguous     one run of tokens on both sides -- the best case
    pages_host     host destination pages shuffled, device source contiguous
    pages_device   device source pages shuffled, host destination contiguous
    pages_both     both sides shuffled at page granularity
    random_both    both sides a random token permutation -- the worst case

The page patterns are the realistic ones: HiCache allocates host and device slots
per page, so a batch is typically a handful of non-adjacent pages rather than one
run. ``random_both`` is the pathological bound.

Timing mirrors HiCacheController.start_writing: start_event is recorded on the
current stream, the transfer is submitted inside a write-stream context, and
finish_event is recorded there, so the reported interval is what the production
bandwidth log would report for the same batch. host_submit is the wall-clock time
the calling thread spends inside the submit block, which is what a per-launch
device sync would inflate.

The kernels are JIT-built on first use, so the oneAPI environment must be active:

    source /opt/intel/oneapi/setvars.sh
    python benchmark/hicache/bench_xpu_transfer.py

Defaults match a gemma-4-31B-it run with tp=2 and page_size=64: the 50-layer
sliding-window pool, 8 KV heads x 256 head_dim in bf16, i.e. item_size 4096 B, so
320 tokens is 125 MiB.
"""

import argparse
import statistics
import time

import torch

from sglang.srt.mem_cache import xpu_kvcacheio


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[64, 320, 512],
        help="batch sizes in tokens; each is measured across all patterns",
    )
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument(
        "--item-size",
        type=int,
        default=4096,
        help="bytes per token per layer per K/V buffer (head_num * head_dim * dtype)",
    )
    parser.add_argument(
        "--pool-tokens",
        type=int,
        default=4096,
        help="slots per pool; larger spreads the scattered patterns further apart",
    )
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", type=str, default="xpu:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the correctness check that the scattered indices really move data",
    )
    return parser.parse_args()


def make_indices(pattern, num_tokens, pool_tokens, page_size, device):
    """Return (src_indices, dst_indices) for one pattern. Both are unique."""

    def contiguous(offset):
        return torch.arange(offset, offset + num_tokens, dtype=torch.int64)

    def shuffled_pages():
        # Whole pages, in random order and non-adjacent: what HiCache produces
        # once the pools have been allocated and freed for a while.
        num_pages = (num_tokens + page_size - 1) // page_size
        pool_pages = pool_tokens // page_size
        assert num_pages <= pool_pages, "pool too small for this batch"
        starts = torch.randperm(pool_pages)[:num_pages] * page_size
        idx = torch.cat([torch.arange(s, s + page_size) for s in starts.tolist()])
        return idx[:num_tokens].to(torch.int64)

    def random_tokens():
        return torch.randperm(pool_tokens)[:num_tokens].to(torch.int64)

    if pattern == "contiguous":
        src, dst = contiguous(pool_tokens - num_tokens), contiguous(0)
    elif pattern == "pages_host":
        src, dst = contiguous(0), shuffled_pages()
    elif pattern == "pages_device":
        src, dst = shuffled_pages(), contiguous(0)
    elif pattern == "pages_both":
        src, dst = shuffled_pages(), shuffled_pages()
    elif pattern == "random_both":
        src, dst = random_tokens(), random_tokens()
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return src.to(device), dst.to(device)


def verify(pools, src, dst, num_tokens):
    """Check a few tokens actually landed, and that untouched slots did not.

    Cheap insurance that a scattered pattern is really moving the bytes it claims
    to: a silently out-of-range index would otherwise just look fast.
    """
    dk, dv, hk, hv, _, _, _, _ = pools
    for device_buf, host_buf, name in ((dk, hk, "K"), (dv, hv, "V")):
        for i in (0, num_tokens // 2, num_tokens - 1):
            s, d = int(src[i]), int(dst[i])
            for layer in (0, device_buf.shape[0] - 1):
                want = device_buf[layer, s].cpu()
                got = host_buf[layer, d]
                if not torch.equal(want, got):
                    raise AssertionError(
                        f"{name} mismatch at layer={layer} src={s} dst={d}"
                    )
    # A slot outside the destination set must still hold its sentinel.
    untouched = set(range(hk.shape[1])) - set(dst.tolist())
    if untouched:
        idx = next(iter(untouched))
        if hk[0, idx].abs().sum().item() != 0:
            raise AssertionError(f"host slot {idx} was written but is not in dst")


def measure(pools, src, dst, item_size, layers, iters, warmup, stream):
    """Time the transfer with the production event pattern.

    Returns (device_ms list, host_submit_ms list).
    """
    dk, dv, hk, hv, dk_p, dv_p, hk_p, hv_p = pools
    device_ms, host_ms = [], []
    for i in range(warmup + iters):
        start = torch.xpu.Event(enable_timing=True)
        finish = torch.xpu.Event(enable_timing=True)
        host_t0 = time.perf_counter()
        start.record()  # current stream, as in start_writing
        with torch.xpu.stream(stream):
            start.wait(stream)
            xpu_kvcacheio.transfer_kv_all_layer(
                dk_p, hk_p, dv_p, hv_p, src, dst, item_size, layers
            )
            finish.record()
        submit_ms = (time.perf_counter() - host_t0) * 1e3
        finish.synchronize()
        if i >= warmup:
            device_ms.append(start.elapsed_time(finish))
            host_ms.append(submit_ms)
    return device_ms, host_ms


def main():
    args = parse_args()
    # Events bind to the current device; without this they would be stamped by
    # xpu:0's clock while the transfer runs on another device, and elapsed_time
    # across two device clocks returns their offset rather than a duration.
    torch.xpu.set_device(args.device)
    torch.manual_seed(args.seed)
    assert args.item_size % 2 == 0, "item_size must be even for the int16 container"
    elems = args.item_size // 2

    xpu_kvcacheio.load()

    # int16 is just a 2-byte container; the kernel copies raw bytes. Values equal
    # the slot index so verify() can tell a real copy from a coincidence.
    def device_pool():
        buf = torch.empty(
            args.layers, args.pool_tokens, elems, dtype=torch.int16, device=args.device
        )
        buf[:] = (
            torch.arange(args.pool_tokens, dtype=torch.int16, device=args.device)
            .view(1, -1, 1)
            .expand_as(buf)
        )
        return buf

    def host_pool():
        return torch.zeros(
            args.layers, args.pool_tokens, elems, dtype=torch.int16, pin_memory=True
        )

    def table(buf):
        return torch.tensor(
            [buf[i].data_ptr() for i in range(buf.shape[0])],
            dtype=torch.uint64,
            device=args.device,
        )

    dk, dv, hk, hv = device_pool(), device_pool(), host_pool(), host_pool()
    pools = (dk, dv, hk, hv, table(dk), table(dv), table(hk), table(hv))
    per_token = args.layers * 2 * args.item_size
    print(
        f"pool={args.pool_tokens} tokens x {args.layers} layers x 2 x {args.item_size} B "
        f"= {dk.nbytes * 2 / 2**30:.2f} GiB device / {hk.nbytes * 2 / 2**30:.2f} GiB pinned host"
        f" | {per_token / 1024:.0f} KiB per token"
    )

    write_stream = torch.xpu.Stream(device=args.device)
    patterns = [
        "contiguous",
        "pages_host",
        "pages_device",
        "pages_both",
        "random_both",
    ]

    for num_tokens in args.tokens:
        num_bytes = num_tokens * per_token
        print(
            f"\ntokens={num_tokens}  bytes={num_bytes / 2**20:.2f} MiB  "
            f"page_size={args.page_size}  iters={args.iters}"
        )
        print(
            f"  {'pattern':<13} {'best ms':>9} {'median ms':>10} {'GiB/s':>8} "
            f"{'vs contig':>10} {'host submit ms':>15}"
        )
        baseline = None
        for pattern in patterns:
            src, dst = make_indices(
                pattern, num_tokens, args.pool_tokens, args.page_size, args.device
            )
            if not args.no_verify:
                hk.zero_(), hv.zero_()
                measure(pools, src, dst, args.item_size, args.layers, 1, 0, write_stream)
                verify(pools, src, dst, num_tokens)
            device_ms, host_ms = measure(
                pools,
                src,
                dst,
                args.item_size,
                args.layers,
                args.iters,
                args.warmup,
                write_stream,
            )
            best, median = min(device_ms), statistics.median(device_ms)
            gibps = num_bytes / 2**30 / (best / 1e3)
            baseline = baseline or gibps
            print(
                f"  {pattern:<13} {best:9.3f} {median:10.3f} {gibps:8.2f} "
                f"{gibps / baseline * 100:9.0f}% {statistics.median(host_ms):15.3f}"
            )


if __name__ == "__main__":
    main()
