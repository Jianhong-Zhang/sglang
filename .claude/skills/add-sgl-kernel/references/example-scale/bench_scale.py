import itertools

import sgl_kernel
import torch
import triton
import triton.testing

from sglang.utils import is_in_ci

IS_CI = is_in_ci()

dtypes = [torch.float16] if IS_CI else [torch.float16, torch.bfloat16, torch.float32]
sizes = [4096] if IS_CI else [2**n for n in range(10, 20)]  # 1K … 512K
factors = [2.0]

configs = list(itertools.product(dtypes, sizes))


def torch_scale(input: torch.Tensor, factor: float) -> torch.Tensor:
    return input * factor


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["dtype", "size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sglang", "torch"],
        line_names=["SGL Kernel", "PyTorch"],
        styles=[("green", "-"), ("red", "--")],
        ylabel="µs (median)",
        plot_name="scale-performance",
        args={},
    )
)
def benchmark(dtype, size, provider):
    input = torch.randn(size, dtype=dtype, device="cuda")
    out = torch.empty_like(input)
    factor = 2.0

    if provider == "sglang":
        fn = lambda: sgl_kernel.scale(input, factor, out=out)
    else:
        fn = lambda: torch_scale(input, factor)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
