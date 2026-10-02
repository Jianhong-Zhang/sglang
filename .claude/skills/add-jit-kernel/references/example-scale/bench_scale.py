import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.elementwise.scale import scale as jit_scale
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@torch.compile()
def torch_impl_scale(src: torch.Tensor, factor: float) -> torch.Tensor:
    return src * factor


FN_MAP = {
    "jit": jit_scale,
    "torch": torch_impl_scale,
}


# `parametrize(name, full_vals, ci_vals)`: the 3rd arg is the smaller sweep
# auto-selected under CI; the full range runs locally.
@marker.parametrize("size", [2**n for n in range(10, 20)], [4096, 65536])  # 1K .. 512K
@marker.benchmark("impl", ["jit", "torch"])
def benchmark(size: int, impl: str):
    src = create_random(size)
    factor = 2.0
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(src, factor),
        # `src` is read -> clone it per iter to avoid L2 reuse; factor is a scalar.
        graph_clone_args=(0,),
        # Defaults already report bandwidth: memory_args="all" counts src,
        # memory_output="out" counts the returned tensor -> bytes(src)+bytes(out).
    )


if __name__ == "__main__":
    benchmark.run()
