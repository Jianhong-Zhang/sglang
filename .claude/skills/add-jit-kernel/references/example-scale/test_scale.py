import pytest
import torch

from sglang.kernels.ops.elementwise.scale import scale
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("size", [1, 127, 128, 1024, 4097])  # cover tail remainder
@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0, 3.0])
def test_scale_correctness(dtype, size, factor):
    src = torch.randn(size, dtype=dtype, device="cuda")
    out = scale(src, factor)
    expected = src * factor

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_scale_out_param(dtype):
    src = torch.randn(1024, dtype=dtype, device="cuda")
    out = torch.empty_like(src)
    result = scale(src, 2.0, out=out)
    assert result is out
    torch.testing.assert_close(out, src * 2.0, rtol=1e-2, atol=1e-2)


def test_scale_cpu_error():
    src = torch.randn(128, dtype=torch.float16)  # CPU tensor
    with pytest.raises(RuntimeError, match="CUDA"):
        scale(src, 2.0)


def test_scale_unsupported_dtype():
    src = torch.randint(0, 10, (128,), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="dtype"):
        scale(src, 2.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
