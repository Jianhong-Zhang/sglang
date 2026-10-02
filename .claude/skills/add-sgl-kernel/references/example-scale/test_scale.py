import pytest
import sgl_kernel
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("size", [128, 1024, 4096, 65536])
@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
def test_scale_correctness(dtype, size, factor):
    input = torch.randn(size, dtype=dtype, device="cuda")
    out = torch.empty_like(input)

    result = sgl_kernel.scale(input, factor, out=out)
    assert result is out

    expected = input * factor
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


def test_scale_shape_mismatch():
    input = torch.randn(128, dtype=torch.float16, device="cuda")
    out = torch.empty(256, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="same shape"):
        sgl_kernel.scale(input, 2.0, out=out)


def test_scale_cpu_input():
    input = torch.randn(128, dtype=torch.float16)  # CPU
    out = torch.empty_like(input)
    with pytest.raises(RuntimeError, match="CUDA"):
        sgl_kernel.scale(input, 2.0, out=out)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
