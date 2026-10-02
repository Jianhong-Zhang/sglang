from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_scale_module(dtype: torch.dtype) -> Module:
    """Compile and cache the JIT scale module for a given dtype."""
    # Checks on the compile key live here, not in `scale`: `cache_once` keys on
    # `dtype`, so this runs once per specialisation instead of once per call.
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            f"Unsupported dtype {dtype}. Supported: float16, bfloat16, float32"
        )
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "scale",
        *args,
        cuda_files=["elementwise/scale.cuh"],
        cuda_wrappers=[("scale", f"scale<{args}>")],
    )


def scale(
    src: torch.Tensor, factor: float, out: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Element-wise scale: dst = src * factor.

    Supported dtypes: torch.float16, torch.bfloat16, torch.float32.

    Parameters
    ----------
    src    : CUDA tensor (FP16 / BF16 / FP32)
    factor : scale factor
    out    : optional pre-allocated output tensor (same shape/dtype as src)

    Returns
    -------
    Scaled tensor (dst = src * factor).
    """
    # DO NOT add proactive validation here: every check costs interpreter time
    # on a per-forward path. Tensor invariants belong in the C++ launcher, and
    # anything about the compile key belongs in `_jit_scale_module`.
    if out is None:
        out = torch.empty_like(src)

    module = _jit_scale_module(src.dtype)
    module.scale(out, src, factor)
    return out
