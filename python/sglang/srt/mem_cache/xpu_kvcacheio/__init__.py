"""SYCL KV-cache transfer kernels for HiCache on Intel XPU.

Provides the XPU counterpart of ``sgl_kernel.kvcacheio``, which is CUDA/HIP
only. The kernels are compiled on first use with ``torch.utils.cpp_extension``
in SYCL mode; ``sgl-kernel-xpu`` does not ship a ``kvcacheio`` module yet, so
there is no AOT package to import from.

Exposes the entry points HiCache needs for the ``kernel_xpu`` io backend, in MHA
(separate K and V buffers) and MLA (one fused latent buffer) flavors:

D2H (:meth:`backup_from_device_all_layer`)
    ``transfer_kv_all_layer``            device layer_first -> host layer_first
    ``transfer_kv_all_layer_lf_pf``      device layer_first -> host page_first
    ``transfer_kv_all_layer_mla``        MLA, layer_first -> layer_first
    ``transfer_kv_all_layer_mla_lf_pf``  MLA, layer_first -> page_first

H2D (:meth:`load_to_device_per_layer`)
    ``transfer_kv_per_layer``            host layer_first -> device layer_first
    ``transfer_kv_per_layer_pf_lf``      host page_first  -> device layer_first
    ``transfer_kv_per_layer_mla``        MLA, layer_first -> layer_first
    ``transfer_kv_per_layer_mla_pf_lf``  MLA, page_first  -> layer_first
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

_CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")
_SOURCES = [os.path.join(_CSRC_DIR, "kvcacheio.sycl")]


@lru_cache(maxsize=1)
def _load_module():
    """Compile and load the SYCL extension. Cached; compilation happens once."""
    from torch.utils.cpp_extension import load

    logger.info("Compiling HiCache XPU SYCL transfer kernels (first use)")
    return load(
        name="sglang_hicache_xpu_kvcacheio",
        sources=_SOURCES,
        with_sycl=True,
        extra_cflags=["-O3"],
        verbose=False,
    )


def load() -> None:
    """Compile the kernels eagerly, raising a diagnosable error if that fails.

    Called from ``ServerArgs`` at startup so the one-time build cost is paid in
    the launcher rather than raced for by every TP rank on the first transfer,
    and so a missing toolchain surfaces immediately instead of mid-request.
    """
    try:
        _load_module()
    except Exception as e:
        raise RuntimeError(
            "Failed to build the HiCache XPU SYCL transfer kernels, which the "
            "kernel_xpu io backend requires. Compilation needs a SYCL toolchain "
            "(icpx); source the oneAPI environment (setvars.sh) or set SYCL_HOME. "
            f"Underlying error: {e}"
        ) from e


def _trace(name, src_indices, dst_indices, item_size, num_layers, num_kv, **extra):
    """Log one line describing a transfer launch.

    Guarded on the level because it formats tensor metadata and would otherwise
    run on every layer of every transfer.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    n = src_indices.numel()
    total_bytes = n * num_layers * num_kv * item_size
    detail = "".join(f" {k}={v}" for k, v in extra.items())
    # DEBUG: index ranges. Each .min()/.max() forces a device sync, so this is
    # diagnostic-only -- it is what tells an out-of-range index (which faults
    # asynchronously and reports as DEVICE_LOST) apart from a real device error.
    if n:
        src_lo, src_hi = src_indices.min().item(), src_indices.max().item()
        dst_lo, dst_hi = dst_indices.min().item(), dst_indices.max().item()
    else:
        src_lo = src_hi = dst_lo = dst_hi = None
    logger.debug(
        "[transfer init kernel] %s: items=%d layers=%d num_kv=%d item_size=%d "
        "total=%.3f MiB src_idx(dev=%s dtype=%s range=[%s,%s]) "
        "dst_idx(dev=%s dtype=%s range=[%s,%s])%s",
        name,
        n,
        num_layers,
        num_kv,
        item_size,
        total_bytes / (1 << 20),
        src_indices.device,
        src_indices.dtype,
        src_lo,
        src_hi,
        dst_indices.device,
        dst_indices.dtype,
        dst_lo,
        dst_hi,
        detail,
    )


def transfer_kv_all_layer(
    src_k_layers,
    dst_k_layers,
    src_v_layers,
    dst_v_layers,
    src_indices,
    dst_indices,
    item_size,
    num_layers,
):
    """D2H, all layers, device layer_first -> host layer_first."""
    _trace(
        "transfer_kv_all_layer (D2H lf->lf)",
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        num_kv=2,
    )
    _load_module().transfer_kv_all_layer(
        src_k_layers,
        dst_k_layers,
        src_v_layers,
        dst_v_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
    )


def transfer_kv_all_layer_lf_pf(
    src_k_layers,
    dst_k,
    src_v_layers,
    dst_v,
    src_indices,
    dst_indices,
    item_size,
    dst_layout_dim,
    num_layers,
):
    """D2H, all layers, device layer_first -> host page_first."""
    _trace(
        "transfer_kv_all_layer_lf_pf (D2H lf->pf)",
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        num_kv=2,
        dst_layout_dim=dst_layout_dim,
    )
    _load_module().transfer_kv_all_layer_lf_pf(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
    )


def transfer_kv_per_layer(
    src_k,
    dst_k,
    src_v,
    dst_v,
    src_indices,
    dst_indices,
    item_size,
):
    """H2D, one layer, host layer_first -> device layer_first."""
    _trace(
        "transfer_kv_per_layer (H2D lf->lf)",
        src_indices,
        dst_indices,
        item_size,
        num_layers=1,
        num_kv=2,
    )
    _load_module().transfer_kv_per_layer(
        src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size
    )


def transfer_kv_per_layer_pf_lf(
    src_k,
    dst_k,
    src_v,
    dst_v,
    src_indices,
    dst_indices,
    layer_id,
    item_size,
    src_layout_dim,
):
    """H2D, one layer, host page_first -> device layer_first."""
    _trace(
        "transfer_kv_per_layer_pf_lf (H2D pf->lf)",
        src_indices,
        dst_indices,
        item_size,
        num_layers=1,
        num_kv=2,
        layer_id=layer_id,
        src_layout_dim=src_layout_dim,
    )
    _load_module().transfer_kv_per_layer_pf_lf(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
    )


def transfer_kv_all_layer_mla(
    src_layers,
    dst_layers,
    src_indices,
    dst_indices,
    item_size,
    num_layers,
):
    """D2H, all layers, MLA device layer_first -> host layer_first."""
    _trace(
        "transfer_kv_all_layer_mla (D2H lf->lf)",
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        num_kv=1,
    )
    _load_module().transfer_kv_all_layer_mla(
        src_layers, dst_layers, src_indices, dst_indices, item_size, num_layers
    )


def transfer_kv_all_layer_mla_lf_pf(
    src_layers,
    dst,
    src_indices,
    dst_indices,
    item_size,
    dst_layout_dim,
    num_layers,
):
    """D2H, all layers, MLA device layer_first -> host page_first."""
    _trace(
        "transfer_kv_all_layer_mla_lf_pf (D2H lf->pf)",
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        num_kv=1,
        dst_layout_dim=dst_layout_dim,
    )
    _load_module().transfer_kv_all_layer_mla_lf_pf(
        src_layers,
        dst,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
    )


def transfer_kv_per_layer_mla(
    src,
    dst,
    src_indices,
    dst_indices,
    item_size,
):
    """H2D, one layer, MLA host layer_first -> device layer_first."""
    _trace(
        "transfer_kv_per_layer_mla (H2D lf->lf)",
        src_indices,
        dst_indices,
        item_size,
        num_layers=1,
        num_kv=1,
    )
    _load_module().transfer_kv_per_layer_mla(
        src, dst, src_indices, dst_indices, item_size
    )


def transfer_kv_per_layer_mla_pf_lf(
    src,
    dst,
    src_indices,
    dst_indices,
    layer_id,
    item_size,
    src_layout_dim,
):
    """H2D, one layer, MLA host page_first -> device layer_first."""
    _trace(
        "transfer_kv_per_layer_mla_pf_lf (H2D pf->lf)",
        src_indices,
        dst_indices,
        item_size,
        num_layers=1,
        num_kv=1,
        layer_id=layer_id,
        src_layout_dim=src_layout_dim,
    )
    _load_module().transfer_kv_per_layer_mla_pf_lf(
        src, dst, src_indices, dst_indices, layer_id, item_size, src_layout_dim
    )
