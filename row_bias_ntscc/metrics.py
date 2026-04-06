"""Lightweight metrics for adaptation scripts."""

from __future__ import annotations

import math


def psnr_from_mse(mse: float) -> float:
    if mse <= 0 or not math.isfinite(mse):
        return float("nan")
    return 10.0 * math.log10(1.0 / mse)


def shannon_bits_per_complex_use(snr_db: float) -> float:
    """``log2(1 + SNR_linear)`` as in NTSCC+ ``test_compatible_ntscc.py`` (``cbr_k`` term)."""
    return math.log2(1.0 + 10.0 ** (float(snr_db) / 10.0))


def side_info_cbr(
    num_params: int,
    image_h: int,
    image_w: int,
    snr_db: float,
    *,
    bits_per_param: int = 8,
    use_channel_capacity_norm: bool = False,
) -> float:
    """
    Side-information CBR for extra scalars (e.g. row bias).

    - Total bits: ``num_params * bits_per_param``.
    - With ``use_channel_capacity_norm=True``:
      ``bits / (log2(1 + 10^(SNR/10)) * 3 * H * W)``.
    - With ``use_channel_capacity_norm=False``:
      ``bits / (3 * H * W)`` only.
    """
    if image_h <= 0 or image_w <= 0 or num_params <= 0:
        return 0.0
    bits = float(num_params) * float(bits_per_param)
    denom_base = 3.0 * float(image_h) * float(image_w)
    if use_channel_capacity_norm:
        cap = shannon_bits_per_complex_use(snr_db)
        if cap <= 0:
            return float("inf")
        return bits / (cap * denom_base)
    return bits / denom_base


def side_info_cbr_from_bits(
    total_bits: float,
    image_h: int,
    image_w: int,
    snr_db: float,
    *,
    use_channel_capacity_norm: bool = False,
) -> float:
    """
    Side-information CBR for an explicit bit budget.

    This matches ``side_info_cbr`` exactly, but accepts total side bits directly
    so quantized/Huffman-coded side information can use the same normalization.
    """
    if image_h <= 0 or image_w <= 0 or total_bits <= 0:
        return 0.0
    denom_base = 3.0 * float(image_h) * float(image_w)
    if use_channel_capacity_norm:
        cap = shannon_bits_per_complex_use(snr_db)
        if cap <= 0:
            return float("inf")
        return float(total_bits) / (cap * denom_base)
    return float(total_bits) / denom_base
