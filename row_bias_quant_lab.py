"""Global quantize/dequantize + Huffman/entropy helpers for row-bias side-info experiments (int3/int4/int5)."""

from __future__ import annotations

import heapq
import math

import torch

# User-requested fixed Kodak denominator for equivalent side CBR (10 dB ~ 3.459 bits/use).
KODAK_CBR_SIDE_DENOM = 3.459 * 3.0 * 768.0 * 512.0


def get_row_bias_row_indices(total_rows: int, mode: str) -> list[int]:
    if mode == "all16":
        return list(range(total_rows))
    if mode == "odd8":
        return list(range(1, total_rows, 2))
    # 0-based indices 1,4,7,... → 1-based rows 2,5,8,... (every third row, skip two between picks)
    if mode == "every3_from1":
        return list(range(1, total_rows, 3))
    # 0-based indices 1,5,9,... → 1-based rows 2,6,10,... (every fourth row, skip three between picks)
    if mode == "every4_from1":
        return list(range(1, total_rows, 4))
    raise ValueError(f"Unsupported row-bias row mode: {mode}")


def compact_row_bias_to_delta_y(
    compact_bias: torch.Tensor,
    *,
    height: int,
    width: int,
    selected_indices: list[int],
    axis: str,
) -> torch.Tensor:
    """
    Build Δy with shape ``[1, C, H, W]`` (broadcast-add to y_anchor) from compact ``[C, K, 1]``.

    - ``axis == "h"``: add along latent rows (dim 2); same bias for all W at each selected row.
    - ``axis == "w"``: add along latent cols (dim 3); same bias for all H at each selected col.
    """
    if compact_bias.dim() != 3 or compact_bias.shape[2] != 1:
        raise ValueError(f"compact_bias must be [C, K, 1], got {tuple(compact_bias.shape)}")
    c, k, _ = compact_bias.shape
    if len(selected_indices) != k:
        raise ValueError(f"len(selected_indices)={len(selected_indices)} != K={k}")
    comp = compact_bias.squeeze(-1)
    device = compact_bias.device
    dtype = compact_bias.dtype
    idx = torch.tensor(selected_indices, device=device, dtype=torch.long)
    if axis == "h":
        d = torch.zeros(1, c, height, 1, device=device, dtype=dtype)
        d[0, :, idx, 0] = comp
        return d
    if axis == "w":
        d = torch.zeros(1, c, 1, width, device=device, dtype=dtype)
        d[0, :, 0, idx] = comp
        return d
    raise ValueError(f"axis must be 'h' or 'w', got {axis!r}")


def expand_compact_row_bias(
    compact_bias: torch.Tensor,
    total_rows: int,
    selected_rows: list[int],
) -> torch.Tensor:
    """
    Expand compact [C, K] or [C, K, 1] bias into full [C, total_rows] or [C, total_rows, 1].
    Non-selected rows are zero.
    """
    if compact_bias.dim() not in (2, 3):
        raise ValueError(f"compact_bias must have dim 2 or 3, got {compact_bias.dim()}")
    if compact_bias.shape[1] != len(selected_rows):
        raise ValueError(
            f"compact second dim {compact_bias.shape[1]} != len(selected_rows) {len(selected_rows)}"
        )
    out_shape = list(compact_bias.shape)
    out_shape[1] = total_rows
    full_bias = torch.zeros(out_shape, dtype=compact_bias.dtype, device=compact_bias.device)
    full_bias[:, selected_rows, ...] = compact_bias
    return full_bias


def quantize_row_bias_int3_global(bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Global symmetric int3-style quantization (8 levels).

    Q in [-4, 3], scale = max|B|/4 + eps, B_hat = Q * scale.
    Unsigned symbols: Q_u = Q + 4 in [0, 7].
    """
    b = bias.detach().float()
    max_abs = b.abs().max().clamp_min(0.0)
    scale = max_abs / 4.0 + 1e-12
    q = torch.round(b / scale).clamp(-4, 3).to(torch.int64)
    bias_hat = q.float() * scale
    bias_hat = bias_hat.to(dtype=bias.dtype, device=bias.device)
    scale_tensor = scale.to(dtype=bias.dtype, device=bias.device)
    return q, scale_tensor, bias_hat


def quantize_row_bias_int5_global(bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Global symmetric int5-style quantization (32 levels).

    Q in [-16, 15], scale = max|B|/16 + eps, B_hat = Q * scale.
    Unsigned symbols: Q_u = Q + 16 in [0, 31].
    """
    b = bias.detach().float()
    max_abs = b.abs().max().clamp_min(0.0)
    scale = max_abs / 16.0 + 1e-12
    q = torch.round(b / scale).clamp(-16, 15).to(torch.int64)
    bias_hat = q.float() * scale
    bias_hat = bias_hat.to(dtype=bias.dtype, device=bias.device)
    scale_tensor = scale.to(dtype=bias.dtype, device=bias.device)
    return q, scale_tensor, bias_hat


def quantize_row_bias_int4_global(bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Global symmetric int4-style quantization (16 levels).

    bias: [C, H] e.g. [320, 16]
    Returns:
        q: int64 in [-8, 7]
        scale: scalar float tensor (same device/dtype as bias)
        bias_hat: dequantized tensor, same shape as bias
    """
    b = bias.detach().float()
    max_abs = b.abs().max().clamp_min(0.0)
    scale = max_abs / 7.0 + 1e-12
    q = torch.round(b / scale).clamp(-8, 7).to(torch.int64)
    bias_hat = q.float() * scale
    bias_hat = bias_hat.to(dtype=bias.dtype, device=bias.device)
    scale_tensor = scale.to(dtype=bias.dtype, device=bias.device)
    return q, scale_tensor, bias_hat


def compute_quant_metrics(bias: torch.Tensor, bias_hat: torch.Tensor) -> dict[str, float]:
    b = bias.detach().float().reshape(-1)
    bh = bias_hat.detach().float().reshape(-1)
    diff = b - bh
    mse = float((diff**2).mean().item())
    rmse = math.sqrt(mse) if mse > 0 else 0.0
    mae = float(diff.abs().mean().item())
    max_abs_err = float(diff.abs().max().item())
    return {
        "row_bias_max_abs": float(b.abs().max().item()),
        "row_bias_mean": float(b.mean().item()),
        "row_bias_std": float(b.std().item()),
        "quant_mse": mse,
        "quant_rmse": rmse,
        "quant_mae": mae,
        "quant_max_abs_error": max_abs_err,
    }


def _histogram_16(q: torch.Tensor) -> torch.Tensor:
    """q int64 in [-8, 7]; return counts for unsigned symbols 0..15."""
    qu = (q + 8).clamp(0, 15).to(torch.int64).reshape(-1)
    hist = torch.bincount(qu, minlength=16)
    return hist


def estimate_symbol_entropy_bits(q: torch.Tensor, scale_bits: int = 32) -> dict[str, float]:
    """Empirical entropy over 16 unsigned symbols + scale_bits."""
    hist = _histogram_16(q)
    n = int(hist.sum().item())
    if n == 0:
        return {"entropy_bits_per_symbol": 0.0, "estimated_entropy_bits": float(scale_bits)}
    p = hist.float() / float(n)
    mask = p > 0
    ent_per_sym = float(-(p[mask] * torch.log2(p[mask])).sum().item())
    return {
        "entropy_bits_per_symbol": ent_per_sym,
        "estimated_entropy_bits": ent_per_sym * float(n) + float(scale_bits),
    }


def _huffman_total_bits_from_counts(counts: list[int]) -> tuple[int, float]:
    """Optimal Huffman total bit length for multiset; per-symbol average over non-zero mass."""
    vals = [int(c) for c in counts if c > 0]
    n_syms = sum(vals)
    if n_syms == 0:
        return 0, 0.0
    if len(vals) == 1:
        return n_syms, 1.0
    pq = list(vals)
    heapq.heapify(pq)
    total = 0
    while len(pq) > 1:
        a = heapq.heappop(pq)
        b = heapq.heappop(pq)
        s = a + b
        total += s
        heapq.heappush(pq, s)
    per_sym = total / n_syms if n_syms else 0.0
    return total, per_sym


def _huffman_tree_header_bits(num_used_symbols: int, alphabet_symbol_bits: int) -> int:
    """
    Serialized Huffman tree cost in bits.

    We use a simple full-binary-tree preorder model:
    - 1 topology bit per node (leaf/internal), totaling ``2m - 1`` for ``m`` leaves.
    - ``alphabet_symbol_bits`` bits for each leaf label.
    """
    m = int(num_used_symbols)
    if m <= 0:
        return 0
    return (2 * m - 1) + m * int(alphabet_symbol_bits)


def _estimate_huffman_bits_from_counts(
    counts: list[int],
    *,
    alphabet_symbol_bits: int,
    scale_bits: int,
) -> dict[str, float]:
    huff_body, huff_per_sym = _huffman_total_bits_from_counts(counts)
    num_used_symbols = sum(1 for c in counts if int(c) > 0)
    tree_bits = _huffman_tree_header_bits(num_used_symbols, alphabet_symbol_bits)
    return {
        "huffman_num_used_symbols": float(num_used_symbols),
        "huffman_tree_bits": float(tree_bits),
        "huffman_bits_body": float(huff_body),
        "huffman_bits_per_symbol": float(huff_per_sym),
        "huffman_bits_total": float(huff_body + tree_bits + scale_bits),
    }


def estimate_huffman_bits(q: torch.Tensor, scale_bits: int = 32) -> dict[str, float]:
    hist = _histogram_16(q)
    counts = [int(x) for x in hist.tolist()]
    return _estimate_huffman_bits_from_counts(
        counts, alphabet_symbol_bits=4, scale_bits=scale_bits
    )


def _histogram_int3(q: torch.Tensor) -> torch.Tensor:
    qu = (q + 4).clamp(0, 7).to(torch.int64).reshape(-1)
    return torch.bincount(qu, minlength=8)


def estimate_symbol_entropy_bits_int3(q: torch.Tensor, scale_bits: int = 32) -> dict[str, float]:
    hist = _histogram_int3(q)
    n = int(hist.sum().item())
    if n == 0:
        return {"entropy_bits_per_symbol": 0.0, "estimated_entropy_bits": float(scale_bits)}
    p = hist.float() / float(n)
    mask = p > 0
    ent_per_sym = float(-(p[mask] * torch.log2(p[mask])).sum().item())
    return {
        "entropy_bits_per_symbol": ent_per_sym,
        "estimated_entropy_bits": ent_per_sym * float(n) + float(scale_bits),
    }


def estimate_huffman_bits_int3(q: torch.Tensor, scale_bits: int = 32) -> dict[str, float]:
    hist = _histogram_int3(q)
    counts = [int(x) for x in hist.tolist()]
    return _estimate_huffman_bits_from_counts(
        counts, alphabet_symbol_bits=3, scale_bits=scale_bits
    )


def _histogram_int5(q: torch.Tensor) -> torch.Tensor:
    qu = (q + 16).clamp(0, 31).to(torch.int64).reshape(-1)
    return torch.bincount(qu, minlength=32)


def estimate_symbol_entropy_bits_int5(q: torch.Tensor, scale_bits: int = 32) -> dict[str, float]:
    hist = _histogram_int5(q)
    n = int(hist.sum().item())
    if n == 0:
        return {"entropy_bits_per_symbol": 0.0, "estimated_entropy_bits": float(scale_bits)}
    p = hist.float() / float(n)
    mask = p > 0
    ent_per_sym = float(-(p[mask] * torch.log2(p[mask])).sum().item())
    return {
        "entropy_bits_per_symbol": ent_per_sym,
        "estimated_entropy_bits": ent_per_sym * float(n) + float(scale_bits),
    }


def estimate_huffman_bits_int5(q: torch.Tensor, scale_bits: int = 32) -> dict[str, float]:
    hist = _histogram_int5(q)
    counts = [int(x) for x in hist.tolist()]
    return _estimate_huffman_bits_from_counts(
        counts, alphabet_symbol_bits=5, scale_bits=scale_bits
    )


def rd_objective(mse: float, cbr_total: float, lambda_rd: float) -> float:
    """NTSCC-style scalar objective for logging: λ·255²·MSE + CBR."""
    return float(lambda_rd) * (255.0**2) * float(mse) + float(cbr_total)


def delta_cbr_kodak_fixed(side_bits: float) -> float:
    return float(side_bits) / KODAK_CBR_SIDE_DENOM


def raw_fixed_bits(num_symbols: int = 5120, bits_per_symbol: int = 4, scale_bits: int = 32) -> float:
    return float(num_symbols * bits_per_symbol + scale_bits)


def raw_fixed_bits_int3(num_symbols: int, scale_bits: int = 32) -> float:
    return float(num_symbols * 3 + scale_bits)


def raw_fixed_bits_int5(num_symbols: int, scale_bits: int = 32) -> float:
    return float(num_symbols * 5 + scale_bits)
