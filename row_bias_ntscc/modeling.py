from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from .config import DEFAULT_MULTIPLE_RATE

_NTSCC_SYMBOLS: dict[str, Any] = {}


def _ensure_repo_on_path(repo_root: str | Path) -> Path:
    repo_root = Path(repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _get_ntscc_symbols(repo_root: str | Path) -> dict[str, Any]:
    repo_root = _ensure_repo_on_path(repo_root)
    cache_key = str(repo_root)
    if cache_key not in _NTSCC_SYMBOLS:
        from net.ntscc import NTSCC_plus  # type: ignore
        from net.utils import DEMUX, MUX  # type: ignore

        _NTSCC_SYMBOLS[cache_key] = {
            "NTSCC_plus": NTSCC_plus,
            "DEMUX": DEMUX,
            "MUX": MUX,
        }
    return _NTSCC_SYMBOLS[cache_key]


def build_runtime_config(*, snr: float, eta: float, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        channel_type="awgn",
        SNR=float(snr),
        device=device,
        logger=False,
        eta=float(eta),
        multiple_rate=DEFAULT_MULTIPLE_RATE,
    )


def load_checkpoint_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    return {key: value for key, value in state_dict.items() if "mask" not in key}


def load_ntscc_plus_model(
    checkpoint_path: str | Path,
    *,
    repo_root: str | Path,
    snr: float,
    eta: float,
    device: str,
) -> tuple[nn.Module, SimpleNamespace]:
    symbols = _get_ntscc_symbols(repo_root)
    config = build_runtime_config(snr=snr, eta=eta, device=device)
    with contextlib.redirect_stdout(io.StringIO()):
        model = symbols["NTSCC_plus"](config)
    model.load_state_dict(load_checkpoint_state(checkpoint_path), strict=False)
    model = model.to(device)
    model.eval()
    return model, config


def encode_to_latent(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    return model.ntc.g_a(image)


def forward_from_y(
    model: nn.Module,
    y: torch.Tensor,
    *,
    repo_root: str | Path,
    no_channel_noise: bool = False,
    deterministic_quant: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Full NTSCC+ forward from latent y (same path as Q-LoRA `ntsccpp_forward_from_y`):
    hyperprior + JSCC encode + channel + f_d + trans_* + g_s.
    """
    demux = _get_ntscc_symbols(repo_root)["DEMUX"]
    mux = _get_ntscc_symbols(repo_root)["MUX"]

    z = model.ntc.h_a(y)
    z_tilde, z_likelihoods = model.ntc.entropy_bottleneck(z, training=not deterministic_quant)
    params = model.ntc.h_s(z_tilde)

    quant_mode = "dequantize" if deterministic_quant else "noise"
    y_tilde = model.ntc.gaussian_conditional.quantize(y, quant_mode)
    y_half = y_tilde.clone()
    x_hat_ntc = model.ntc.g_s(y_tilde)

    y_half[:, :, 0::2, 0::2] = 0
    y_half[:, :, 1::2, 1::2] = 0
    sc_params = model.ntc.sc_transform(y_half)
    sc_params[:, :, 0::2, 1::2] = 0
    sc_params[:, :, 1::2, 0::2] = 0
    gaussian_params = model.ntc.entropy_parameters(torch.cat((params, sc_params), dim=1))
    scales_hat, means_hat = gaussian_params.chunk(2, 1)
    _, y_likelihoods = model.ntc.gaussian_conditional(y, scales_hat, means=means_hat)

    likelihoods_non_anchor, likelihoods_anchor = demux(y_likelihoods)
    y_non_anchor, y_anchor = demux(y)

    y_anchor_sep = model.trans_sep_enc(y_anchor)
    y_non_anchor_sep = model.trans_sep_enc(y_non_anchor)
    y_non_anchor_ctx = model.trans_ctx_enc(y_non_anchor_sep, y_anchor_sep)

    y_concat = torch.cat([y_anchor_sep, y_non_anchor_ctx], dim=0)
    likelihoods_concat = torch.cat([likelihoods_anchor, likelihoods_non_anchor], dim=0)
    s_masked, mask, indexes = model.f_e(y_concat, likelihoods_concat, model.eta)

    if no_channel_noise:
        s_hat = s_masked
        channel_usage = mask.float().sum()
    else:
        s_hat, channel_usage = model.feature_pass_channel(s_masked, mask)

    y_hat_concat = model.f_d(s_hat, indexes)
    y_hat_anchor, y_hat_non_anchor = y_hat_concat.chunk(2, 0)
    y_hat_anchor = model.trans_sep_dec(y_hat_anchor)
    y_hat_non_anchor = model.trans_ctx_dec(y_hat_non_anchor, y_hat_anchor)
    y_hat_non_anchor = model.trans_sep_dec(y_hat_non_anchor)
    y_hat = mux(y_hat_non_anchor, y_hat_anchor)
    x_hat = model.ntc.g_s(y_hat)

    return {
        "x_hat": x_hat,
        "x_hat_ntc": x_hat_ntc,
        "indexes": indexes,
        "k": channel_usage,
        "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
    }


def extract_pre_powernorm_signal(
    model: nn.Module,
    latent: torch.Tensor,
    *,
    repo_root: str | Path,
    deterministic_quant: bool = False,
) -> dict[str, torch.Tensor]:
    demux = _get_ntscc_symbols(repo_root)["DEMUX"]

    z = model.ntc.h_a(latent)
    z_tilde, z_likelihoods = model.ntc.entropy_bottleneck(z, training=not deterministic_quant)
    params = model.ntc.h_s(z_tilde)

    quant_mode = "dequantize" if deterministic_quant else "noise"
    y_tilde = model.ntc.gaussian_conditional.quantize(latent, quant_mode)
    x_hat_ntc = model.ntc.g_s(y_tilde)
    y_half = y_tilde.clone()
    y_half[:, :, 0::2, 0::2] = 0
    y_half[:, :, 1::2, 1::2] = 0
    sc_params = model.ntc.sc_transform(y_half)
    sc_params[:, :, 0::2, 1::2] = 0
    sc_params[:, :, 1::2, 0::2] = 0
    gaussian_params = model.ntc.entropy_parameters(torch.cat((params, sc_params), dim=1))
    scales_hat, means_hat = gaussian_params.chunk(2, 1)
    _, y_likelihoods = model.ntc.gaussian_conditional(latent, scales_hat, means=means_hat)

    likelihoods_non_anchor, likelihoods_anchor = demux(y_likelihoods)
    y_non_anchor, y_anchor = demux(latent)
    y_anchor_sep = model.trans_sep_enc(y_anchor)
    y_non_anchor_sep = model.trans_sep_enc(y_non_anchor)
    y_non_anchor_ctx = model.trans_ctx_enc(y_non_anchor_sep, y_anchor_sep)

    y_concat = torch.cat([y_anchor_sep, y_non_anchor_ctx], dim=0)
    likelihoods_concat = torch.cat([likelihoods_anchor, likelihoods_non_anchor], dim=0)
    s_masked, mask, indexes = model.f_e(y_concat, likelihoods_concat, model.eta)
    return {
        "s_masked": s_masked,
        "mask": mask,
        "indexes": indexes,
        "x_hat_ntc": x_hat_ntc,
        "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
    }


def decode_from_pre_powernorm_signal(
    model: nn.Module,
    s: torch.Tensor,
    mask: torch.Tensor,
    indexes: torch.Tensor,
    *,
    repo_root: str | Path,
) -> dict[str, torch.Tensor]:
    s_hat, channel_usage = model.feature_pass_channel(s, mask)
    y_hat_concat = model.f_d(s_hat, indexes)
    decoded = decode_from_y_hat_concat(model, y_hat_concat, repo_root=repo_root)
    return {
        "x_hat": decoded["x_hat"],
        "k": channel_usage,
        "s_hat": s_hat,
        "y_hat_concat": y_hat_concat,
    }


def decode_from_y_hat_concat(
    model: nn.Module,
    y_hat_concat: torch.Tensor,
    *,
    repo_root: str | Path,
) -> dict[str, torch.Tensor]:
    mux = _get_ntscc_symbols(repo_root)["MUX"]

    y_hat_anchor_in, y_hat_non_anchor_in = y_hat_concat.chunk(2, 0)
    y_hat_anchor = model.trans_sep_dec(y_hat_anchor_in)
    y_hat_non_anchor = model.trans_ctx_dec(y_hat_non_anchor_in, y_hat_anchor)
    y_hat_non_anchor = model.trans_sep_dec(y_hat_non_anchor)
    y_hat = mux(y_hat_non_anchor, y_hat_anchor)
    x_hat = model.ntc.g_s(y_hat)
    return {
        "x_hat": x_hat,
        "y_hat_anchor_in": y_hat_anchor_in,
        "y_hat_non_anchor_in": y_hat_non_anchor_in,
        "y_hat_anchor": y_hat_anchor,
        "y_hat_non_anchor": y_hat_non_anchor,
        "y_hat": y_hat,
    }
