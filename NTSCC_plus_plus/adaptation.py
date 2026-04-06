from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from row_bias_ntscc.dataset import CropInfo, crop_back, pad_image_to_multiple
from row_bias_ntscc.metrics import psnr_from_mse

from .modeling import encode_to_latent, freeze_non_adaptation_params, ntsccpp_forward_from_y


ENCODER_STATE_PREFIXES = ("f_e0.", "trans_sep_enc.", "trans_ctx_enc.")


@dataclass
class AdaptConfig:
    lambda_rd: float
    steps: int = 20
    latent_lr: float = 5e-3
    encoder_lr: float = 1e-4
    grad_clip_norm: float = 1.0
    deterministic_quant_eval: bool = False
    train_with_noise: bool = True


class AdaptLoss(nn.Module):
    """Original NTSCC++ RD objective: lambda * 255^2 * MSE + bpp."""

    def __init__(self, lambda_rd: float):
        super().__init__()
        self.lambda_rd = float(lambda_rd)

    def forward(self, output: dict[str, Any], target: torch.Tensor, *, num_pixels: int) -> dict[str, torch.Tensor]:
        n, _, h, w = target.shape
        out: dict[str, torch.Tensor] = {}
        out["bpp_y"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2.0) * num_pixels)
        out["bpp_z"] = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2.0) * num_pixels)
        out["bpp"] = out["bpp_y"] + out["bpp_z"]
        out["cbr"] = output["k"] / (n * 3 * h * w)
        out["mse_loss"] = F.mse_loss(output["x_hat"], target)
        out["objective"] = self.lambda_rd * (255.0**2) * out["mse_loss"] + out["bpp"]
        return out


def capture_online_state(model: nn.Module, y_param: torch.nn.Parameter) -> dict[str, Any]:
    return {
        "y": y_param.detach().cpu().clone(),
        "modules": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if any(key.startswith(prefix) for prefix in ENCODER_STATE_PREFIXES)
        },
    }


def restore_online_state(model: nn.Module, y_param: torch.nn.Parameter, state: dict[str, Any]) -> None:
    with torch.no_grad():
        y_param.copy_(state["y"].to(y_param.device, dtype=y_param.dtype))
    current = model.state_dict()
    current.update(state["modules"])
    model.load_state_dict(current, strict=False)


def evaluate_output_record(
    output: dict[str, Any],
    image: torch.Tensor,
    *,
    crop: CropInfo,
    criterion: AdaptLoss,
) -> dict[str, float]:
    num_pixels = crop.padded_h * crop.padded_w
    output = dict(output)
    output["x_hat"] = crop_back(output["x_hat"], crop)
    loss = criterion(output, image, num_pixels=num_pixels)
    return {
        "psnr": psnr_from_mse(float(loss["mse_loss"].detach().cpu().item())),
        "mse": float(loss["mse_loss"].detach().cpu().item()),
        "bpp_y": float(loss["bpp_y"].detach().cpu().item()),
        "bpp_z": float(loss["bpp_z"].detach().cpu().item()),
        "bpp": float(loss["bpp"].detach().cpu().item()),
        "cbr_total": float(loss["cbr"].detach().cpu().item()),
        "objective": float(loss["objective"].detach().cpu().item()),
    }


def adapt_ntsccpp(
    model: nn.Module,
    image: torch.Tensor,
    cfg: AdaptConfig,
    *,
    repo_root: str | None = None,
) -> tuple[dict[str, float], dict[str, Any], dict[str, float]]:
    """Full online tuning of latent y + encoder stack f_e0/trans_sep_enc/trans_ctx_enc."""
    device = next(model.parameters()).device
    image = image.to(device)
    image_pad, crop_info = pad_image_to_multiple(image, multiple=256)
    criterion = AdaptLoss(lambda_rd=cfg.lambda_rd)
    repo_root = "" if repo_root is None else repo_root

    encoder_params = freeze_non_adaptation_params(model)
    model.eval()
    with torch.no_grad():
        y_init = encode_to_latent(model, image_pad)
    y_param = nn.Parameter(y_init.detach().clone(), requires_grad=True)
    optimizer = torch.optim.Adam(
        [
            {"params": encoder_params, "lr": float(cfg.encoder_lr)},
            {"params": [y_param], "lr": float(cfg.latent_lr)},
        ]
    )
    tuned_param_count = float(sum(param.numel() for param in encoder_params) + y_param.numel())

    with torch.no_grad():
        init_output = ntsccpp_forward_from_y(
            model,
            y_param,
            repo_root=repo_root,
            no_channel_noise=(not cfg.train_with_noise),
            deterministic_quant=cfg.deterministic_quant_eval,
        )
    init_eval = evaluate_output_record(init_output, image, crop=crop_info, criterion=criterion)
    best_score = float(init_eval["objective"])
    best_state = capture_online_state(model, y_param)

    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        output = ntsccpp_forward_from_y(
            model,
            y_param,
            repo_root=repo_root,
            no_channel_noise=(not cfg.train_with_noise),
            deterministic_quant=False,
        )
        output["x_hat"] = crop_back(output["x_hat"], crop_info)
        loss = criterion(output, image, num_pixels=crop_info.padded_h * crop_info.padded_w)
        loss["objective"].backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(encoder_params + [y_param], cfg.grad_clip_norm)
        optimizer.step()

        with torch.no_grad():
            eval_output = ntsccpp_forward_from_y(
                model,
                y_param,
                repo_root=repo_root,
                no_channel_noise=False,
                deterministic_quant=cfg.deterministic_quant_eval,
            )
        eval_rec = evaluate_output_record(eval_output, image, crop=crop_info, criterion=criterion)
        if float(eval_rec["objective"]) < best_score:
            best_score = float(eval_rec["objective"])
            best_state = capture_online_state(model, y_param)

    restore_online_state(model, y_param, best_state)
    with torch.no_grad():
        best_output = ntsccpp_forward_from_y(
            model,
            y_param,
            repo_root=repo_root,
            no_channel_noise=False,
            deterministic_quant=cfg.deterministic_quant_eval,
        )
    return (
        evaluate_output_record(best_output, image, crop=crop_info, criterion=criterion),
        best_output,
        {"tuned_param_count": tuned_param_count},
    )


__all__ = [
    "AdaptConfig",
    "AdaptLoss",
    "adapt_ntsccpp",
    "capture_online_state",
    "evaluate_output_record",
    "restore_online_state",
]
