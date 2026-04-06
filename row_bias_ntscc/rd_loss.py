"""Rate–distortion objective aligned with Q-LoRA / NTSCC+ training (main.py RateDistortionLoss)."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn


class AdaptLoss(nn.Module):
    """
    objective = lambda_rd * (255**2) * MSE(x_hat, x) + bpp_y + bpp_z

    bpp_* are in bits per pixel of the **target** image canvas (num_pixels).
    """

    def __init__(self, lambda_rd: float) -> None:
        super().__init__()
        self.lambda_rd = float(lambda_rd)
        self.mse = nn.MSELoss()

    def forward(
        self,
        output: dict[str, Any],
        target: torch.Tensor,
        *,
        num_pixels: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        n, _, h, w = target.size()
        if num_pixels is None:
            num_pixels = n * h * w
        out: dict[str, torch.Tensor] = {}
        out["bpp_y"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["bpp_z"] = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)
        out["bpp"] = out["bpp_y"] + out["bpp_z"]
        out["cbr"] = output["k"] / (n * 3 * h * w)
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["objective"] = self.lambda_rd * (255.0**2) * out["mse_loss"] + out["bpp"]
        return out
