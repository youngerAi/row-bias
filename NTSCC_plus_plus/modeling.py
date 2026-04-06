from __future__ import annotations

import torch
import torch.nn as nn

from row_bias_ntscc.modeling import (
    encode_to_latent,
    forward_from_y as ntsccpp_forward_from_y,
    load_ntscc_plus_model,
)


ENCODER_MODULES = ("f_e0", "trans_sep_enc", "trans_ctx_enc")


def freeze_non_adaptation_params(model: nn.Module) -> list[nn.Parameter]:
    """Freeze everything except the encoder stack tuned by NTSCC++."""
    for param in model.parameters():
        param.requires_grad = False

    trainable: list[nn.Parameter] = []
    for name in ENCODER_MODULES:
        module = getattr(model, name)
        for param in module.parameters():
            param.requires_grad = True
            trainable.append(param)
    return trainable


__all__ = [
    "ENCODER_MODULES",
    "encode_to_latent",
    "freeze_non_adaptation_params",
    "load_ntscc_plus_model",
    "ntsccpp_forward_from_y",
]
