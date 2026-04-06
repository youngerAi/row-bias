from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class CropInfo:
    left: int
    right: int
    top: int
    bottom: int
    orig_h: int
    orig_w: int
    padded_h: int
    padded_w: int


class ImageFolderDataset(Dataset[torch.Tensor]):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.files = sorted(
            [*self.root.glob("*.png"), *self.root.glob("*.jpg"), *self.root.glob("*.jpeg")]
        )
        self.to_tensor = transforms.ToTensor()
        if not self.files:
            raise FileNotFoundError(f"No PNG/JPG images found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.files[index]).convert("RGB")
        return self.to_tensor(image)


def pad_image_to_multiple(x: torch.Tensor, multiple: int = 256) -> tuple[torch.Tensor, CropInfo]:
    _, _, h, w = x.shape
    new_h = (h + multiple - 1) // multiple * multiple
    new_w = (w + multiple - 1) // multiple * multiple
    left = (new_w - w) // 2
    right = new_w - w - left
    top = (new_h - h) // 2
    bottom = new_h - h - top
    padded = F.pad(x, (left, right, top, bottom), mode="constant", value=0.0)
    crop = CropInfo(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        orig_h=h,
        orig_w=w,
        padded_h=new_h,
        padded_w=new_w,
    )
    return padded, crop


def crop_back(x: torch.Tensor, crop: CropInfo) -> torch.Tensor:
    return F.pad(x, (-crop.left, -crop.right, -crop.top, -crop.bottom))
