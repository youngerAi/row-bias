from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROW_BIAS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROW_BIAS_ROOT.parent


def _resolve_default_ntscc_repo_root() -> Path:
    env_value = os.environ.get("NTSCC_REPO_ROOT")
    if env_value:
        return Path(env_value)

    sibling_repo = WORKSPACE_ROOT / "NTSCC_plus"
    if sibling_repo.exists():
        return sibling_repo

    globecom_repo = WORKSPACE_ROOT / "globecom2026" / "NTSCC_plus"
    if globecom_repo.exists():
        return globecom_repo

    return Path("/data/project/NTSCC_plus")


def _resolve_default_dataset_root() -> Path:
    env_value = os.environ.get("NTSCC_DATASET_ROOT")
    if env_value:
        return Path(env_value)

    sibling_dataset = WORKSPACE_ROOT / "Dataset"
    if sibling_dataset.exists():
        return sibling_dataset

    globecom_dataset = WORKSPACE_ROOT / "globecom2026" / "Dataset"
    if globecom_dataset.exists():
        return globecom_dataset

    return Path("/data/project/Dataset")


DEFAULT_NTSCC_REPO_ROOT = _resolve_default_ntscc_repo_root()
DEFAULT_CHECKPOINT_DIR = DEFAULT_NTSCC_REPO_ROOT / "checkpoint"
DEFAULT_DATASET_ROOT = _resolve_default_dataset_root()
DEFAULT_KODAK_DIR = DEFAULT_DATASET_ROOT / "kodak"
DEFAULT_OUTPUT_ROOT = ROW_BIAS_ROOT / "results"

DEFAULT_FIXED_SNR = 10.0
DEFAULT_FIXED_ETA = 0.2
DEFAULT_MULTIPLE_RATE = [
    1,
    4,
    8,
    12,
    16,
    20,
    24,
    32,
    48,
    64,
    80,
    96,
    112,
    128,
    144,
    160,
    176,
    192,
    208,
    224,
    240,
    256,
    272,
    288,
    304,
    320,
]

CHECKPOINT_SPECS = [
    {"idx": 0, "lambda": 0.013, "filename": "ckbd2_lmbd_0.013.pth.tar"},
    {"idx": 1, "lambda": 0.0483, "filename": "ckbd2_lmbd_0.0483.pth.tar"},
    {"idx": 2, "lambda": 0.18, "filename": "ckbd2_lmbd_0.18.pth.tar"},
    {"idx": 3, "lambda": 0.36, "filename": "ckbd2_lmbd_0.36.pth.tar"},
    {"idx": 4, "lambda": 0.72, "filename": "ckbd2_lmbd_0.72.pth.tar"},
]


@dataclass(frozen=True)
class RuntimePaths:
    ntscc_repo_root: Path = DEFAULT_NTSCC_REPO_ROOT
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    dataset_root: Path = DEFAULT_DATASET_ROOT
    default_kodak_dir: Path = DEFAULT_KODAK_DIR
    output_root: Path = DEFAULT_OUTPUT_ROOT


def parse_checkpoint_indices(raw: str) -> list[int]:
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def get_checkpoint_specs(indices: list[int], checkpoint_dir: str | Path) -> list[dict[str, object]]:
    checkpoint_dir = Path(checkpoint_dir)
    specs: list[dict[str, object]] = []
    for index in indices:
        base_spec = CHECKPOINT_SPECS[index]
        specs.append(
            {
                "idx": int(base_spec["idx"]),
                "lambda": float(base_spec["lambda"]),
                "filename": str(base_spec["filename"]),
                "path": checkpoint_dir / str(base_spec["filename"]),
            }
        )
    return specs
