#!/usr/bin/env python3
"""Public entry point for the current NTSCC+ fine-tuning release.

This wrapper exposes a clean CLI for the main public workflow:
- baseline NTSCC+
- lightweight test-time fine-tuning of the pre-powernorm signal ``s``
- optional comparison against NTSCC++

Internally it reuses the research trajectory logger under
``test_time_eval/collect_time_trajectories.py`` and then writes a
compact final-step summary CSV for easier reporting.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
COLLECT_SCRIPT = REPO_ROOT / "test_time_eval" / "collect_time_trajectories.py"
RAW_METRICS_FILENAME = "raw_step_metrics.csv"
SUMMARY_FILENAME = "summary_last_step.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate NTSCC+ test-time fine-tuning and optionally compare against NTSCC++."
    )
    parser.add_argument("--ntscc_repo_root", default="")
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--checkpoint_indices", default="0,1,2,3,4")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_images", type=int, default=0)
    parser.add_argument("--snr_db", type=float, default=10.0)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--index_sideinfo_codec", choices=("auto", "flif", "png", "none"), default="png")
    parser.add_argument("--side_cbr_no_capacity", action="store_true")
    parser.add_argument("--s_steps", type=int, default=20)
    parser.add_argument("--s_lr", type=float, default=0.1)
    parser.add_argument("--compare_ntsccpp", action="store_true")
    parser.add_argument("--ntsccpp_steps", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _parse_int(value: str) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def summarize_last_step(raw_csv_path: Path, summary_csv_path: Path) -> None:
    with raw_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    last_rows: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row["method"],
            row["checkpoint_idx"],
            row["lambda"],
            row["image_name"],
        )
        step = _parse_int(row["step"])
        existing = last_rows.get(key)
        if existing is None or step >= _parse_int(existing["step"]):
            last_rows[key] = row

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in last_rows.values():
        grouped[(row["method"], row["checkpoint_idx"], row["lambda"])].append(row)

    fieldnames = [
        "method",
        "checkpoint_idx",
        "lambda",
        "num_images",
        "avg_psnr",
        "avg_cbr_total",
        "avg_runtime_s",
        "avg_peak_memory_mb",
        "avg_tuned_param_count",
    ]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (method, checkpoint_idx, lambda_value), group_rows in sorted(
            grouped.items(), key=lambda item: (item[0][0], int(item[0][1]))
        ):
            count = len(group_rows)
            avg_psnr = sum(_parse_float(row["psnr"]) for row in group_rows) / count
            avg_cbr_total = sum(_parse_float(row["cbr_total"]) for row in group_rows) / count
            avg_runtime_s = sum(_parse_float(row["cumulative_compute_time_s"]) for row in group_rows) / count
            avg_peak_memory_mb = sum(_parse_float(row["peak_memory_mb"]) for row in group_rows) / count
            avg_tuned_param_count = sum(_parse_float(row["tuned_param_count"]) for row in group_rows) / count
            writer.writerow(
                {
                    "method": method,
                    "checkpoint_idx": checkpoint_idx,
                    "lambda": lambda_value,
                    "num_images": count,
                    "avg_psnr": f"{avg_psnr:.6f}",
                    "avg_cbr_total": f"{avg_cbr_total:.6f}",
                    "avg_runtime_s": f"{avg_runtime_s:.6f}",
                    "avg_peak_memory_mb": f"{avg_peak_memory_mb:.6f}",
                    "avg_tuned_param_count": f"{avg_tuned_param_count:.2f}",
                }
            )


def build_collect_command(args: argparse.Namespace) -> list[str]:
    methods = ["baseline", "s_only"]
    if args.compare_ntsccpp:
        methods.append("ntsccpp")

    command = [
        sys.executable,
        str(COLLECT_SCRIPT),
        "--image_dir",
        args.image_dir,
        "--output_dir",
        args.output_dir,
        "--checkpoint_indices",
        args.checkpoint_indices,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--num_images",
        str(args.num_images),
        "--snr_db",
        str(args.snr_db),
        "--eta",
        str(args.eta),
        "--index_sideinfo_codec",
        args.index_sideinfo_codec,
        "--methods",
        ",".join(methods),
        "--s_steps",
        str(args.s_steps),
        "--ntsccpp_steps",
        str(args.ntsccpp_steps),
        "--s_lr",
        str(args.s_lr),
        "--overwrite",
    ]

    if args.ntscc_repo_root:
        command += ["--ntscc_repo_root", args.ntscc_repo_root]
    if args.checkpoint_dir:
        command += ["--checkpoint_dir", args.checkpoint_dir]
    if args.side_cbr_no_capacity:
        command.append("--side_cbr_no_capacity")
    return command


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_collect_command(args)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))

    raw_csv_path = output_dir / RAW_METRICS_FILENAME
    summary_csv_path = output_dir / SUMMARY_FILENAME
    summarize_last_step(raw_csv_path, summary_csv_path)
    print(f"Saved summary to {summary_csv_path}", flush=True)


if __name__ == "__main__":
    main()
