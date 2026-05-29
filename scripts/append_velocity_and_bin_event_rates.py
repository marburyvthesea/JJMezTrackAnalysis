#!/usr/bin/env python3
"""Append velocity to saved event-rate CSVs and summarize by velocity bins."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_VELOCITY_BINS = [(0, 0.5), (0.5, 2), (2, 5), (5, 10), (10, np.inf)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load per-mouse event-rate CSVs, append matching velocity columns "
            "from each mouse's GCAMP_with_velocity.csv, save the per-mouse "
            "augmented CSVs, and write a velocity-binned summary table."
        )
    )
    parser.add_argument(
        "events_csvs",
        nargs="+",
        help="One or more per-mouse event-rate CSVs to process.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help=(
            "Base directory containing m*_analysis folders. "
            "Default: current directory"
        ),
    )
    parser.add_argument(
        "--window-len",
        type=int,
        default=20,
        help="Moving-window length in samples used to generate the event-rate CSVs.",
    )
    parser.add_argument(
        "--velocity-column",
        default="Velocity_spatial_filtered",
        help="Column to load from GCAMP_with_velocity.csv. Default: Velocity_spatial_filtered",
    )
    parser.add_argument(
        "--per-mouse-output-name",
        default=None,
        help=(
            "Filename to use for each saved per-mouse augmented CSV. "
            "Defaults to eventsPerSecond_with_velocity_window<window_len>.csv"
        ),
    )
    parser.add_argument(
        "--summary-output-csv",
        default=None,
        help=(
            "Optional explicit output path for the velocity-binned summary CSV. "
            "Defaults under --base-dir with a timestamped filename."
        ),
    )
    return parser.parse_args()


def moving_window_nanmean(x: np.ndarray, window_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x).astype(float)
    x0 = np.where(np.isfinite(x), x, 0.0)

    num = np.convolve(x0, np.ones(window_len, dtype=float), mode="valid")
    den = np.convolve(valid, np.ones(window_len, dtype=float), mode="valid")

    return np.divide(
        num,
        den,
        out=np.full_like(num, np.nan, dtype=float),
        where=den > 0,
    )


def infer_mouse_from_path(csv_path: Path) -> str:
    parent_name = csv_path.parent.name
    if parent_name.endswith("_analysis"):
        return parent_name.replace("_analysis", "")
    raise ValueError(
        f"Could not infer mouse name from parent directory: {csv_path.parent}"
    )


def default_per_mouse_output_name(window_len: int) -> str:
    return f"eventsPerSecond_with_velocity_window{window_len}.csv"


def default_summary_output_csv(base_dir: Path, window_len: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"mouse_velocity_binned_window{window_len}_{timestamp}.csv"


def append_velocity_to_mouse_df(
    *,
    mouse_df: pd.DataFrame,
    gcamp_csv: Path,
    velocity_column: str,
    window_len: int,
) -> pd.DataFrame:
    window_start_idx = mouse_df.index.to_numpy(dtype=np.int64)
    velocity = pd.read_csv(
        gcamp_csv,
        usecols=[velocity_column],
        dtype={velocity_column: "float32"},
    )[velocity_column].to_numpy()

    velocity_window_mean = moving_window_nanmean(velocity, window_len)

    if window_start_idx.max() >= len(velocity_window_mean):
        raise ValueError(
            f"{gcamp_csv.parent.name}: window index exceeds available windowed "
            f"velocity length ({window_start_idx.max()} vs {len(velocity_window_mean) - 1})"
        )

    velocity_meta = pd.DataFrame(
        {
            velocity_column: velocity[window_start_idx],
            f"{velocity_column}_window_mean": velocity_window_mean[window_start_idx],
        },
        index=mouse_df.index,
    )
    velocity_meta.index.name = "window_start_idx"

    return mouse_df.join(velocity_meta)


def bin_mouse_df(
    mouse_df: pd.DataFrame,
    mouse: str,
    velocity_bins: list[tuple[float, float]],
    velocity_window_mean_col: str,
) -> pd.DataFrame:
    bin_edges = [b[0] for b in velocity_bins] + [velocity_bins[-1][1]]
    bin_labels = [
        f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+"
        for lo, hi in velocity_bins
    ]
    label_to_bounds = dict(zip(bin_labels, velocity_bins))

    cell_cols = [c for c in mouse_df.columns if str(c).startswith("cell_")]
    if not cell_cols:
        raise ValueError(f"{mouse}: no cell_* columns found in event-rate CSV")

    mean_events = mouse_df[cell_cols].mean(axis=1, skipna=True)
    velocity_for_binning = pd.to_numeric(
        mouse_df[velocity_window_mean_col],
        errors="coerce",
    )
    velocity_bin = pd.cut(
        velocity_for_binning,
        bins=bin_edges,
        labels=bin_labels,
        right=False,
        include_lowest=True,
    )

    tmp = pd.DataFrame(
        {
            "mean_events_per_second_across_cells": mean_events,
            "velocity_for_binning": velocity_for_binning,
            "velocity_bin": velocity_bin,
        }
    )

    summary = (
        tmp.dropna(subset=["velocity_bin"])
        .groupby("velocity_bin", observed=False)
        .agg(
            mean_events_per_second=("mean_events_per_second_across_cells", "mean"),
            sem_events_per_second=("mean_events_per_second_across_cells", "sem"),
            n_windows=("mean_events_per_second_across_cells", "size"),
            mean_velocity=("velocity_for_binning", "mean"),
        )
        .reset_index()
    )

    summary["mouse"] = mouse
    summary["speed_bin_low"] = summary["velocity_bin"].map(
        lambda lbl: label_to_bounds[str(lbl)][0]
    )
    summary["speed_bin_high"] = summary["velocity_bin"].map(
        lambda lbl: label_to_bounds[str(lbl)][1]
    )
    return summary


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir).resolve()
    per_mouse_output_name = (
        args.per_mouse_output_name
        if args.per_mouse_output_name is not None
        else default_per_mouse_output_name(args.window_len)
    )
    summary_output_csv = (
        Path(args.summary_output_csv).resolve()
        if args.summary_output_csv is not None
        else default_summary_output_csv(base_dir, args.window_len)
    )

    mouse_bin_rows: list[pd.DataFrame] = []

    for csv_str in args.events_csvs:
        csv_path = Path(csv_str).resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"Event-rate CSV not found: {csv_path}")

        mouse = infer_mouse_from_path(csv_path)
        print(mouse, flush=True)

        mouse_df = pd.read_csv(csv_path, index_col=0)
        mouse_df.index.name = "window_start_idx"

        gcamp_csv = base_dir / f"{mouse}_analysis" / "GCAMP_with_velocity.csv"
        if not gcamp_csv.is_file():
            raise FileNotFoundError(f"GCAMP_with_velocity.csv not found: {gcamp_csv}")

        print("calculating velocity_window_mean", flush=True)
        mouse_with_velocity_df = append_velocity_to_mouse_df(
            mouse_df=mouse_df,
            gcamp_csv=gcamp_csv,
            velocity_column=args.velocity_column,
            window_len=args.window_len,
        )

        per_mouse_output_csv = base_dir / f"{mouse}_analysis" / per_mouse_output_name
        per_mouse_output_csv.parent.mkdir(parents=True, exist_ok=True)
        print("saving", flush=True)
        mouse_with_velocity_df.to_csv(per_mouse_output_csv, index=True)
        print(f"saved: {per_mouse_output_csv}", flush=True)

        print("binning velocity", flush=True)
        summary = bin_mouse_df(
            mouse_df=mouse_with_velocity_df,
            mouse=mouse,
            velocity_bins=DEFAULT_VELOCITY_BINS,
            velocity_window_mean_col=f"{args.velocity_column}_window_mean",
        )
        mouse_bin_rows.append(summary)

    mouse_velocity_binned_df = pd.concat(mouse_bin_rows, ignore_index=True)
    summary_output_csv.parent.mkdir(parents=True, exist_ok=True)
    mouse_velocity_binned_df.to_csv(summary_output_csv, index=False)

    print(f"Summary output: {summary_output_csv}", flush=True)
    print(f"Summary shape:  {mouse_velocity_binned_df.shape}", flush=True)


if __name__ == "__main__":
    main()
