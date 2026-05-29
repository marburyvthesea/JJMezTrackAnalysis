#!/usr/bin/env python3
"""Compute moving-window event counts per cell from a peak-stats CSV."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a peak-stats CSV, compute moving-window event counts from "
            "onset_idx for each included cell, and save the result as a CSV."
        )
    )
    parser.add_argument(
        "--peak-stats-csv",
        required=True,
        help="Path to all_peak_stats_*.csv",
    )
    parser.add_argument(
        "--window-len",
        type=int,
        default=20,
        help="Moving-window length in samples. Default: 20",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output path. Defaults beside the input CSV.",
    )
    parser.add_argument(
        "--gcamp-csv",
        default=None,
        help=(
            "Optional GCAMP_with_velocity.csv used to determine n_rows. "
            "Defaults to a sibling file in the peak-stats directory."
        ),
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help=(
            "Explicit total trace length. If omitted, count rows from "
            "GCAMP_with_velocity.csv."
        ),
    )
    parser.add_argument(
        "--labels-mat",
        default=None,
        help="Optional ActSort precomputed_output_LABELS.mat for filtering.",
    )
    parser.add_argument(
        "--label-source",
        choices=("ex", "ml", "overall", "labels_ex", "labels_ml", "labels_overall"),
        default="overall",
        help="Which label field to use when filtering. Default: overall",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Rows per CSV chunk while reading peak stats. Default: 50000",
    )
    return parser.parse_args()


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r") as handle:
        return sum(1 for _ in handle) - 1


def normalize_label_source(label_source: str) -> str:
    mapping = {
        "ex": "labels_ex",
        "ml": "labels_ml",
        "overall": "labels_overall",
        "labels_ex": "labels_ex",
        "labels_ml": "labels_ml",
        "labels_overall": "labels_overall",
    }
    return mapping[label_source]


def natural_cell_key(cell_name: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"cell_(\d+)", str(cell_name))
    if match:
        return (0, int(match.group(1)))
    return (1, str(cell_name))


def list_cells_in_peak_stats(stats_path: Path, chunksize: int) -> list[str]:
    seen: set[str] = set()
    for chunk in pd.read_csv(stats_path, usecols=["cell"], chunksize=chunksize):
        seen.update(chunk["cell"].dropna().astype(str).unique())
    return sorted(seen, key=natural_cell_key)


def load_good_cells_from_labels(labels_path: Path, label_source: str) -> list[str]:
    mat = sio.loadmat(labels_path, squeeze_me=True, struct_as_record=False)
    labels_struct = mat["labels"]
    label_attr = normalize_label_source(label_source)
    labels = np.asarray(getattr(labels_struct, label_attr)).squeeze().astype(int)
    good_python_indices = np.flatnonzero(labels == 1)
    return [f"cell_{idx + 1}" for idx in good_python_indices]


def infer_gcamp_csv(peak_stats_csv: Path, gcamp_csv: str | None) -> Path:
    if gcamp_csv is not None:
        return Path(gcamp_csv)
    return peak_stats_csv.parent / "GCAMP_with_velocity.csv"


def infer_output_csv(
    peak_stats_csv: Path,
    output_csv: str | None,
    window_len: int,
    label_source: str | None,
) -> Path:
    if output_csv is not None:
        return Path(output_csv)

    suffix = f"window{window_len}"
    if label_source is not None:
        suffix = f"{suffix}_{normalize_label_source(label_source)}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return peak_stats_csv.with_name(
        f"{peak_stats_csv.stem}_events_per_second_{suffix}_{timestamp}.csv"
    )


def collect_onsets_by_cell(
    peak_stats_csv: Path,
    n_rows: int,
    chunksize: int,
    included_cells: set[str] | None,
) -> dict[str, list[np.ndarray]]:
    onsets_by_cell: dict[str, list[np.ndarray]] = defaultdict(list)

    for chunk in pd.read_csv(
        peak_stats_csv,
        usecols=["cell", "onset_idx"],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk = chunk.dropna(subset=["cell", "onset_idx"]).copy()
        chunk["cell"] = chunk["cell"].astype(str)
        chunk["onset_idx"] = pd.to_numeric(chunk["onset_idx"], errors="coerce")
        chunk = chunk.dropna(subset=["onset_idx"])
        chunk["onset_idx"] = chunk["onset_idx"].astype(np.int64)

        chunk = chunk.loc[
            (chunk["onset_idx"] >= 0) & (chunk["onset_idx"] < n_rows)
        ]
        if included_cells is not None:
            chunk = chunk.loc[chunk["cell"].isin(included_cells)]

        for cell_name, sub_df in chunk.groupby("cell", sort=False):
            onsets_by_cell[cell_name].append(
                sub_df["onset_idx"].to_numpy(dtype=np.int64, copy=False)
            )

    return onsets_by_cell


def compute_events_per_second_df(
    *,
    all_cells: list[str],
    onsets_by_cell: dict[str, list[np.ndarray]],
    n_rows: int,
    window_len: int,
) -> pd.DataFrame:
    if window_len < 1:
        raise ValueError("window_len must be >= 1")
    if n_rows < window_len:
        raise ValueError(
            f"n_rows ({n_rows}) must be >= window_len ({window_len})"
        )

    window_kernel = np.ones(window_len, dtype=np.int32)
    window_starts = np.arange(n_rows - window_len + 1, dtype=np.int64)
    event_rate_dict: dict[str, np.ndarray] = {}

    for cell_name in all_cells:
        if cell_name in onsets_by_cell:
            onset_idx = np.concatenate(onsets_by_cell[cell_name]).astype(
                np.int64,
                copy=False,
            )
            onset_counts = np.bincount(onset_idx, minlength=n_rows)
            events_per_second = np.convolve(
                onset_counts,
                window_kernel,
                mode="valid",
            ).astype(np.int32, copy=False)
        else:
            events_per_second = np.zeros(n_rows - window_len + 1, dtype=np.int32)

        event_rate_dict[cell_name] = events_per_second

    events_per_second_df = pd.DataFrame(event_rate_dict, index=window_starts)
    events_per_second_df.index.name = "window_start_idx"
    return events_per_second_df


def main() -> None:
    args = parse_args()

    peak_stats_csv = Path(args.peak_stats_csv)
    if not peak_stats_csv.is_file():
        raise FileNotFoundError(f"Peak-stats CSV not found: {peak_stats_csv}")

    gcamp_csv = infer_gcamp_csv(peak_stats_csv, args.gcamp_csv)
    if args.n_rows is not None:
        n_rows = int(args.n_rows)
    else:
        if not gcamp_csv.is_file():
            raise FileNotFoundError(
                "Could not infer n_rows because GCAMP_with_velocity.csv was not found. "
                "Pass --gcamp-csv or --n-rows."
            )
        n_rows = count_csv_rows(gcamp_csv)

    if args.labels_mat:
        labels_path = Path(args.labels_mat)
        if not labels_path.is_file():
            raise FileNotFoundError(f"Labels .mat not found: {labels_path}")
        all_cells = sorted(
            load_good_cells_from_labels(labels_path, args.label_source),
            key=natural_cell_key,
        )
        included_cells = set(all_cells)
        label_source = args.label_source
    else:
        all_cells = list_cells_in_peak_stats(peak_stats_csv, chunksize=args.chunksize)
        included_cells = set(all_cells)
        label_source = None

    print(f"Peak stats: {peak_stats_csv}", flush=True)
    print(f"n_rows: {n_rows}", flush=True)
    print(f"window_len: {args.window_len}", flush=True)
    print(f"cells to include: {len(all_cells)}", flush=True)

    onsets_by_cell = collect_onsets_by_cell(
        peak_stats_csv=peak_stats_csv,
        n_rows=n_rows,
        chunksize=args.chunksize,
        included_cells=included_cells,
    )

    zero_event_cells = [cell for cell in all_cells if cell not in onsets_by_cell]
    if zero_event_cells:
        print(
            f"Including {len(zero_event_cells)} cells with no peak rows as all-zero columns.",
            flush=True,
        )

    events_per_second_df = compute_events_per_second_df(
        all_cells=all_cells,
        onsets_by_cell=onsets_by_cell,
        n_rows=n_rows,
        window_len=args.window_len,
    )

    output_csv = infer_output_csv(
        peak_stats_csv=peak_stats_csv,
        output_csv=args.output_csv,
        window_len=args.window_len,
        label_source=label_source,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    events_per_second_df.to_csv(output_csv, index=True)

    print(f"Output: {output_csv}", flush=True)
    print(f"Shape:  {events_per_second_df.shape}", flush=True)


if __name__ == "__main__":
    main()
