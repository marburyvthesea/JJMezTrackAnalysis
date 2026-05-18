#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class PeakStats:
    peak_idx: int
    onset_idx: int
    offset_idx: int
    amp_max: float
    length_samples: int
    clip_start: int
    clip_end: int
    clipped_region: Optional[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute conservative calcium peak stats from a single "
            "GCAMP_with_velocity.csv file and write a timestamped CSV."
        )
    )
    parser.add_argument(
        "--gcamp-csv",
        required=True,
        help="Path to one GCAMP_with_velocity.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the output CSV. Defaults to the input file's parent directory.",
    )
    parser.add_argument(
        "--output-prefix",
        default="all_peak_stats",
        help="Prefix for the timestamped output CSV filename.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional explicit output CSV path. Overrides --output-dir/--output-prefix.",
    )
    parser.add_argument(
        "--cell-prefix",
        default="cell_",
        help="Prefix used to identify calcium trace columns.",
    )
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument("--smooth-window-samples", type=int, default=3)
    parser.add_argument("--min-peak-prominence", type=float, default=1.5)
    parser.add_argument("--min-peak-distance-samples", type=int, default=20)
    parser.add_argument("--min-peak-width-samples", type=float, default=2)
    parser.add_argument("--return-to-baseline-tol", type=float, default=0.4)
    parser.add_argument("--return-to-baseline-window-samples", type=int, default=5)
    parser.add_argument("--tol", type=float, default=0.25)
    parser.add_argument("--window-len-onset", type=int, default=3)
    parser.add_argument("--window-len-offset", type=int, default=3)
    parser.add_argument("--plt-region", type=int, default=10)
    parser.add_argument(
        "--omit-clips",
        action="store_true",
        help=(
            "Write a lighter CSV that excludes clipped trace/behavior arrays and "
            "keeps only scalar peak summary columns."
        ),
    )
    parser.add_argument(
        "--progress-every-cells",
        type=int,
        default=10,
        help="Print a progress update after this many cells have been processed.",
    )
    parser.add_argument(
        "--progress-every-events",
        type=int,
        default=1000,
        help="Print a progress update whenever cumulative detected events cross this interval.",
    )
    return parser.parse_args()


def load_aligned_gcamp_df(source: Path) -> pd.DataFrame:
    df = pd.read_csv(source)
    if len(df.columns) > 0:
        first_col = str(df.columns[0])
        if first_col.startswith("Unnamed:") or first_col in {"index", "level_0"}:
            df = df.set_index(df.columns[0], drop=True)
    return df


def cell_columns(df: pd.DataFrame, cell_prefix: str = "cell_") -> List[str]:
    return [c for c in df.columns if c.startswith(cell_prefix)]


def as_1d_float_array(x: Sequence[Any], name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr.astype(float, copy=False)


def detect_peak_indices_from_trace(
    trace: Sequence[Any],
    *,
    threshold: float,
    smooth_window_samples: Optional[int] = None,
    min_peak_prominence: Optional[float] = None,
    min_peak_distance_samples: int = 1,
    min_peak_width_samples: Optional[float] = None,
    return_to_baseline_tol: Optional[float] = None,
    return_to_baseline_window_samples: int = 1,
) -> np.ndarray:
    trace_arr = as_1d_float_array(trace, "trace")
    work = trace_arr.copy()

    if smooth_window_samples is not None and smooth_window_samples > 1:
        work = (
            pd.Series(work)
            .rolling(window=int(smooth_window_samples), center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

    peak_kwargs: Dict[str, Any] = {
        "height": threshold,
        "distance": max(1, int(min_peak_distance_samples)),
    }
    if min_peak_prominence is not None:
        peak_kwargs["prominence"] = float(min_peak_prominence)
    if min_peak_width_samples is not None:
        peak_kwargs["width"] = float(min_peak_width_samples)

    peak_idxs, _ = find_peaks(work, **peak_kwargs)

    if return_to_baseline_tol is None or peak_idxs.size <= 1:
        return peak_idxs.astype(int)

    tol = float(return_to_baseline_tol)
    window = max(1, int(return_to_baseline_window_samples))
    accepted: List[int] = [int(peak_idxs[0])]

    def has_baseline_return(start_idx: int, stop_idx: int) -> bool:
        if stop_idx <= start_idx + 1:
            return True
        segment = np.abs(work[start_idx + 1:stop_idx])
        if segment.size < window:
            return bool(np.all(segment <= tol))
        baseline_mask = segment <= tol
        kernel = np.ones(window, dtype=int)
        return bool(np.any(np.convolve(baseline_mask.astype(int), kernel, mode="valid") == window))

    for candidate in peak_idxs[1:]:
        if has_baseline_return(accepted[-1], int(candidate)):
            accepted.append(int(candidate))

    return np.asarray(accepted, dtype=int)


def get_peak_stats_for_peak(
    peak_idx: int,
    *,
    z_scores: Sequence[Any],
    cell_trace: Sequence[Any],
    tol: float,
    window_len_onset: int,
    window_len_offset: int,
    plt_region: int,
    include_clipped_region: bool = True,
) -> PeakStats:
    z = as_1d_float_array(z_scores, "z_scores")
    tr = as_1d_float_array(cell_trace, "cell_trace")

    n = len(z)
    if len(tr) != n:
        raise ValueError(f"z_scores and cell_trace must have same length, got {n} and {len(tr)}")
    if not (0 <= peak_idx < n):
        raise ValueError(f"peak_idx out of range: {peak_idx} for length {n}")

    onset_idx = None
    for i in range(peak_idx, window_len_onset - 2, -1):
        start = i - window_len_onset + 1
        if start < 0:
            break
        if np.all(np.abs(z[start:i + 1]) < tol):
            onset_idx = start
            break
    if onset_idx is None:
        onset_idx = 0

    baseline_val = tr[onset_idx]
    offset_idx = None
    last_start = n - window_len_offset
    for i in range(peak_idx, last_start + 1):
        window = slice(i, i + window_len_offset)
        if np.all(np.abs(z[window]) < tol) and np.all(tr[window] < baseline_val):
            offset_idx = i + window_len_offset - 1
            break
    if offset_idx is None:
        offset_idx = n - 1

    clip_start = max(0, onset_idx - plt_region)
    clip_end = min(n - 1, offset_idx + plt_region)

    clipped_region = tr[clip_start:clip_end + 1].copy() if include_clipped_region else None

    return PeakStats(
        peak_idx=int(peak_idx),
        onset_idx=int(onset_idx),
        offset_idx=int(offset_idx),
        amp_max=float(np.max(tr[onset_idx:offset_idx + 1])),
        length_samples=int(offset_idx - onset_idx),
        clip_start=int(clip_start),
        clip_end=int(clip_end),
        clipped_region=clipped_region,
    )


def resolve_behavior_clip_columns(
    gcamp_df: pd.DataFrame,
    requested: Sequence[str],
) -> List[str]:
    return [col for col in requested if col in gcamp_df.columns]


def slice_clipped_region(values: Sequence[Any], clip_start: int, clip_end: int) -> np.ndarray:
    arr = np.asarray(values)
    return arr[int(clip_start):int(clip_end) + 1].copy()


def serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, pd.Series):
        return json.dumps(value.to_list())
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return value


def build_output_path(args: argparse.Namespace, gcamp_csv: Path) -> Path:
    if args.output_csv is not None:
        return Path(args.output_csv)

    output_dir = Path(args.output_dir) if args.output_dir is not None else gcamp_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{args.output_prefix}_{timestamp}.csv"


def compute_peak_stats_for_all_cells(
    gcamp_df: pd.DataFrame,
    *,
    cell_prefix: str,
    threshold: float,
    smooth_window_samples: int,
    min_peak_prominence: float,
    min_peak_distance_samples: int,
    min_peak_width_samples: float,
    return_to_baseline_tol: float,
    return_to_baseline_window_samples: int,
    tol: float,
    window_len_onset: int,
    window_len_offset: int,
    plt_region: int,
    behavior_columns: Sequence[str],
    include_clips: bool = True,
    progress_every_cells: int = 10,
    progress_every_events: int = 1000,
) -> pd.DataFrame:
    requested_behavior_cols = resolve_behavior_clip_columns(gcamp_df, behavior_columns)
    cell_cols = cell_columns(gcamp_df, cell_prefix=cell_prefix)
    rows: List[Dict[str, Any]] = []
    n_cells = len(cell_cols)
    next_event_report = max(1, int(progress_every_events))

    print(
        f"Starting peak-stats analysis for {n_cells} cells across {len(gcamp_df)} frames "
        f"| include_clips={include_clips}",
        flush=True,
    )

    for cell_idx, cell_name in enumerate(cell_cols, start=1):
        trace = gcamp_df[cell_name]
        peak_indices = detect_peak_indices_from_trace(
            trace,
            threshold=threshold,
            smooth_window_samples=smooth_window_samples,
            min_peak_prominence=min_peak_prominence,
            min_peak_distance_samples=min_peak_distance_samples,
            min_peak_width_samples=min_peak_width_samples,
            return_to_baseline_tol=return_to_baseline_tol,
            return_to_baseline_window_samples=return_to_baseline_window_samples,
        )

        for event_id, peak_idx in enumerate(peak_indices):
            stats = get_peak_stats_for_peak(
                int(peak_idx),
                z_scores=trace,
                cell_trace=trace,
                tol=tol,
                window_len_onset=window_len_onset,
                window_len_offset=window_len_offset,
                plt_region=plt_region,
                include_clipped_region=include_clips,
            )
            row: Dict[str, Any] = {
                "cell": cell_name,
                "event_id": int(event_id),
                "peak_idx": stats.peak_idx,
                "onset_idx": stats.onset_idx,
                "offset_idx": stats.offset_idx,
                "amp_max": stats.amp_max,
                "length_samples": stats.length_samples,
                "clip_start": stats.clip_start,
                "clip_end": stats.clip_end,
            }
            if include_clips:
                row["clipped_region"] = stats.clipped_region
                for col in requested_behavior_cols:
                    row[f"{col}_clipped_region"] = slice_clipped_region(
                        gcamp_df[col],
                        stats.clip_start,
                        stats.clip_end,
                    )
            rows.append(row)

        cumulative_events = len(rows)
        should_report = (
            cell_idx == 1
            or cell_idx == n_cells
            or (progress_every_cells > 0 and cell_idx % int(progress_every_cells) == 0)
            or cumulative_events >= next_event_report
        )
        if should_report:
            pct = (100.0 * cell_idx / n_cells) if n_cells > 0 else 100.0
            print(
                f"[progress] cells {cell_idx}/{n_cells} ({pct:.1f}%) "
                f"| current={cell_name} | cumulative_events={cumulative_events}",
                flush=True,
            )
            while cumulative_events >= next_event_report:
                next_event_report += max(1, int(progress_every_events))

    if not rows:
        return pd.DataFrame(
            columns=[
                "cell",
                "event_id",
                "peak_idx",
                "onset_idx",
                "offset_idx",
                "amp_max",
                "length_samples",
                "clip_start",
                "clip_end",
                *(
                    ["clipped_region", *[f"{col}_clipped_region" for col in requested_behavior_cols]]
                    if include_clips
                    else []
                ),
            ]
        )

    return pd.DataFrame(rows)


def save_peak_stats_csv(stats_df: pd.DataFrame, output_csv_path: Path) -> Path:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_df = stats_df.copy()
    for col in serializable_df.columns:
        if serializable_df[col].dtype == "object":
            serializable_df[col] = serializable_df[col].map(serialize_value)
    serializable_df.to_csv(output_csv_path, index=False)
    return output_csv_path


def main() -> int:
    args = parse_args()
    gcamp_csv = Path(args.gcamp_csv).expanduser().resolve()
    if not gcamp_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {gcamp_csv}")

    output_csv = build_output_path(args, gcamp_csv)
    gcamp_df = load_aligned_gcamp_df(gcamp_csv)
    cell_cols = cell_columns(gcamp_df, cell_prefix=args.cell_prefix)
    if not cell_cols:
        raise ValueError(f"No cell columns starting with {args.cell_prefix!r} found in {gcamp_csv}")

    stats_df = compute_peak_stats_for_all_cells(
        gcamp_df,
        cell_prefix=args.cell_prefix,
        threshold=args.threshold,
        smooth_window_samples=args.smooth_window_samples,
        min_peak_prominence=args.min_peak_prominence,
        min_peak_distance_samples=args.min_peak_distance_samples,
        min_peak_width_samples=args.min_peak_width_samples,
        return_to_baseline_tol=args.return_to_baseline_tol,
        return_to_baseline_window_samples=args.return_to_baseline_window_samples,
        tol=args.tol,
        window_len_onset=args.window_len_onset,
        window_len_offset=args.window_len_offset,
        plt_region=args.plt_region,
        behavior_columns=["X_coor", "Y_coor", "Velocity_spatial_filtered"],
        include_clips=not args.omit_clips,
        progress_every_cells=args.progress_every_cells,
        progress_every_events=args.progress_every_events,
    )
    save_peak_stats_csv(stats_df, output_csv)

    print(f"Processed: {gcamp_csv}", flush=True)
    print(f"Output:    {output_csv}", flush=True)
    print(f"Cells:     {len(cell_cols)}", flush=True)
    print(f"Events:    {len(stats_df)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
