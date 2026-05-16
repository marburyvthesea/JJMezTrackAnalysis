from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ArrayLike1D = Union[pd.Series, np.ndarray, List[float], List[int]]


@dataclass
class PeakStats:
    peak_idx: int
    onset_idx: int
    offset_idx: int
    amp_max: float
    length_samples: int
    clip_start: int
    clip_end: int
    clipped_region: np.ndarray


def _as_1d_float_array(x: ArrayLike1D, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr.astype(float, copy=False)


def _as_1d_int_array(x: ArrayLike1D, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return (arr.astype(int, copy=False) != 0).astype(int)


def suprathreshold_to_events(x: ArrayLike1D) -> pd.Series:
    """Convert contiguous 1-runs into rising-edge events."""
    arr = _as_1d_int_array(x, "x")
    out = np.zeros_like(arr, dtype=int)
    if arr.size == 0:
        return pd.Series(out)

    out[0] = 1 if arr[0] == 1 else 0
    out[1:] = ((arr[1:] == 1) & (arr[:-1] == 0)).astype(int)
    return pd.Series(out)


def detect_peak_indices_from_trace(
    trace: ArrayLike1D,
    *,
    threshold: float,
    smooth_window_samples: Optional[int] = None,
    min_peak_prominence: Optional[float] = None,
    min_peak_distance_samples: int = 1,
    min_peak_width_samples: Optional[float] = None,
    return_to_baseline_tol: Optional[float] = None,
    return_to_baseline_window_samples: int = 1,
) -> np.ndarray:
    """
    Detect one peak index per calcium event from a continuous trace.

    This is stricter than simple threshold binarization and is useful when a
    single large event has small post-peak wiggles that should not count as
    separate events.
    """
    from scipy.signal import find_peaks

    tr = _as_1d_float_array(trace, "trace")
    work = tr.copy()

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


def peaks_binary_from_indices(length: int, peak_indices: Sequence[int]) -> pd.Series:
    """Create a sparse 0/1 peak vector with 1 only at accepted peak indices."""
    out = np.zeros(int(length), dtype=int)
    peak_indices = np.asarray(peak_indices, dtype=int)
    valid = peak_indices[(peak_indices >= 0) & (peak_indices < len(out))]
    out[valid] = 1
    return pd.Series(out)


def detect_peaks_binary_from_trace(
    trace: ArrayLike1D,
    *,
    threshold: float,
    smooth_window_samples: Optional[int] = None,
    min_peak_prominence: Optional[float] = None,
    min_peak_distance_samples: int = 1,
    min_peak_width_samples: Optional[float] = None,
    return_to_baseline_tol: Optional[float] = None,
    return_to_baseline_window_samples: int = 1,
) -> pd.Series:
    """Convenience wrapper returning a sparse 0/1 peak series."""
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
    return peaks_binary_from_indices(len(_as_1d_float_array(trace, "trace")), peak_indices)


def detect_peaks_df(
    traces_df: pd.DataFrame,
    *,
    threshold: float,
    cell_prefix: str = "cell_",
    smooth_window_samples: Optional[int] = None,
    min_peak_prominence: Optional[float] = None,
    min_peak_distance_samples: int = 1,
    min_peak_width_samples: Optional[float] = None,
    return_to_baseline_tol: Optional[float] = None,
    return_to_baseline_window_samples: int = 1,
) -> pd.DataFrame:
    """
    Detect sparse per-frame peak markers for each cell column in a dataframe.
    """
    cell_cols = _cell_columns(traces_df, cell_prefix=cell_prefix)
    out = {}
    for cell_name in cell_cols:
        out[cell_name] = detect_peaks_binary_from_trace(
            traces_df[cell_name],
            threshold=threshold,
            smooth_window_samples=smooth_window_samples,
            min_peak_prominence=min_peak_prominence,
            min_peak_distance_samples=min_peak_distance_samples,
            min_peak_width_samples=min_peak_width_samples,
            return_to_baseline_tol=return_to_baseline_tol,
            return_to_baseline_window_samples=return_to_baseline_window_samples,
        ).to_numpy()

    return pd.DataFrame(out, index=traces_df.index)


def get_peak_stats_for_peak(
    peak_idx: int,
    *,
    tol: float,
    window_len_onset: int,
    window_len_offset: int,
    plt_region: int,
    z_scores: ArrayLike1D,
    cell_trace: ArrayLike1D,
) -> PeakStats:
    """Compute onset/offset/amplitude stats for one detected peak."""
    z = _as_1d_float_array(z_scores, "z_scores")
    tr = _as_1d_float_array(cell_trace, "cell_trace")

    n = len(z)
    if len(tr) != n:
        raise ValueError(f"z_scores and cell_trace must have same length, got {n} and {len(tr)}")
    if not (0 <= peak_idx < n):
        raise ValueError(f"peak_idx out of range: {peak_idx} for length {n}")
    if window_len_onset < 1 or window_len_offset < 1:
        raise ValueError("window_len_onset and window_len_offset must be >= 1")
    if plt_region < 0:
        raise ValueError("plt_region must be >= 0")

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

    amp_max = float(np.max(tr[onset_idx:offset_idx + 1]))
    length_samples = int(offset_idx - onset_idx)
    clip_start = max(0, onset_idx - plt_region)
    clip_end = min(n - 1, offset_idx + plt_region)
    clipped_region = tr[clip_start:clip_end + 1].copy()

    return PeakStats(
        peak_idx=int(peak_idx),
        onset_idx=int(onset_idx),
        offset_idx=int(offset_idx),
        amp_max=amp_max,
        length_samples=length_samples,
        clip_start=int(clip_start),
        clip_end=int(clip_end),
        clipped_region=clipped_region,
    )


def find_peak_indices_from_binarized(
    peaks_binary: ArrayLike1D,
    *,
    z_scores: Optional[ArrayLike1D] = None,
) -> np.ndarray:
    """Return one representative peak index per contiguous 1-run."""
    b = _as_1d_int_array(peaks_binary, "peaks_binary")
    idx1 = np.flatnonzero(b == 1)
    if idx1.size == 0:
        return np.array([], dtype=int)

    breaks = np.where(np.diff(idx1) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, idx1.size - 1]

    if z_scores is not None:
        z = _as_1d_float_array(z_scores, "z_scores")
        peak_idxs = []
        for s, e in zip(starts, ends):
            event_idx = idx1[s:e + 1]
            peak_idxs.append(int(event_idx[np.argmax(z[event_idx])]))
        return np.array(peak_idxs, dtype=int)

    peak_idxs = []
    for s, e in zip(starts, ends):
        event_idx = idx1[s:e + 1]
        peak_idxs.append(int(event_idx[len(event_idx) // 2]))
    return np.array(peak_idxs, dtype=int)


def compute_peak_stats_from_cell(
    peaks_binary: ArrayLike1D,
    *,
    z_scores: Optional[ArrayLike1D] = None,
    cell_trace: Optional[ArrayLike1D] = None,
    tol: float = 0.25,
    window_len_onset: int = 3,
    window_len_offset: int = 3,
    plt_region: int = 10,
    behavior_df: Optional[pd.DataFrame] = None,
    behavior_columns: Optional[Sequence[str]] = None,
    gcamp_source: Optional[Union[pd.DataFrame, str, Path]] = None,
    cell_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute peak stats for all events in one cell trace.

    Parameters
    ----------
    behavior_df : pd.DataFrame, optional
        Behavior-aligned dataframe containing columns like X/Y/Velocity. If
        provided, the clipped window for each available behavior column is added
        to the returned dataframe as `<column>_clipped_region`.
    behavior_columns : sequence[str], optional
        Which behavior columns to append. If omitted, uses X/Y and whichever of
        Velocity_spatial_filtered or Velocity is present.
    gcamp_source : DataFrame or path-like, optional
        Convenience source for a saved aligned_GCAMP table. If provided together
        with `cell_name`, z_scores/cell_trace and default behavior_df can be
        pulled from the saved table.
    cell_name : str, optional
        Cell column name to use when loading trace data from gcamp_source.
    """
    gcamp_df = _coerce_gcamp_dataframe(gcamp_source) if gcamp_source is not None else None

    if z_scores is None:
        if gcamp_df is None or cell_name is None:
            raise ValueError("Provide z_scores or provide both gcamp_source and cell_name")
        if cell_name not in gcamp_df.columns:
            raise KeyError(f"{cell_name!r} not found in gcamp_source columns")
        z_scores = gcamp_df[cell_name]

    if cell_trace is None:
        if gcamp_df is not None and cell_name is not None and cell_name in gcamp_df.columns:
            cell_trace = gcamp_df[cell_name]
        else:
            cell_trace = z_scores

    if behavior_df is None and gcamp_df is not None:
        behavior_df = gcamp_df

    z = _as_1d_float_array(z_scores, "z_scores")
    tr = _as_1d_float_array(cell_trace, "cell_trace")
    if behavior_df is not None and len(behavior_df) != len(z):
        raise ValueError(
            "behavior_df must have the same number of rows as the trace, "
            f"got {len(behavior_df)} and {len(z)}"
        )

    peak_idxs = find_peak_indices_from_binarized(peaks_binary, z_scores=z)
    clip_behavior_cols = _resolve_behavior_clip_columns(behavior_df, behavior_columns)

    rows: List[Dict[str, Any]] = []
    for pidx in peak_idxs:
        stats = get_peak_stats_for_peak(
            pidx,
            tol=tol,
            window_len_onset=window_len_onset,
            window_len_offset=window_len_offset,
            plt_region=plt_region,
            z_scores=z,
            cell_trace=tr,
        )
        row: Dict[str, Any] = {
            "peak_idx": stats.peak_idx,
            "onset_idx": stats.onset_idx,
            "offset_idx": stats.offset_idx,
            "amp_max": stats.amp_max,
            "length_samples": stats.length_samples,
            "clip_start": stats.clip_start,
            "clip_end": stats.clip_end,
            "clipped_region": stats.clipped_region,
        }
        for col in clip_behavior_cols:
            row[f"{col}_clipped_region"] = _slice_clipped_region(
                behavior_df[col],
                stats.clip_start,
                stats.clip_end,
            )
        rows.append(row)

    return pd.DataFrame(rows)


def load_aligned_gcamp_df(source: Union[str, Path]) -> pd.DataFrame:
    """
    Load a saved aligned_GCAMP table from disk.

    Currently supports CSV, parquet, and pickle formats. For CSVs saved with a
    default pandas index, the unnamed first column is restored as the index.
    """
    path = Path(source)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        if len(df.columns) > 0:
            first_col = str(df.columns[0])
            if first_col.startswith("Unnamed:") or first_col in {"index", "level_0"}:
                df = df.set_index(df.columns[0], drop=True)
        return df

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError(
        f"Unsupported aligned_GCAMP file extension {suffix!r} for {path}. "
        "Use .csv, .parquet, .pq, .pkl, or .pickle."
    )


def _coerce_gcamp_dataframe(source: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source
    return load_aligned_gcamp_df(source)


def _resolve_behavior_clip_columns(
    behavior_df: Optional[pd.DataFrame],
    behavior_columns: Optional[Sequence[str]],
) -> List[str]:
    if behavior_df is None:
        return []

    if behavior_columns is not None:
        return [col for col in behavior_columns if col in behavior_df.columns]

    cols: List[str] = [col for col in ("X_coor", "Y_coor") if col in behavior_df.columns]
    if "Velocity_spatial_filtered" in behavior_df.columns:
        cols.append("Velocity_spatial_filtered")
    elif "Velocity" in behavior_df.columns:
        cols.append("Velocity")
    return cols


def _slice_clipped_region(
    values: Union[pd.Series, np.ndarray, Sequence[Any]],
    clip_start: int,
    clip_end: int,
) -> np.ndarray:
    arr = np.asarray(values)
    return arr[int(clip_start):int(clip_end) + 1].copy()


def _serialize_peak_stats_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, pd.Series):
        return json.dumps(value.to_list())
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return value


def save_peak_stats_csv(
    stats_df: pd.DataFrame,
    output_csv_path: Union[str, Path],
) -> Path:
    """
    Save a peak-stats dataframe to CSV, serializing array-valued clipped-region
    columns into JSON strings.
    """
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_df = stats_df.copy()
    for col in serializable_df.columns:
        if serializable_df[col].dtype == "object":
            serializable_df[col] = serializable_df[col].map(_serialize_peak_stats_value)

    serializable_df.to_csv(output_path, index=False)
    return output_path


def _cell_columns(df: pd.DataFrame, cell_prefix: str = "cell_") -> List[str]:
    return [c for c in df.columns if c.startswith(cell_prefix)]


def compute_peak_stats_for_all_cells(
    signal_peaks: pd.DataFrame,
    aligned_cell_traces: pd.DataFrame,
    *,
    cell_prefix: str = "cell_",
    tol: float = 0.25,
    window_len_onset: int = 3,
    window_len_offset: int = 3,
    plt_region: int = 10,
    use_onsets: bool = False,
    behavior_df: Optional[pd.DataFrame] = None,
    behavior_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute peak stats for every cell column shared by both dataframes."""
    peaks_binary_df = signal_peaks.apply(suprathreshold_to_events, axis=0) if use_onsets else signal_peaks
    clip_behavior_cols = _resolve_behavior_clip_columns(behavior_df, behavior_columns)

    cell_cols = [c for c in _cell_columns(signal_peaks, cell_prefix=cell_prefix) if c in aligned_cell_traces.columns]
    rows = []
    for cell_name in cell_cols:
        stats_df = compute_peak_stats_from_cell(
            peaks_binary_df[cell_name],
            z_scores=aligned_cell_traces[cell_name],
            cell_trace=aligned_cell_traces[cell_name],
            tol=tol,
            window_len_onset=window_len_onset,
            window_len_offset=window_len_offset,
            plt_region=plt_region,
            behavior_df=behavior_df,
            behavior_columns=behavior_columns,
        )
        if stats_df.empty:
            continue

        stats_df = stats_df.copy()
        stats_df.insert(0, "cell", cell_name)
        stats_df.insert(1, "event_id", np.arange(len(stats_df), dtype=int))
        rows.append(stats_df)

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
                "clipped_region",
                *[f"{col}_clipped_region" for col in clip_behavior_cols],
            ]
        )

    return pd.concat(rows, ignore_index=True)


def compute_peak_stats_from_gcamp_df(
    gcamp_with_velocity: Union[pd.DataFrame, str, Path],
    signal_peaks: pd.DataFrame,
    cell_name: str,
    *,
    trace_source: Optional[Union[pd.DataFrame, str, Path]] = None,
    tol: float = 0.25,
    window_len_onset: int = 3,
    window_len_offset: int = 3,
    plt_region: int = 10,
    behavior_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Thin wrapper that uses one cell column directly from GCAMP_with_velocity."""
    gcamp_df = _coerce_gcamp_dataframe(gcamp_with_velocity)
    trace_df = gcamp_df if trace_source is None else _coerce_gcamp_dataframe(trace_source)
    if cell_name not in signal_peaks.columns:
        raise KeyError(f"{cell_name!r} not found in signal_peaks columns")
    if cell_name not in trace_df.columns:
        raise KeyError(f"{cell_name!r} not found in trace dataframe columns")

    return compute_peak_stats_from_cell(
        signal_peaks[cell_name],
        z_scores=trace_df[cell_name],
        cell_trace=trace_df[cell_name],
        tol=tol,
        window_len_onset=window_len_onset,
        window_len_offset=window_len_offset,
        plt_region=plt_region,
        behavior_df=gcamp_df,
        behavior_columns=behavior_columns,
    )


def compute_peak_stats_for_all_cells_from_gcamp_df(
    gcamp_with_velocity: Union[pd.DataFrame, str, Path],
    signal_peaks: pd.DataFrame,
    *,
    cell_prefix: str = "cell_",
    tol: float = 0.25,
    window_len_onset: int = 3,
    window_len_offset: int = 3,
    plt_region: int = 10,
    use_onsets: bool = False,
    behavior_columns: Optional[Sequence[str]] = None,
    output_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Wrapper around compute_peak_stats_for_all_cells using GCAMP_with_velocity
    as trace source. Accepts either an in-memory dataframe or a saved aligned_GCAMP file.
    """
    gcamp_df = _coerce_gcamp_dataframe(gcamp_with_velocity)
    stats_df = compute_peak_stats_for_all_cells(
        signal_peaks,
        gcamp_df,
        cell_prefix=cell_prefix,
        tol=tol,
        window_len_onset=window_len_onset,
        window_len_offset=window_len_offset,
        plt_region=plt_region,
        use_onsets=use_onsets,
        behavior_df=gcamp_df,
        behavior_columns=behavior_columns,
    )
    if output_csv_path is not None:
        save_peak_stats_csv(stats_df, output_csv_path)
    return stats_df


def slice_cell_trace_and_peaks(
    gcamp_with_velocity: pd.DataFrame,
    signal_peaks: pd.DataFrame,
    cell_name: str,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    *,
    extra_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Build one tidy dataframe for plotting/inspection of a cell, peaks, and
    optional behavior columns over a selected row range.
    """
    if cell_name not in gcamp_with_velocity.columns:
        raise KeyError(f"{cell_name!r} not found in gcamp_with_velocity columns")
    if cell_name not in signal_peaks.columns:
        raise KeyError(f"{cell_name!r} not found in signal_peaks columns")

    start = 0 if start is None else int(start)
    stop = len(gcamp_with_velocity) if stop is None else int(stop)

    out = pd.DataFrame(
        {
            "trace": gcamp_with_velocity[cell_name].iloc[start:stop].to_numpy(),
            "peaks": signal_peaks[cell_name].iloc[start:stop].to_numpy(),
        },
        index=np.arange(start, stop),
    )

    for col in extra_cols or ():
        if col in gcamp_with_velocity.columns:
            out[col] = gcamp_with_velocity[col].iloc[start:stop].to_numpy()

    return out


def plot_cell_trace_and_peaks(
    gcamp_with_velocity: pd.DataFrame,
    signal_peaks: pd.DataFrame,
    cell_name: str,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    *,
    extra_cols: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> Tuple[Any, np.ndarray]:
    """
    Plot a cell trace, its binary peaks, and optional behavior columns from
    GCAMP_with_velocity over the same row interval.
    """
    extra_cols = list(extra_cols or [])
    plot_df = slice_cell_trace_and_peaks(
        gcamp_with_velocity,
        signal_peaks,
        cell_name,
        start=start,
        stop=stop,
        extra_cols=extra_cols,
    )

    n_axes = 2 + len(extra_cols)
    fig, axes = plt.subplots(n_axes, 1, figsize=figsize, sharex=True)
    if n_axes == 1:
        axes = np.array([axes])

    x = plot_df.index.to_numpy()

    axes[0].plot(x, plot_df["trace"], color="C0")
    axes[0].set_ylabel(cell_name)
    axes[0].set_title(f"{cell_name}: rows {x[0]} to {x[-1]}")

    axes[1].step(x, plot_df["peaks"], where="mid", color="C1")
    axes[1].set_ylabel("peaks")
    axes[1].set_ylim(-0.1, 1.1)

    for i, col in enumerate(extra_cols, start=2):
        axes[i].plot(x, plot_df[col], color=f"C{i}")
        axes[i].set_ylabel(col)

    axes[-1].set_xlabel("row index")
    fig.tight_layout()
    return fig, axes


def append_behavior_to_peak_stats(
    stats_df: pd.DataFrame,
    gcamp_with_velocity: pd.DataFrame,
    *,
    columns: Sequence[str] = ("Velocity", "Velocity_spatial_filtered", "X_coor", "Y_coor"),
) -> pd.DataFrame:
    """Attach behavior values at onset/peak/offset rows for each event."""
    out = stats_df.copy()

    for src_idx, label in [
        ("onset_idx", "onset"),
        ("peak_idx", "peak"),
        ("offset_idx", "offset"),
    ]:
        idx = out[src_idx].to_numpy(dtype=int)
        for col in columns:
            if col not in gcamp_with_velocity.columns:
                continue
            out[f"{label}_{col}"] = gcamp_with_velocity[col].iloc[idx].to_numpy()

    return out


def _cell_index_from_name(cell_name_or_idx: Union[str, int]) -> int:
    if isinstance(cell_name_or_idx, (int, np.integer)):
        return int(cell_name_or_idx)
    text = str(cell_name_or_idx)
    if text.startswith("cell_"):
        return int(text.split("_", 1)[1])
    return int(text)


def _load_spatial_temporal_arrays(
    extract_mat_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and orient spatial/temporal weights to (n_cells, H, W) and (n_cells, T).
    """
    with h5py.File(str(extract_mat_path), "r") as f:
        out = f["output"]
        spatial = np.asarray(out["spatial_weights"][()])
        temporal = np.asarray(out["temporal_weights"][()])

    if temporal.ndim != 2:
        raise ValueError(f"temporal_weights must be 2D, got shape {temporal.shape}")
    if spatial.ndim != 3:
        raise ValueError(f"spatial_weights must be 3D, got shape {spatial.shape}")

    if temporal.shape[0] <= temporal.shape[1]:
        n_cells = temporal.shape[0]
        temporal_nc_t = temporal
    else:
        n_cells = temporal.shape[1]
        temporal_nc_t = temporal.T

    if spatial.shape[0] == n_cells:
        spatial_nc_hw = spatial
    elif spatial.shape[2] == n_cells:
        spatial_nc_hw = np.moveaxis(spatial, 2, 0)
    else:
        raise ValueError(
            "Could not align spatial_weights with temporal_weights shapes: "
            f"spatial={spatial.shape}, temporal={temporal.shape}"
        )

    return (
        np.asarray(spatial_nc_hw, dtype=float),
        np.asarray(temporal_nc_t, dtype=float),
    )


def load_single_cell_spatiotemporal(
    extract_mat_path: Union[str, Path],
    cell_name_or_idx: Union[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one cell's spatial footprint and temporal trace from a CNMF-E/CaliAli
    extract HDF5/MAT file.

    Returns
    -------
    spatial_2d : np.ndarray
        Shape (H, W)
    temporal_1d : np.ndarray
        Shape (T,)
    """
    cell_idx = _cell_index_from_name(cell_name_or_idx)
    spatial_nc_hw, temporal_nc_t = _load_spatial_temporal_arrays(extract_mat_path)
    n_cells = temporal_nc_t.shape[0]
    if cell_idx >= n_cells:
        raise IndexError(f"cell index {cell_idx} out of range for {n_cells} temporal traces")

    return spatial_nc_hw[cell_idx], temporal_nc_t[cell_idx]


def load_all_cells_spatiotemporal(
    extract_mat_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all cell spatial footprints and temporal traces from a CNMF-E/CaliAli
    extract HDF5/MAT file.

    Returns
    -------
    spatial_nc_hw : np.ndarray
        Shape (n_cells, H, W)
    temporal_nc_t : np.ndarray
        Shape (n_cells, T)
    """
    return _load_spatial_temporal_arrays(extract_mat_path)


def _resolve_peak_center_idx(
    peak_idx: int,
    *,
    stats_df: Optional[pd.DataFrame] = None,
    peak_col: str = "peak_idx",
) -> int:
    if stats_df is not None:
        return int(stats_df.loc[peak_idx, peak_col])
    return int(peak_idx)


def _normalize_movie_float(movie: np.ndarray, normalize_mode: str) -> np.ndarray:
    movie = np.asarray(movie, dtype=float)
    if normalize_mode == "clip":
        lo = np.nanmin(movie)
        hi = np.nanmax(movie)
        if hi > lo:
            return (movie - lo) / (hi - lo)
        return np.zeros_like(movie)
    if normalize_mode == "frame":
        mins = np.nanmin(movie, axis=(1, 2), keepdims=True)
        maxs = np.nanmax(movie, axis=(1, 2), keepdims=True)
        denom = maxs - mins
        return np.divide(movie - mins, denom, out=np.zeros_like(movie), where=denom > 0)
    raise ValueError("normalize_mode must be 'clip' or 'frame'")


def _float_movie_to_uint8(movie: np.ndarray) -> np.ndarray:
    return np.clip(np.round(movie * 255), 0, 255).astype(np.uint8)


def _movie_uint8_to_rgb(movie_uint8: np.ndarray) -> np.ndarray:
    if movie_uint8.ndim == 3:
        return np.repeat(movie_uint8[..., None], 3, axis=3)
    if movie_uint8.ndim == 4 and movie_uint8.shape[-1] == 3:
        return movie_uint8.copy()
    raise ValueError(
        "movie_uint8 must have shape (T, H, W) or (T, H, W, 3), "
        f"got {movie_uint8.shape}"
    )


def build_single_cell_peak_movie(
    extract_mat_path: Union[str, Path],
    cell_name_or_idx: Union[str, int],
    peak_idx: int,
    *,
    window: Tuple[int, int] = (100, 500),
    stats_df: Optional[pd.DataFrame] = None,
    peak_col: str = "peak_idx",
    normalize_mode: str = "clip",
    include_all_cells: bool = False,
    highlight_cell: bool = True,
    highlight_strength: float = 0.85,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build a movie clip for one cell around a selected peak.

    Parameters
    ----------
    extract_mat_path : path-like
        Extract .mat/.h5 file containing output/spatial_weights and temporal_weights.
    cell_name_or_idx : str or int
        Cell name like 'cell_0' or raw cell index.
    peak_idx : int
        Either a direct sample index or, if stats_df is provided, the row label in
        stats_df whose `peak_col` gives the center sample.
    window : tuple[int, int]
        (samples_before, samples_after)
    stats_df : pd.DataFrame or None
        Optional stats dataframe. If given, `peak_idx` is treated as a row label and
        the actual center sample is read from `stats_df.loc[peak_idx, peak_col]`.
    normalize_mode : {'clip', 'frame'}
        'clip' scales the whole clip by one min/max; 'frame' scales each frame separately.
    include_all_cells : bool
        If True, reconstruct activity from all cells over the clip. Otherwise, show
        only the selected cell.
    highlight_cell : bool
        If True and include_all_cells is True, tint the selected cell green.
    highlight_strength : float
        Blend strength for the selected-cell green highlight in [0, 1].

    Returns
    -------
    movie_uint8 : np.ndarray
        Shape (T, H, W) for grayscale or (T, H, W, 3) for RGB, uint8
    meta : dict
        start/end/center indices describing the clip.
    """
    center_idx = _resolve_peak_center_idx(peak_idx, stats_df=stats_df, peak_col=peak_col)

    before, after = int(window[0]), int(window[1])
    if before < 0 or after < 0:
        raise ValueError("window values must be >= 0")

    cell_idx = _cell_index_from_name(cell_name_or_idx)
    spatial_2d, temporal_1d = load_single_cell_spatiotemporal(extract_mat_path, cell_idx)

    start = max(0, center_idx - before)
    end = min(len(temporal_1d), center_idx + after)
    if end <= start:
        raise ValueError(f"Invalid clip bounds: start={start}, end={end}")

    if include_all_cells:
        spatial_nc_hw, temporal_nc_t = load_all_cells_spatiotemporal(extract_mat_path)
        if cell_idx >= spatial_nc_hw.shape[0]:
            raise IndexError(
                f"cell index {cell_idx} out of range for {spatial_nc_hw.shape[0]} spatial footprints"
            )
        temporal_slice = temporal_nc_t[:, start:end]
        movie = np.tensordot(temporal_slice.T, spatial_nc_hw, axes=(1, 0))
        movie = np.asarray(movie, dtype=float)
        movie_norm = _normalize_movie_float(movie, normalize_mode)
        movie_rgb = np.repeat(movie_norm[..., None], 3, axis=3)

        if highlight_cell:
            cell_movie = temporal_nc_t[cell_idx, start:end][:, None, None] * spatial_nc_hw[cell_idx][None, :, :]
            cell_positive = np.maximum(np.asarray(cell_movie, dtype=float), 0.0)
            if np.nanmax(cell_positive) > 0:
                highlight = cell_positive / np.nanmax(cell_positive)
            else:
                highlight = np.zeros_like(cell_positive)
            highlight = np.clip(highlight_strength * highlight, 0.0, 1.0)
            movie_rgb[..., 0] *= (1.0 - 0.8 * highlight)
            movie_rgb[..., 2] *= (1.0 - 0.8 * highlight)
            movie_rgb[..., 1] = np.maximum(movie_rgb[..., 1], highlight)

        movie_uint8 = _float_movie_to_uint8(movie_rgb)
    else:
        temporal_slice = temporal_1d[start:end]
        movie = temporal_slice[:, None, None] * spatial_2d[None, :, :]
        movie_uint8 = _float_movie_to_uint8(_normalize_movie_float(movie, normalize_mode))

    meta = {"start": start, "end": end, "center_idx": center_idx, "cell_idx": cell_idx}
    return movie_uint8, meta


def render_static_trace_panel(
    trace: ArrayLike1D,
    *,
    width: int,
    height: int = 96,
    clip_bounds: Optional[Tuple[int, int]] = None,
    center_idx: Optional[int] = None,
    panel_label: Optional[str] = None,
) -> np.ndarray:
    """
    Render a static trace panel that can be appended beneath movie frames.

    The full trace is shown once, with the current clip bounds lightly shaded and
    the selected peak center marked by a vertical line.
    """
    import cv2

    arr = _as_1d_float_array(trace, "trace")
    if arr.size == 0:
        raise ValueError("trace must contain at least one sample")
    if width < 16 or height < 16:
        raise ValueError("trace panel width and height must both be at least 16 pixels")

    panel = np.full((height, width, 3), 255, dtype=np.uint8)

    margin_left = 8
    margin_right = 8
    margin_top = 12 if panel_label else 8
    margin_bottom = 8
    inner_w = max(2, width - margin_left - margin_right)
    inner_h = max(2, height - margin_top - margin_bottom)

    finite = np.isfinite(arr)
    if np.any(finite):
        arr_finite = arr[finite]
        lo = float(np.min(arr_finite))
        hi = float(np.max(arr_finite))
    else:
        lo = 0.0
        hi = 1.0

    if np.isclose(hi, lo):
        lo -= 0.5
        hi += 0.5

    def _sample_to_x(sample_idx: int) -> int:
        if arr.size == 1:
            return margin_left
        clipped = int(np.clip(sample_idx, 0, arr.size - 1))
        frac = clipped / float(arr.size - 1)
        return margin_left + int(round(frac * (inner_w - 1)))

    if clip_bounds is not None:
        clip_start, clip_end = clip_bounds
        x0 = _sample_to_x(clip_start)
        x1 = _sample_to_x(max(clip_start, clip_end - 1))
        x0, x1 = sorted((x0, x1))
        cv2.rectangle(
            panel,
            (x0, margin_top),
            (x1, margin_top + inner_h - 1),
            (244, 236, 214),
            thickness=-1,
        )

    sample_positions = np.linspace(0, arr.size - 1, inner_w)
    sampled = np.interp(sample_positions, np.arange(arr.size), arr)
    y_frac = (sampled - lo) / (hi - lo)
    y_coords = margin_top + np.round((1.0 - y_frac) * (inner_h - 1)).astype(np.int32)
    x_coords = margin_left + np.arange(inner_w, dtype=np.int32)
    points = np.column_stack([x_coords, y_coords]).reshape(-1, 1, 2)

    cv2.rectangle(
        panel,
        (margin_left, margin_top),
        (margin_left + inner_w - 1, margin_top + inner_h - 1),
        (208, 208, 208),
        thickness=1,
    )

    if center_idx is not None:
        x_center = _sample_to_x(center_idx)
        cv2.line(
            panel,
            (x_center, margin_top),
            (x_center, margin_top + inner_h - 1),
            (70, 135, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    if clip_bounds is not None:
        clip_start, clip_end = clip_bounds
        for boundary_idx in (clip_start, max(clip_start, clip_end - 1)):
            x_boundary = _sample_to_x(boundary_idx)
            cv2.line(
                panel,
                (x_boundary, margin_top),
                (x_boundary, margin_top + inner_h - 1),
                (184, 140, 64),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    cv2.polylines(
        panel,
        [points],
        isClosed=False,
        color=(52, 52, 52),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    if panel_label:
        cv2.putText(
            panel,
            panel_label,
            (margin_left, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

    return panel


def append_static_trace_panel(
    movie_uint8: np.ndarray,
    trace: ArrayLike1D,
    *,
    clip_bounds: Optional[Tuple[int, int]] = None,
    center_idx: Optional[int] = None,
    panel_height: int = 96,
    panel_label: Optional[str] = None,
) -> np.ndarray:
    """
    Append one static trace image beneath every movie frame.
    """
    movie_rgb = _movie_uint8_to_rgb(movie_uint8)
    trace_panel = render_static_trace_panel(
        trace,
        width=movie_rgb.shape[2],
        height=panel_height,
        clip_bounds=clip_bounds,
        center_idx=center_idx,
        panel_label=panel_label,
    )
    panel_stack = np.repeat(trace_panel[None, ...], movie_rgb.shape[0], axis=0)
    return np.concatenate([movie_rgb, panel_stack], axis=1)


def overlay_frame_numbers(
    movie_uint8: np.ndarray,
    *,
    start_frame_idx: int,
    font_scale: float = 0.5,
    thickness: int = 1,
    margin_px: int = 8,
) -> np.ndarray:
    """
    Draw aligned-GCAMP frame numbers in white in the upper-right corner.
    """
    import cv2

    movie_bgr = _movie_uint8_to_rgb(movie_uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    _, h, w, _ = movie_bgr.shape
    for frame_offset in range(movie_bgr.shape[0]):
        text = f"Frame {start_frame_idx + frame_offset}"
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = max(margin_px, w - tw - margin_px)
        y = max(th + margin_px, margin_px + th)
        # draw a black outline under the white text for readability
        cv2.putText(
            movie_bgr[frame_offset],
            text,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            movie_bgr[frame_offset],
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return movie_bgr


def _save_movie_uint8_with_cv2(
    movie_uint8: np.ndarray,
    output_path: Union[str, Path],
    *,
    fps: float = 20.0,
    resize_factor: float = 1.0,
    codec: str,
) -> Path:
    """
    Save a grayscale or RGB movie array using OpenCV.
    """
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if movie_uint8.ndim == 3:
        t, h, w = movie_uint8.shape
        is_rgb = False
    elif movie_uint8.ndim == 4 and movie_uint8.shape[-1] == 3:
        t, h, w, _ = movie_uint8.shape
        is_rgb = True
    else:
        raise ValueError(
            "movie_uint8 must have shape (T, H, W) or (T, H, W, 3), "
            f"got {movie_uint8.shape}"
        )

    out_w = max(1, int(round(w * resize_factor)))
    out_h = max(1, int(round(h * resize_factor)))
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (out_w, out_h), True)
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV could not open a video writer for {output_path} using codec {codec!r}"
        )

    for i in range(t):
        frame = movie_uint8[i]
        if resize_factor != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        if is_rgb:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(frame_bgr)

    writer.release()
    return output_path


def save_movie_uint8_as_avi(
    movie_uint8: np.ndarray,
    output_path: Union[str, Path],
    *,
    fps: float = 20.0,
    resize_factor: float = 1.0,
    codec: str = "MJPG",
) -> Path:
    """
    Save a grayscale movie array of shape (T, H, W) to an AVI file.
    """
    return _save_movie_uint8_with_cv2(
        movie_uint8,
        output_path,
        fps=fps,
        resize_factor=resize_factor,
        codec=codec,
    )


def save_movie_uint8_as_mp4(
    movie_uint8: np.ndarray,
    output_path: Union[str, Path],
    *,
    fps: float = 20.0,
    resize_factor: float = 1.0,
    codec: str = "mp4v",
) -> Path:
    """
    Save a grayscale movie array of shape (T, H, W) to an MP4 file that
    browsers can usually render inline inside notebooks.
    """
    return _save_movie_uint8_with_cv2(
        movie_uint8,
        output_path,
        fps=fps,
        resize_factor=resize_factor,
        codec=codec,
    )


def create_inline_peak_clip(
    extract_mat_path: Union[str, Path],
    cell_name_or_idx: Union[str, int],
    peak_idx: int,
    *,
    window: Tuple[int, int] = (100, 500),
    stats_df: Optional[pd.DataFrame] = None,
    peak_col: str = "peak_idx",
    fps: float = 20.0,
    resize_factor: float = 1.0,
    output_path: Optional[Union[str, Path]] = None,
    normalize_mode: str = "clip",
    include_all_cells: bool = False,
    highlight_cell: bool = True,
    highlight_strength: float = 0.85,
    show_trace_panel: bool = False,
    trace_panel_height: int = 96,
    show_frame_numbers: bool = True,
    frame_number_font_scale: float = 0.5,
    frame_number_thickness: int = 1,
    frame_number_margin_px: int = 8,
    output_format: str = "mp4",
    embed: bool = True,
    html_attributes: str = "controls",
):
    """
    Create a short movie clip around one detected peak and return an inline
    notebook display object when IPython is available.

    Parameters
    ----------
    include_all_cells : bool
        If True, render reconstructed activity from all cells instead of only the
        selected cell.
    highlight_cell : bool
        If True together with include_all_cells, tint the selected cell green.
    show_trace_panel : bool
        If True, append a static panel of the selected cell's full trace beneath
        each movie frame, with the current clip bounds and peak center marked.
    show_frame_numbers : bool
        If True, overlay the aligned-GCAMP/miniscope frame number in the upper
        right corner of each frame.
    """
    from IPython.display import Video

    movie_uint8, meta = build_single_cell_peak_movie(
        extract_mat_path,
        cell_name_or_idx,
        peak_idx,
        window=window,
        stats_df=stats_df,
        peak_col=peak_col,
        normalize_mode=normalize_mode,
        include_all_cells=include_all_cells,
        highlight_cell=highlight_cell,
        highlight_strength=highlight_strength,
    )

    if show_trace_panel:
        _, temporal_1d = load_single_cell_spatiotemporal(extract_mat_path, meta["cell_idx"])
        movie_uint8 = append_static_trace_panel(
            movie_uint8,
            temporal_1d,
            clip_bounds=(meta["start"], meta["end"]),
            center_idx=meta["center_idx"],
            panel_height=trace_panel_height,
            panel_label=f"{cell_name_or_idx} trace",
        )

    if show_frame_numbers:
        movie_uint8 = overlay_frame_numbers(
            movie_uint8,
            start_frame_idx=meta["start"],
            font_scale=frame_number_font_scale,
            thickness=frame_number_thickness,
            margin_px=frame_number_margin_px,
        )

    if output_path is None:
        temp_dir = Path(tempfile.gettempdir()) / "gcamp_peak_clips"
        cell_text = str(cell_name_or_idx).replace("/", "_")
        suffix = ".mp4" if output_format.lower() == "mp4" else ".avi"
        output_path = temp_dir / f"{cell_text}_peak_{meta['center_idx']}{suffix}"

    output_format = output_format.lower()
    if output_format == "mp4":
        saved_path = save_movie_uint8_as_mp4(
            movie_uint8,
            output_path,
            fps=fps,
            resize_factor=resize_factor,
        )
        video_obj = Video(
            filename=str(saved_path),
            embed=embed,
            mimetype="video/mp4",
            html_attributes=html_attributes,
        )
    elif output_format == "avi":
        saved_path = save_movie_uint8_as_avi(
            movie_uint8,
            output_path,
            fps=fps,
            resize_factor=resize_factor,
        )
        video_obj = Video(
            filename=str(saved_path),
            embed=embed,
            mimetype="video/x-msvideo",
            html_attributes=html_attributes,
        )
    else:
        raise ValueError("output_format must be 'mp4' or 'avi'")

    return video_obj, saved_path, meta
