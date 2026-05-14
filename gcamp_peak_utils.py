from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    z_scores: ArrayLike1D,
    cell_trace: Optional[ArrayLike1D] = None,
    tol: float = 0.25,
    window_len_onset: int = 3,
    window_len_offset: int = 3,
    plt_region: int = 10,
) -> pd.DataFrame:
    """Compute peak stats for all events in one cell trace."""
    z = _as_1d_float_array(z_scores, "z_scores")
    tr = _as_1d_float_array(cell_trace if cell_trace is not None else z_scores, "cell_trace")

    peak_idxs = find_peak_indices_from_binarized(peaks_binary, z_scores=z)

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
        rows.append(
            {
                "peak_idx": stats.peak_idx,
                "onset_idx": stats.onset_idx,
                "offset_idx": stats.offset_idx,
                "amp_max": stats.amp_max,
                "length_samples": stats.length_samples,
                "clip_start": stats.clip_start,
                "clip_end": stats.clip_end,
                "clipped_region": stats.clipped_region,
            }
        )

    return pd.DataFrame(rows)


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
) -> pd.DataFrame:
    """Compute peak stats for every cell column shared by both dataframes."""
    peaks_binary_df = signal_peaks.apply(suprathreshold_to_events, axis=0) if use_onsets else signal_peaks

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
            ]
        )

    return pd.concat(rows, ignore_index=True)


def compute_peak_stats_from_gcamp_df(
    gcamp_with_velocity: pd.DataFrame,
    signal_peaks: pd.DataFrame,
    cell_name: str,
    *,
    trace_source: Optional[pd.DataFrame] = None,
    tol: float = 0.25,
    window_len_onset: int = 3,
    window_len_offset: int = 3,
    plt_region: int = 10,
) -> pd.DataFrame:
    """Thin wrapper that uses one cell column directly from GCAMP_with_velocity."""
    trace_df = gcamp_with_velocity if trace_source is None else trace_source
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
    )


def compute_peak_stats_for_all_cells_from_gcamp_df(
    gcamp_with_velocity: pd.DataFrame,
    signal_peaks: pd.DataFrame,
    *,
    cell_prefix: str = "cell_",
    tol: float = 0.25,
    window_len_onset: int = 3,
    window_len_offset: int = 3,
    plt_region: int = 10,
    use_onsets: bool = False,
) -> pd.DataFrame:
    """Wrapper around compute_peak_stats_for_all_cells using GCAMP_with_velocity as trace source."""
    return compute_peak_stats_for_all_cells(
        signal_peaks,
        gcamp_with_velocity,
        cell_prefix=cell_prefix,
        tol=tol,
        window_len_onset=window_len_onset,
        window_len_offset=window_len_offset,
        plt_region=plt_region,
        use_onsets=use_onsets,
    )


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
