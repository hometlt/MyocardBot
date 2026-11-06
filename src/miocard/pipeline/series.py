"""Shared one-dimensional helpers for the signal stages (11 to 13).

A path is a row index per column; a signal is a deflection per sample. The
conversion is not quite a subtraction, because a single-valued path cannot
represent the vertical stroke of a QRS complex: in those columns the pen covered
a whole run of rows, and the amplitude lives at the far end of that run.
"""

from __future__ import annotations

import cv2
import numpy as np

from miocard.pipeline.state import LeadPath


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, _odd(int(window)))
    if window <= 1 or values.size < 3:
        return values.astype(np.float32)
    return cv2.blur(
        values.reshape(1, -1).astype(np.float32),
        (window, 1),
        borderType=cv2.BORDER_REPLICATE,
    ).ravel()


def median_filter(values: np.ndarray, window: int) -> np.ndarray:
    """Running median via a strided view; used for baseline estimates."""
    window = max(1, _odd(int(window)))
    if window <= 1 or values.size < window:
        return values.astype(np.float32)
    half = window // 2
    padded = np.pad(values.astype(np.float32), half, mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(strided, axis=-1).astype(np.float32)


def effective_trace(path: LeadPath, smooth_px: int) -> np.ndarray:
    """Row of the pen per column, taking the far end of tall vertical strokes.

    Where the ink run in a column is only a stroke wide the path itself is the
    answer; where it spans many rows the pen swept through them all in one
    sample, and the extreme farther from the local baseline is the amplitude the
    recorder drew.
    """
    y = path.y.astype(np.float32)
    reference = moving_average(y, max(3, smooth_px))
    thickness = path.bottom_envelope - path.top_envelope
    tall = thickness > max(3.0, smooth_px * 0.35)
    above = reference - path.top_envelope
    below = path.bottom_envelope - reference
    extreme = np.where(above >= below, path.top_envelope, path.bottom_envelope)
    return np.where(tall, extreme, y).astype(np.float32)


def to_deflection(trace_rows: np.ndarray, baseline_rows: np.ndarray) -> np.ndarray:
    """Image rows grow downwards; a deflection above the baseline is positive."""
    return (baseline_rows - trace_rows).astype(np.float32)


def robust_scale(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    spread = float(np.percentile(np.abs(values - np.median(values)), 75))
    return max(spread, 1e-3)
