"""Shared ink extraction used by stages 4 to 7.

Everything the pen and the printer put on the paper is darker than the paper
around it, so the honest way to find ink is to estimate the paper and subtract
it. A morphological closing wider than the widest dark stroke reconstructs the
paper underneath the print without needing a global threshold, which is what
makes this survive the uneven lighting of a phone photo.

This layer deliberately does *not* decide what the ink is. Separating trace from
grid, text, noise and artifact happens in stages 6 to 8, on top of this.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_BACKGROUND_KERNEL = 15
MAX_BACKGROUND_KERNEL = 51


@dataclass(frozen=True)
class InkLayers:
    """Ink strength and the intermediate layers the class detectors build on."""

    ink: np.ndarray
    ridge: np.ndarray
    strong: np.ndarray
    background: np.ndarray
    scale: float
    threshold: float
    kernel_px: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.ink.shape[0], self.ink.shape[1]


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def background_kernel_px(grid_period_px: float | None, image_width: int) -> int:
    """Closing width that spans the widest stroke without swallowing the paper."""
    if grid_period_px and grid_period_px > 1:
        estimate = int(round(3.5 * grid_period_px))
    else:
        estimate = int(round(image_width / 60))
    return _odd(int(np.clip(estimate, MIN_BACKGROUND_KERNEL, MAX_BACKGROUND_KERNEL)))


def estimate_background(gray: np.ndarray, kernel_px: int) -> np.ndarray:
    """Reconstruct the paper luminance under the print."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_px, kernel_px))
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)


def extract_ink(image: np.ndarray, grid_period_px: float | None = None) -> InkLayers:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_px = background_kernel_px(grid_period_px, image.shape[1])
    background = estimate_background(gray, kernel_px)
    raw = cv2.subtract(background, gray).astype(np.float32)

    # A high percentile rather than the maximum, so a single specular speck does
    # not push every real stroke towards zero.
    scale = float(np.percentile(raw, 99.5))
    if scale < 4.0:
        scale = max(float(raw.max()), 4.0)
    ink = np.clip(raw / scale, 0.0, 1.0).astype(np.float32)

    # The grid is a fine mesh; a white top-hat with a kernel a little wider than
    # the mesh keeps thin strokes and removes broad shading that survived the
    # background subtraction.
    ridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ridge = cv2.morphologyEx(ink, cv2.MORPH_TOPHAT, ridge_kernel)
    ridge = np.clip(ridge / max(float(np.percentile(ridge, 99.0)), 1e-3), 0.0, 1.0)

    threshold = _ink_threshold(ink)
    strong = (ink >= threshold).astype(np.float32)
    return InkLayers(
        ink=ink,
        ridge=ridge.astype(np.float32),
        strong=strong,
        background=background,
        scale=scale,
        threshold=threshold,
        kernel_px=kernel_px,
    )


def _ink_threshold(ink: np.ndarray) -> float:
    """Split pen ink from grid ink, with a floor so faint grids never qualify."""
    as_bytes = np.clip(ink * 255.0, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(as_bytes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.clip(otsu / 255.0, 0.18, 0.75))


def row_ink_profile(layers: InkLayers, smooth_px: int = 7) -> np.ndarray:
    """Fraction of strong ink per image row, lightly smoothed."""
    profile = layers.strong.mean(axis=1).astype(np.float32)
    if smooth_px >= 3:
        profile = cv2.GaussianBlur(
            profile.reshape(-1, 1), (1, _odd(smooth_px)), 0
        ).ravel()
    return profile


def column_ink_profile(layers: InkLayers, smooth_px: int = 7) -> np.ndarray:
    profile = layers.strong.mean(axis=0).astype(np.float32)
    if smooth_px >= 3:
        profile = cv2.GaussianBlur(
            profile.reshape(1, -1), (_odd(smooth_px), 1), 0
        ).ravel()
    return profile
