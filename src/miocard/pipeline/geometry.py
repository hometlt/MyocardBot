"""Stage 2 - Geometry analyzer.

An ECG sheet is a ruled document whose rows are horizontal, so the rotation that
makes the row-ink projection profile as peaky as possible is the rotation that
straightens the sheet. That criterion beats line fitting on a phone photo, where
Hough segments latch onto the waveform as readily as onto the grid.

The stage also trims the empty margin around the print, and reports how sharp
the angle peak was so later stages know whether the horizontal axis is
trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from miocard.pipeline.loader import ImageProcessingError
from miocard.pipeline.state import PipelineStageResult, clamp01

COARSE_RANGE_DEG = 6.0
COARSE_STEP_DEG = 0.50
FINE_STEP_DEG = 0.05
MIN_ROTATION_DEG = 0.20
# The row structure of an ECG sheet survives heavy downscaling, and the angle
# sweep rotates this image once per candidate, so it is kept small.
DETECTION_EDGE_PX = 640


@dataclass(frozen=True)
class GeometryReport:
    angle_deg: float
    angle_sharpness: float
    rotation_applied: bool
    crop_applied: bool
    retained_fraction: float
    brightness: float
    contrast: float
    blur_score: float
    width: int
    height: int
    # Where the kept rectangle sat in the decoded image, and how big that image
    # was. Together with the angle these are enough to put a point of the
    # photograph the user sent onto the picture the pipeline reasoned about -
    # which is what an annotation drawn back onto the original needs, and what a
    # hand-marked reference point needs to be scored against a detected band.
    crop_left: int = 0
    crop_top: int = 0
    source_width: int = 0
    source_height: int = 0

    @property
    def probability(self) -> float:
        # A sharp, well-supported peak means the row axis is known; a flat score
        # curve means we could not tell and simply left the sheet alone.
        base = 0.40 + 0.55 * clamp01(self.angle_sharpness)
        if self.crop_applied and self.retained_fraction < 0.6:
            base -= 0.05
        return clamp01(base)

    def source_matrix(self, scale: float = 1.0) -> np.ndarray:
        """Affine mapping a decoded-source point to the analysed picture.

        The stage rotates about the centre of the decoded image, crops, and the
        caller may then rescale; composing those three gives the 2x3 matrix.
        """
        if self.rotation_applied and self.source_width and self.source_height:
            matrix = cv2.getRotationMatrix2D(
                (self.source_width / 2, self.source_height / 2), self.angle_deg, 1.0
            )
        else:
            matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        matrix = matrix.copy()
        matrix[0, 2] -= self.crop_left
        matrix[1, 2] -= self.crop_top
        return matrix * float(scale)


def map_source_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine from :meth:`GeometryReport.source_matrix` to (x, y) pairs."""
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return xy @ matrix[:, :2].T + matrix[:, 2]


def _ink_profile_image(image: np.ndarray) -> np.ndarray:
    """Background-corrected ink, downscaled, as float32 in [0, 1]."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, DETECTION_EDGE_PX / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    ink = cv2.subtract(background, gray).astype(np.float32)
    peak = float(np.percentile(ink, 99.5))
    if peak <= 1e-3:
        return np.zeros_like(ink)
    return np.clip(ink / peak, 0.0, 1.0)


def _row_profile_score(ink: np.ndarray, angle_deg: float) -> float:
    """How peaky the horizontal projection is at this rotation.

    Lead rows stack their ink onto a handful of image rows once they are level,
    so the squared row-to-row difference of the projection peaks at the correct
    angle.
    """
    if abs(angle_deg) > 1e-6:
        height, width = ink.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1.0)
        rotated = cv2.warpAffine(
            ink,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
    else:
        rotated = ink
    profile = rotated.mean(axis=1)
    if profile.size < 3:
        return 0.0
    return float(np.square(np.diff(profile)).sum())


def estimate_skew(image: np.ndarray) -> tuple[float, float]:
    """Return the sheet rotation in degrees and how sharp that estimate is."""
    ink = _ink_profile_image(image)
    if float(ink.max()) <= 0.0:
        return 0.0, 0.0

    coarse = np.arange(-COARSE_RANGE_DEG, COARSE_RANGE_DEG + 1e-6, COARSE_STEP_DEG)
    scores = np.array([_row_profile_score(ink, float(angle)) for angle in coarse])
    if not np.isfinite(scores).any() or float(scores.max()) <= 0.0:
        return 0.0, 0.0

    best_index = int(np.argmax(scores))
    best_angle = float(coarse[best_index])
    fine = np.arange(
        best_angle - COARSE_STEP_DEG,
        best_angle + COARSE_STEP_DEG + 1e-6,
        FINE_STEP_DEG,
    )
    fine_scores = np.array([_row_profile_score(ink, float(angle)) for angle in fine])
    best_angle = float(fine[int(np.argmax(fine_scores))])

    # Sharpness compares the peak against the typical score, so a sheet that
    # looks the same at every angle scores near zero however large the maximum.
    peak = float(scores.max())
    median = float(np.median(scores))
    sharpness = clamp01((peak - median) / max(peak, 1e-6) * 2.5)
    return best_angle, sharpness


def rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_empty_margins(image: np.ndarray) -> tuple[np.ndarray, bool, float, tuple[int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_empty = gray < 248
    ys, xs = np.where(non_empty)
    if len(xs) == 0:
        raise ImageProcessingError("Image has no non-white content.")
    height, width = gray.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(8, int((x2 - x1) * 0.015))
    pad_y = max(8, int((y2 - y1) * 0.015))
    x1, x2 = max(0, x1 - pad_x), min(width, x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(height, y2 + pad_y)
    retained = ((x2 - x1) * (y2 - y1)) / (width * height)
    # Cropping away more than half the frame usually means the sheet itself is
    # faint rather than that the margin is empty, so the frame is kept.
    if retained < 0.45:
        return image, False, 1.0, (0, 0)
    cropped = image[y1:y2, x1:x2]
    changed = cropped.shape[:2] != image.shape[:2]
    return cropped, changed, float(retained), (x1, y1)


def analyze_geometry(image: np.ndarray) -> tuple[np.ndarray, GeometryReport]:
    """Straighten and trim the sheet, and report what was measured."""
    source_height, source_width = image.shape[:2]
    angle, sharpness = estimate_skew(image)
    rotation_applied = abs(angle) >= MIN_ROTATION_DEG and sharpness >= 0.10
    if rotation_applied:
        image = rotate(image, angle)
    else:
        angle = 0.0

    image, crop_applied, retained, (crop_left, crop_top) = crop_empty_margins(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    if contrast < 8:
        raise ImageProcessingError("Image contrast too low for safe analysis.")

    height, width = image.shape[:2]
    report = GeometryReport(
        angle_deg=float(angle),
        angle_sharpness=float(sharpness),
        rotation_applied=rotation_applied,
        crop_applied=crop_applied,
        retained_fraction=retained,
        brightness=float(gray.mean()),
        contrast=contrast,
        blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        width=width,
        height=height,
        crop_left=int(crop_left),
        crop_top=int(crop_top),
        source_width=int(source_width),
        source_height=int(source_height),
    )
    return image, report


def geometry_stage(report: GeometryReport) -> PipelineStageResult:
    if report.rotation_applied:
        summary = f"Sheet rotated by {report.angle_deg:+.2f}° to level the lead rows."
    else:
        summary = "No confident skew found; the sheet was left unrotated."
    return PipelineStageResult(
        "Geometry analyzer",
        report.probability,
        summary,
        evidence=(
            f"deskew_angle_deg={report.angle_deg:+.2f}",
            f"angle_sharpness={report.angle_sharpness:.2f}",
            f"cropped={str(report.crop_applied).lower()}",
        ),
    )
