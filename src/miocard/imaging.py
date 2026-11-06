"""Image preparation for the model, and the entry point to the MyocardBot2 pipeline.

Two jobs live here and they are deliberately kept apart. Preparing the picture
the model will look at - straighten, trim, rescale to the visual token budget -
is a presentation concern. Reconstructing the signal is the fifteen-stage
pipeline in :mod:`miocard.pipeline`, which runs on exactly the pixels the model
is shown so its probability maps and coordinates line up with what it sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from miocard.models import LocalizedFinding
from miocard.pipeline import (
    ECGPipelineState,
    GridMetrics,
    ImageProcessingError,
    PipelineStageResult,
    Preprocessing,
    measurement_text,
    run_pipeline,
    signal_text,
)
from miocard.pipeline.config import DEFAULT_CONFIG, PipelineConfig
from miocard.pipeline.geometry import analyze_geometry
from miocard.pipeline.grid import (
    GRID_PEAK_THRESHOLD,
    LARGE_CELL_SMALL_SQUARES,
    MAX_GRID_PERIOD_PX,
    MIN_GRID_PERIOD_PX,
    measure_grid,
)
from miocard.pipeline.loader import MAX_DECODED_PIXELS, load_image

__all__ = [
    "GRID_PEAK_THRESHOLD",
    "LARGE_CELL_SMALL_SQUARES",
    "MAX_DECODED_PIXELS",
    "MAX_GRID_PERIOD_PX",
    "MIN_GRID_PERIOD_PX",
    "ECGPipelineState",
    "GridMetrics",
    "ImageProcessingError",
    "ImageProfile",
    "PipelineStageResult",
    "ProcessedImage",
    "annotate_findings",
    "estimate_visual_tokens",
    "measure_grid",
    "process_ecg_image",
    "profile_for_model",
]


@dataclass(frozen=True)
class ImageProfile:
    max_edge: int
    visual_token_budget: int


@dataclass(frozen=True)
class ProcessedImage:
    data: bytes
    media_type: str
    width: int
    height: int
    estimated_visual_tokens: int
    blur_score: float
    brightness: float
    contrast: float
    pipeline: ECGPipelineState | None = None
    transformations: tuple[str, ...] = field(default_factory=tuple)
    grid: GridMetrics | None = None

    def _grid_text(self) -> str:
        if self.grid is None or self.grid.confidence == "low":
            return (
                "- Grid measurement skipped: not confident enough; "
                "falling back to visible grid interval cues."
            )
        small = self.grid.small_cell_px
        large = self.grid.large_cell_px
        # At the standard 25 mm/s one millimetre is 0.04 s, so the measured pitch turns
        # the picture into a time scale without the model having to count squares.
        seconds_per_row = self.width / (25.0 * small)
        return (
            f"- Grid pitch: small cell ~ {small:.1f} px, "
            f"large (5 mm) ~ {large:.1f} px, confidence: {self.grid.confidence}\n"
            f"- At 25 mm/s, one row corresponds to about {seconds_per_row:.1f} s, "
            f"so 300 / (large cells per 300 seconds) approximates heart rate\n"
        )

    def metadata_text(self) -> str:
        changes = ", ".join(self.transformations) if self.transformations else "none"
        lines = [
            f"- Processed image size: {self.width}x{self.height} px",
            f"- Estimated visual tokens: ~{self.estimated_visual_tokens} tokens",
            f"- Laplacian variance: {self.blur_score:.1f}",
            f"- Brightness/contrast (0-255): {self.brightness:.1f}/{self.contrast:.1f}",
            self._grid_text().rstrip("\n"),
            f"- Changes applied: {changes}",
            (
                "The metadata is intended as compact notes for the model, not a clinical report. "
                "Grid calibration is included only when it is repeatable and consistent; "
                "ambiguous measurements should be ignored."
            ),
        ]
        if self.pipeline is not None:
            measurements = measurement_text(self.pipeline)
            if measurements:
                lines.append("")
                lines.append("- Reconstruction measurements (MyocardBot2):")
                lines.append(measurements)
            waveform = signal_text(self.pipeline)
            if waveform:
                lines.append("")
                lines.append(waveform)
            lines.append("")
            lines.append(self.pipeline.metadata_text().rstrip("\n"))
        return "\n".join(lines) + "\n"


def profile_for_model(model: str) -> ImageProfile:
    if model == "claude-opus-4-6":
        return ImageProfile(max_edge=1568, visual_token_budget=1568)
    return ImageProfile(max_edge=2576, visual_token_budget=4784)


def estimate_visual_tokens(width: int, height: int) -> int:
    return math.ceil(width / 28) * math.ceil(height / 28)


def _enhance_for_ecg(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(12, 12))
    enhanced_light = clahe.apply(light)
    enhanced = cv2.merge((enhanced_light, a, b))
    bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=0.8)
    return cv2.addWeighted(bgr, 1.12, blurred, -0.12, 0)


def _resize_to_profile(image: np.ndarray, profile: ImageProfile) -> tuple[np.ndarray, bool]:
    height, width = image.shape[:2]
    edge_scale = min(1.0, profile.max_edge / max(width, height))
    token_scale = min(
        1.0,
        math.sqrt(profile.visual_token_budget / max(1, estimate_visual_tokens(width, height))),
    )
    scale = min(edge_scale, token_scale)
    if scale >= 0.995:
        return image, False
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # Ceilings in the token formula can leave the first estimate a few tokens over budget.
    while estimate_visual_tokens(new_width, new_height) > profile.visual_token_budget:
        new_width -= 1
        new_height = max(1, int(height * (new_width / width)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, True


def process_ecg_image(
    raw: bytes,
    model: str,
    max_bytes: int,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> ProcessedImage:
    loaded = load_image(raw, max_bytes)

    transformations: list[str] = []
    image, geometry = analyze_geometry(loaded.image)
    if geometry.rotation_applied:
        transformations.append(f"deskewed {geometry.angle_deg:+.1f}°")
    if geometry.crop_applied:
        transformations.append("cropped empty margins")

    # Preserve waveform morphology when the source already has good contrast. Non-linear
    # enhancement can make the grid/noise look like small deflections and is only useful
    # for genuinely faded or poorly exposed photographs.
    if geometry.contrast < 30 or geometry.brightness < 100 or geometry.brightness > 250:
        image = _enhance_for_ecg(image)
        transformations.append("light local contrast enhancement")
    else:
        transformations.append("contrast preserved")
    image, resized = _resize_to_profile(image, profile_for_model(model))
    if resized:
        transformations.append("resized to fit model image profile")

    height, width = image.shape[:2]
    success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not success:
        raise ImageProcessingError("Unable to encode processed image to PNG.")

    # Run on the final pixels, so every probability map and coordinate the
    # pipeline reports refers to the picture the model is actually shown.
    pipeline = run_pipeline(image, Preprocessing(loaded=loaded, geometry=geometry), config)
    grid = pipeline.grid
    if grid is not None and grid.confidence != "low":
        transformations.append(f"measured grid pitch {grid.small_cell_px:.1f} px/mm")
    return ProcessedImage(
        data=encoded.tobytes(),
        media_type="image/png",
        width=width,
        height=height,
        estimated_visual_tokens=estimate_visual_tokens(width, height),
        blur_score=geometry.blur_score,
        brightness=geometry.brightness,
        contrast=geometry.contrast,
        pipeline=pipeline,
        transformations=tuple(transformations),
        grid=grid,
    )


def annotate_findings(
    image: ProcessedImage,
    findings: list[LocalizedFinding],
) -> bytes:
    """Draw numbered boxes using the same normalized coordinate space seen by the model."""
    decoded = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ImageProcessingError("Unable to decode image for annotation overlay.")
    if not findings:
        return image.data

    height, width = decoded.shape[:2]
    palette = (
        (20, 20, 230),
        (0, 135, 255),
        (220, 90, 20),
    )
    rectangle_thickness = max(2, round(min(width, height) / 300))
    font_scale = max(0.65, min(1.8, min(width, height) / 700))
    text_thickness = max(2, rectangle_thickness)

    # A hollow outline disappears into a dense millimetre grid, so the region is tinted
    # first and the traces are then blended back over the tint.
    highlighted = decoded.copy()
    boxes: list[tuple[int, int, int, int, tuple[int, int, int]]] = []
    for marker, finding in enumerate(findings[:3], start=1):
        box = finding.box
        left = round(box.left * (width - 1) / 1000)
        top = round(box.top * (height - 1) / 1000)
        right = round(box.right * (width - 1) / 1000)
        bottom = round(box.bottom * (height - 1) / 1000)
        color = palette[(marker - 1) % len(palette)]
        boxes.append((left, top, right, bottom, color))
        cv2.rectangle(highlighted, (left, top), (right, bottom), color, -1)
    cv2.addWeighted(highlighted, 0.12, decoded, 0.88, 0, decoded)

    for marker, (left, top, right, bottom, color) in enumerate(boxes, start=1):
        cv2.rectangle(
            decoded,
            (left, top),
            (right, bottom),
            color,
            rectangle_thickness,
            cv2.LINE_AA,
        )

        label = str(marker)
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        pad = max(4, rectangle_thickness * 2)
        badge_width = text_width + pad * 2
        badge_height = text_height + baseline + pad * 2
        badge_left = min(left, max(0, width - 1 - badge_width))
        # Near the top edge there is no room above the box, and a badge clamped to zero
        # used to sit across the frame it labels. Drop it inside the box instead.
        badge_top = top - badge_height
        if badge_top < 0:
            badge_top = min(top + rectangle_thickness, max(0, height - 1 - badge_height))
        badge_right = min(width - 1, badge_left + badge_width)
        badge_bottom = min(height - 1, badge_top + badge_height)
        cv2.rectangle(
            decoded,
            (badge_left, badge_top),
            (badge_right, badge_bottom),
            color,
            -1,
        )
        text_x = badge_left + pad
        text_y = badge_top + pad + text_height
        cv2.putText(
            decoded,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    success, encoded = cv2.imencode(
        ".png",
        decoded,
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )
    if not success:
        raise ImageProcessingError("Unable to encode annotated image.")
    return encoded.tobytes()
