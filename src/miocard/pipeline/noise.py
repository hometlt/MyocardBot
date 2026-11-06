"""Stage 7 - Noise / artifact detector.

Noise and artifacts are separated because they behave differently. Noise is
high-frequency speckle that makes a pixel unreliable but leaves the trace
findable; an artifact - glare, a shadow, a fold, a locally blurred patch - hides
the trace altogether over a region. The path solver pays a small toll for the
first and a large one for the second.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from miocard.pipeline.ink import InkLayers
from miocard.pipeline.state import PipelineStageResult, clamp01


@dataclass(frozen=True)
class NoiseReport:
    noise: np.ndarray
    artifact: np.ndarray
    noise_statistics: dict[str, float]
    artifact_statistics: dict[str, float]
    probability: float


def _speckle(layers: InkLayers) -> np.ndarray:
    """Ink that disagrees with its own immediate neighbourhood."""
    smoothed = cv2.medianBlur((layers.ink * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    residual = np.abs(layers.ink - smoothed)
    scale = max(float(np.percentile(residual, 99.0)), 1e-3)
    return np.clip(residual / scale, 0.0, 1.0).astype(np.float32)


def _glare_and_shadow(image: np.ndarray, layers: InkLayers, px_per_mm: float) -> np.ndarray:
    """Regions where the paper itself carries no readable ink.

    Both tests are relative to the rest of the sheet, because absolute levels
    say nothing useful: a well-exposed photo of white ECG paper sits at 250-255
    over its whole area, so thresholding on brightness would flag every gap
    between two traces as glare. Real glare is a patch that is simultaneously
    among the brightest on the sheet *and* among the least inked - the printed
    grid has been washed away there, which is what makes it unusable.
    """
    paper = layers.background.astype(np.float32)

    window = max(9, int(round(4.0 * max(px_per_mm, 3.0))))
    if window % 2 == 0:
        window += 1
    inked = cv2.blur(layers.ink, (window, window))
    # The quiet level of the sheet: bare paper between traces still carries grid.
    floor = max(float(np.percentile(inked, 20)), 1e-4)
    inkless = np.clip((floor - inked) / floor, 0.0, 1.0)
    ceiling = float(np.percentile(paper, 95))
    blown = np.clip((paper - ceiling) / 4.0, 0.0, 1.0)
    glare = blown * inkless

    # Shadow: the paper itself is far darker than the rest of the sheet.
    reference = float(np.percentile(paper, 60)) or 1.0
    shadow = np.clip((reference * 0.62 - paper) / max(reference * 0.30, 1.0), 0.0, 1.0)
    return np.maximum(glare, shadow).astype(np.float32)


def _local_blur(image: np.ndarray, layers: InkLayers, px_per_mm: float) -> np.ndarray:
    """Where ink is present but its edges have gone soft.

    Blank paper has no edge energy either, so the deficit only counts where
    there is ink to be blurred; otherwise every margin on the sheet would be
    reported as an artifact.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    window = max(9, int(round(6.0 * max(px_per_mm, 3.0))))
    if window % 2 == 0:
        window += 1
    energy = cv2.blur(laplacian * laplacian, (window, window))
    inked = cv2.blur(layers.strong, (window, window))
    reference = float(np.percentile(energy[inked > 0.01], 85)) if (inked > 0.01).any() else 0.0
    if reference <= 1e-6:
        return np.zeros_like(energy, dtype=np.float32)
    deficit = np.clip(1.0 - energy / (0.18 * reference), 0.0, 1.0)
    presence = np.clip(inked / max(float(np.percentile(inked, 90)), 1e-4), 0.0, 1.0)
    return (deficit * presence).astype(np.float32)


def detect_noise(
    image: np.ndarray,
    layers: InkLayers,
    trace_evidence: np.ndarray,
    px_per_mm: float,
    blur_score: float,
    contrast: float,
) -> NoiseReport:
    speckle = _speckle(layers)
    # Speckle riding on the trace is part of the pen stroke, not noise.
    noise = np.clip(speckle * (1.0 - 0.75 * trace_evidence), 0.0, 1.0).astype(np.float32)

    glare_shadow = _glare_and_shadow(image, layers, px_per_mm)
    blur = _local_blur(image, layers, px_per_mm)
    # A blurred patch only matters where the sheet is supposed to carry ink.
    artifact = np.maximum(glare_shadow, blur * 0.75).astype(np.float32)
    artifact = cv2.GaussianBlur(artifact, (0, 0), max(1.0, px_per_mm / 2.0))
    artifact = np.clip(artifact, 0.0, 1.0).astype(np.float32)

    noise_statistics = {
        "mean": float(noise.mean()),
        "p95": float(np.percentile(noise, 95)),
        "blur_score": float(blur_score),
        "contrast": float(contrast),
    }
    artifact_statistics = {
        "mean": float(artifact.mean()),
        "coverage": float((artifact > 0.5).mean()),
        "glare_fraction": float((glare_shadow > 0.5).mean()),
        "blur_fraction": float((blur > 0.5).mean()),
    }

    # The stage is confident when it can say the sheet is clean, and equally
    # confident when it can say a quarter of it is unusable; it is unconfident
    # in between, where a marginal artifact may or may not hide a trace.
    burden = clamp01(0.6 * artifact_statistics["mean"] * 4.0 + 0.4 * noise_statistics["mean"] * 3.0)
    probability = clamp01(1.0 - burden)
    return NoiseReport(
        noise=noise,
        artifact=artifact,
        noise_statistics=noise_statistics,
        artifact_statistics=artifact_statistics,
        probability=probability,
    )


def noise_stage(report: NoiseReport) -> PipelineStageResult:
    return PipelineStageResult(
        "Noise / artifact detector",
        report.probability,
        f"Artifacts cover {report.artifact_statistics['coverage'] * 100:.1f}% of the sheet; "
        f"mean speckle {report.noise_statistics['mean']:.3f}.",
        evidence=(
            f"artifact_coverage={report.artifact_statistics['coverage']:.4f}",
            f"glare_fraction={report.artifact_statistics['glare_fraction']:.4f}",
            f"blur_fraction={report.artifact_statistics['blur_fraction']:.4f}",
            f"noise_mean={report.noise_statistics['mean']:.3f}",
            f"laplacian_variance={report.noise_statistics['blur_score']:.1f}",
        ),
    )
