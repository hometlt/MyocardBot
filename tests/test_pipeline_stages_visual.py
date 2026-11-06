"""Visual tests for each of the 15 MyocardBot2 pipeline stages.

For every stage (1–15) this module:
  1. Generates a synthetic multi-lead ECG image.
  2. Processes it through ``process_ecg_image``.
  3. Renders the stage output as a diagnostic image.
  4. Saves the image to ``tmp-test-results/<timestamp>/``.
  5. Asserts the image is valid (non‑empty, correct dimensions, finite values).

Run with::

    python -m pytest tests/test_pipeline_stages_visual.py -v --no-header
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from miocard.imaging import process_ecg_image
from miocard.pipeline.state import CLASS_NAMES
from miocard.stage_render import (
    add_label,
    render_all_stages,
    render_probability_maps_board,
    render_reconstruction_overlay,
    render_stage,
)

# ---------------------------------------------------------------------------
# Synthetic ECG image generator
# ---------------------------------------------------------------------------

_LEAD_Y_OFFSETS = [0, 200, 400, 600, 800, 1000]
_LEAD_AMPLITUDES = [40, 55, 35, 50, 45, 60]
_LEAD_PHASE_SHIFTS = [0, 30, 60, 90, 120, 150]


def make_multilead_ecg(
    width: int = 1800,
    height: int = 1200,
    grid_spacing: int = 20,
) -> bytes:
    """Create a synthetic 6-row ECG with grid and QRS-like complexes."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # ---- millimetre grid (light red) ----
    for x in range(0, width, grid_spacing):
        draw.line((x, 0, x, height), fill=(235, 210, 210), width=1)
    for y in range(0, height, grid_spacing):
        draw.line((0, y, width, y), fill=(235, 210, 210), width=1)

    # ---- thicker grid at 5-cell intervals ----
    for x in range(0, width, grid_spacing * 5):
        draw.line((x, 0, x, height), fill=(220, 190, 190), width=2)
    for y in range(0, height, grid_spacing * 5):
        draw.line((0, y, width, y), fill=(220, 190, 190), width=2)

    # ---- lead labels (left margin) ----
    labels = ["I", "II", "III", "aVR", "aVL", "aVF"]
    for idx, label in enumerate(labels):
        cy = _LEAD_Y_OFFSETS[idx] + 100
        draw.text((8, cy - 6), label, fill=(0, 0, 0))

    # ---- ECG traces (one per lead row) ----
    for lead_idx, y_offset in enumerate(_LEAD_Y_OFFSETS):
        amplitude = _LEAD_AMPLITUDES[lead_idx]
        phase_shift = _LEAD_PHASE_SHIFTS[lead_idx]
        y_center = y_offset + 100
        points: list[tuple[int, int]] = []
        for x in range(40, width - 40):
            phase = (x + phase_shift) % 160
            y = y_center
            # Simulate a QRS complex
            if 70 <= phase < 74:
                y -= amplitude * 1.5 * (phase - 70) / 4
            elif 74 <= phase < 76:
                y -= amplitude * 1.5
            elif 76 <= phase < 80:
                y += amplitude * 1.5 * (phase - 76) / 4
            elif 80 <= phase < 82:
                y += amplitude
            elif 82 <= phase < 86:
                y -= amplitude * 0.5 * (phase - 82) / 4
            # P wave
            elif 20 <= phase < 30:
                y -= amplitude * 0.2 * np.sin(np.pi * (phase - 20) / 10)
            # T wave
            elif 120 <= phase < 145:
                y -= amplitude * 0.35 * np.sin(np.pi * (phase - 120) / 25)
            points.append((x, int(max(0, min(height - 1, y)))))
        draw.line(points, fill="black", width=3)

    # ---- calibration rectangle (10 mm = 1 mV) ----
    cal_x = width - 160
    cal_y = height - 70
    draw.rectangle([cal_x, cal_y - 50, cal_x + 50, cal_y], outline=(0, 0, 0), width=2)

    # ---- speed text ----
    draw.text((width - 160, height - 90), "25 mm/s", fill=(80, 80, 80))
    draw.text((width - 160, height - 105), "10 mm/mV", fill=(80, 80, 80))

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92)
    return output.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TEST_DIR = Path(__file__).resolve().parents[1] / "tmp-test-results"
_OUTPUT_DIR = _TEST_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_pipeline_stages")


@pytest.fixture(scope="session")
def output_dir() -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


@pytest.fixture(scope="session")
def ecg_image() -> bytes:
    return make_multilead_ecg()


@pytest.fixture(scope="session")
def processed(ecg_image: bytes) -> tuple[np.ndarray, np.ndarray, object]:
    """Return (source, processed, ProcessedImage)."""
    raw = np.frombuffer(ecg_image, dtype=np.uint8)
    source = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    assert source is not None, "Failed to decode source image"

    result = process_ecg_image(ecg_image, "gpt-5.6-sol", 20 * 1024 * 1024)
    processed_arr = cv2.imdecode(np.frombuffer(result.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert processed_arr is not None, "Failed to decode processed image"

    return source, processed_arr, result


# ---------------------------------------------------------------------------
# Stage names for display
# ---------------------------------------------------------------------------

STAGE_NAMES = {
    1: "Image loader",
    2: "Geometry analyzer",
    3: "Grid detector",
    4: "Layout detector",
    5: "OCR & semantic analysis",
    6: "Trace probability estimator",
    7: "Noise / artifact detector",
    8: "Cost map builder",
    9: "Path solver",
    10: "Gap recovery",
    11: "Beat detector",
    12: "Template builder",
    13: "Signal optimizer",
    14: "Quality analyzer",
    15: "Export",
}


# ---------------------------------------------------------------------------
# Per-stage visual tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage_idx", list(range(1, 16)))
def test_stage_renders_valid_image(
    stage_idx: int,
    processed: tuple[np.ndarray, np.ndarray, object],
    output_dir: Path,
) -> None:
    """Each stage renders a non‑empty, finite image."""
    source, processed_arr, result = processed
    # Unpack ProcessedImage from the object
    from miocard.imaging import ProcessedImage as PI

    assert isinstance(result, PI)

    panel = render_stage(stage_idx, source, processed_arr, result)
    assert panel is not None
    assert panel.ndim == 3, f"Stage {stage_idx} panel should be 3‑channel"
    assert panel.shape[2] == 3, f"Stage {stage_idx} panel should be BGR"
    assert panel.size > 0, f"Stage {stage_idx} panel is empty"
    assert np.isfinite(panel).all(), f"Stage {stage_idx} panel has non‑finite values"

    # Check the stage result metadata
    assert result.pipeline is not None
    assert len(result.pipeline.stages) == 15
    stage_info = result.pipeline.stages[stage_idx - 1]
    assert stage_info.stage == STAGE_NAMES[stage_idx]
    assert 0.0 <= stage_info.probability <= 1.0

    # Save (sanitize filename: replace non-alphanumeric chars except _ and .)
    safe_name = STAGE_NAMES[stage_idx].lower().replace(" ", "_").replace("/", "_")
    name = f"{stage_idx:02d}_{safe_name}.png"
    path = output_dir / name
    success = cv2.imwrite(str(path), panel)
    assert success, f"Failed to write {path}"
    assert path.exists() and path.stat().st_size > 0, f"Saved image {path} is empty"


# ---------------------------------------------------------------------------
# Composite board tests
# ---------------------------------------------------------------------------


def test_full_stage_board_renders(
    processed: tuple[np.ndarray, np.ndarray, object],
    output_dir: Path,
) -> None:
    """The full 1–15 stage board renders as a single tall image."""
    source, processed_arr, result = processed
    from miocard.imaging import ProcessedImage as PI

    assert isinstance(result, PI)

    board = render_all_stages(source, processed_arr, result, card_width=800)
    assert board is not None
    assert board.ndim == 3
    assert board.shape[2] == 3
    assert board.shape[0] > 0, "Board has zero height"
    assert np.isfinite(board).all()

    path = output_dir / "board_all_stages.png"
    success = cv2.imwrite(str(path), board)
    assert success
    assert path.exists() and path.stat().st_size > 0


def test_stage_board_subset(
    processed: tuple[np.ndarray, np.ndarray, object],
    output_dir: Path,
) -> None:
    """A subset of stages (4–8) renders correctly."""
    source, processed_arr, result = processed
    from miocard.imaging import ProcessedImage as PI

    assert isinstance(result, PI)

    board = render_all_stages(
        source, processed_arr, result, card_width=800, stage_start=4, stage_end=8
    )
    assert board.shape[0] > 0

    path = output_dir / "board_stages_04_08.png"
    success = cv2.imwrite(str(path), board)
    assert success


# ---------------------------------------------------------------------------
# Probability maps tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("map_name", list(CLASS_NAMES))
def test_individual_probability_map(
    map_name: str,
    processed: tuple[np.ndarray, np.ndarray, object],
    output_dir: Path,
) -> None:
    """Each probability map renders as a valid composite image."""
    source, processed_arr, result = processed
    from miocard.imaging import ProcessedImage as PI

    assert isinstance(result, PI)
    maps = result.pipeline.probability_maps if result.pipeline else {}
    assert map_name in maps, f"Probability map '{map_name}' not found in pipeline state"

    map_data = maps[map_name]
    assert map_data.shape == (result.height, result.width)
    assert map_data.dtype == np.float32
    assert np.isfinite(map_data).all()
    assert float(map_data.min()) >= 0.0
    assert float(map_data.max()) <= 1.0

    # Render the probability map via the board renderer
    panels = render_probability_maps_board(processed_arr, result)
    panel_names = [p[0] for p in panels]
    assert map_name in panel_names

    for name, panel in panels:
        if name == map_name:
            path = output_dir / f"probability_map_{name}.png"
            success = cv2.imwrite(str(path), panel)
            assert success
            assert path.stat().st_size > 0


def test_probability_maps_sum_to_one(
    processed: tuple[np.ndarray, np.ndarray, object],
) -> None:
    """Per-pixel sum of every class probability is ≈ 1.0."""
    source, processed_arr, result = processed
    from miocard.imaging import ProcessedImage as PI

    assert isinstance(result, PI)
    maps = result.pipeline.probability_maps
    assert maps is not None

    prob_sum = sum(maps[name] for name in CLASS_NAMES)
    assert np.isfinite(prob_sum).all()
    assert float(prob_sum.min()) >= 0.9999
    assert float(prob_sum.max()) <= 1.0001


# ---------------------------------------------------------------------------
# Pipeline metadata and stage ordering test
# ---------------------------------------------------------------------------


def test_pipeline_has_all_15_stages(
    processed: tuple[np.ndarray, np.ndarray, object],
) -> None:
    """The pipeline state contains exactly 15 stages in the correct order."""
    source, processed_arr, result = processed
    from miocard.imaging import ProcessedImage as PI

    assert isinstance(result, PI)
    assert result.pipeline is not None
    stages = result.pipeline.stages
    assert len(stages) == 15

    expected_order = [
        "Image loader",
        "Geometry analyzer",
        "Grid detector",
        "Layout detector",
        "OCR & semantic analysis",
        "Trace probability estimator",
        "Noise / artifact detector",
        "Cost map builder",
        "Path solver",
        "Gap recovery",
        "Beat detector",
        "Template builder",
        "Signal optimizer",
        "Quality analyzer",
        "Export",
    ]
    for stage, expected in zip(stages, expected_order, strict=True):
        assert stage.stage == expected, f"Expected '{expected}', got '{stage.stage}'"


# ---------------------------------------------------------------------------
# Edge case: noise-only image
# ---------------------------------------------------------------------------


def test_noise_image_stages_are_bounded(
    output_dir: Path,
) -> None:
    """A noise image produces bounded stage probabilities and valid renderings."""
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 255, (900, 1300, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(noise).save(buf, format="PNG")
    raw = buf.getvalue()

    result = process_ecg_image(raw, "gpt-5.6-sol", 20 * 1024 * 1024)
    assert result.pipeline is not None
    assert len(result.pipeline.stages) == 15

    for stage in result.pipeline.stages:
        assert 0.0 <= stage.probability <= 1.0, f"{stage.stage} p={stage.probability} out of bounds"

    source = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    processed_arr = cv2.imdecode(
        np.frombuffer(result.data, dtype=np.uint8), cv2.IMREAD_COLOR
    )

    # Render and save stage 1 to verify noise path
    panel = render_stage(1, source, processed_arr, result)
    path = output_dir / "noise_stage_1.png"
    success = cv2.imwrite(str(path), panel)
    assert success


# ---------------------------------------------------------------------------
# Verify that the overview image is valid
# ---------------------------------------------------------------------------


def test_overview_image_renders(
    processed: tuple[np.ndarray, np.ndarray, object],
    output_dir: Path,
) -> None:
    """The overview (source + processed + metadata) is a valid image."""
    source, processed_arr, result = processed
    from miocard.imaging import ProcessedImage as PI

    assert isinstance(result, PI)

    # Build a side-by-side overview
    target_height = max(source.shape[0], processed_arr.shape[0])
    s = np.full((target_height, source.shape[1], 3), 245, dtype=np.uint8)
    s[: source.shape[0]] = source
    p = np.full((target_height, processed_arr.shape[1], 3), 245, dtype=np.uint8)
    p[: processed_arr.shape[0]] = processed_arr
    gap = np.full((target_height, 12, 3), 220, dtype=np.uint8)
    body = np.hstack((s, gap, p))

    # Metadata text
    meta_lines = [
        f"Source size: {source.shape[1]}x{source.shape[0]}",
        f"Processed size: {result.width}x{result.height}",
        f"Visual tokens: {result.estimated_visual_tokens}",
        f"Grid: {result.grid.small_cell_px:.1f} px" if result.grid else "Grid: none",
        f"Quality grade: {result.pipeline.quality_grade}",
        "",
        "Pipeline stages:",
    ]
    for stage in result.pipeline.stages:
        meta_lines.append(f"  {stage.stage}: p={stage.probability:.2f} ({stage.confidence})")

    from miocard.stage_render import draw_text_block

    text_block = draw_text_block(meta_lines, body.shape[1], 360)
    overview = np.vstack((body, text_block))
    overview = add_label(overview, "MyocardBot2 Pipeline Overview")

    path = output_dir / "overview.png"
    success = cv2.imwrite(str(path), overview)
    assert success
    assert path.stat().st_size > 0

# ---------------------------------------------------------------------------
# Reconstruction overlay - the check that the paths land on the printed traces
# ---------------------------------------------------------------------------


def test_reconstruction_overlay_tracks_the_synthetic_traces(
    processed: tuple[np.ndarray, np.ndarray, object],
    output_dir: Path,
) -> None:
    """Every solved path must sit on the trace the generator drew."""
    _source, processed_arr, result = processed
    pipeline = result.pipeline
    assert pipeline is not None
    assert pipeline.paths, "No paths were solved for the synthetic sheet."

    trace = pipeline.probability_maps["trace"]
    columns = np.arange(result.width)
    for path in pipeline.paths:
        rows = np.clip(np.round(path.y).astype(int), 0, result.height - 1)
        on_path = trace[rows, columns][path.x_start : path.x_end]
        assert float(on_path.mean()) > 0.25, (
            f"Path for {path.label} left the trace (mean P(trace)={on_path.mean():.3f})"
        )

    panel = render_reconstruction_overlay(processed_arr, result)
    path = output_dir / "reconstruction_overlay.png"
    assert cv2.imwrite(str(path), panel)
    assert path.stat().st_size > 0


def test_synthetic_beats_match_the_generated_rhythm(
    processed: tuple[np.ndarray, np.ndarray, object],
) -> None:
    """make_multilead_ecg repeats a QRS every 160 px; the rate must follow."""
    _source, _processed_arr, result = processed
    pipeline = result.pipeline
    assert pipeline is not None
    assert pipeline.grid is not None

    beats = pipeline.beats
    assert beats is not None, "No beats detected on a clean synthetic sheet."
    expected_bpm = 60.0 / (160.0 / (pipeline.grid.small_cell_px * 25.0))
    assert beats.heart_rate_bpm == pytest.approx(expected_bpm, rel=0.10)
    assert beats.regularity > 0.6


def test_lead_rows_are_found_and_named(
    processed: tuple[np.ndarray, np.ndarray, object],
) -> None:
    _source, _processed_arr, result = processed
    pipeline = result.pipeline
    assert pipeline is not None
    # The generator draws six evenly spaced leads.
    assert len(pipeline.bands) == 6
    assert [band.label for band in pipeline.bands] == [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
    ]
    baselines = [band.baseline for band in pipeline.bands]
    assert all(later > earlier for earlier, later in zip(baselines, baselines[1:], strict=False))
