from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from miocard.imaging import (
    annotate_findings,
    create_rhythm_views,
    detect_rhythm_timing_hints,
    estimate_visual_tokens,
    measure_grid,
    process_ecg_image,
    profile_for_model,
)
from miocard.models import LocalizedFinding


def _encode_png(pixels: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def make_ecg(width: int = 1800, height: int = 1200) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 20):
        draw.line((x, 0, x, height), fill=(235, 210, 210), width=1)
    for y in range(0, height, 20):
        draw.line((0, y, width, y), fill=(235, 210, 210), width=1)
    points = []
    for x in range(30, width - 30):
        y = height // 2
        phase = x % 180
        if 70 <= phase < 76:
            y -= (phase - 70) * 45
        elif 76 <= phase < 82:
            y += (82 - phase) * 45
        points.append((x, y))
    draw.line(points, fill="black", width=3)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92)
    return output.getvalue()


def test_processing_stays_within_opus_visual_budget() -> None:
    result = process_ecg_image(make_ecg(), "claude-opus-4-6", 20 * 1024 * 1024)
    profile = profile_for_model("claude-opus-4-6")
    assert result.media_type == "image/png"
    assert max(result.width, result.height) <= profile.max_edge
    assert result.estimated_visual_tokens <= profile.visual_token_budget
    assert result.data.startswith(b"\x89PNG")


def test_sonnet_profile_preserves_useful_resolution() -> None:
    result = process_ecg_image(make_ecg(1200, 800), "claude-sonnet-5", 20 * 1024 * 1024)
    assert result.width <= 1200
    assert result.height <= 800
    assert result.estimated_visual_tokens == estimate_visual_tokens(result.width, result.height)
    assert "контраст исходника сохранен" in result.transformations


def test_grid_pitch_is_measured_and_reported_to_the_model() -> None:
    result = process_ecg_image(make_ecg(1800, 1200), "gpt-5.6-sol", 20 * 1024 * 1024)

    assert result.grid is not None
    # make_ecg draws the grid every 20 px and the sheet is not resized at this size.
    assert result.grid.small_cell_px == pytest.approx(20.0, abs=0.5)
    assert result.grid.large_cell_px == pytest.approx(100.0, abs=2.5)
    assert result.grid.confidence in {"high", "medium"}

    metadata = result.metadata_text()
    assert "малая клетка ≈ 20.0 px" in metadata
    assert "большая (5 мм) ≈ 100.0 px" in metadata


def test_grid_measurement_rejects_an_image_without_a_square_grid() -> None:
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (900, 1300, 3), dtype=np.uint8)

    grid = measure_grid(noise)

    # Noise may still produce a peak on one axis; the axes must agree to be trusted.
    assert grid is None or grid.confidence == "low"


def test_metadata_says_so_when_the_grid_cannot_be_measured() -> None:
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (900, 1300, 3), dtype=np.uint8)
    result = process_ecg_image(
        _encode_png(noise), "gpt-5.6-sol", 20 * 1024 * 1024
    )

    assert "шаг миллиметровой сетки измерить не удалось" in result.metadata_text()


def test_rhythm_views_cover_the_full_sheet_with_overlap() -> None:
    processed = process_ecg_image(make_ecg(1200, 800), "gpt-5.6-sol", 20 * 1024 * 1024)

    views = create_rhythm_views(processed)

    assert [(view.name, view.left, view.right) for view in views] == [
        ("left", 0, 600),
        ("right", 480, 1000),
        ("terminal", 600, 1000),
    ]
    # This sheet already fits the visual budget, so there is nothing to magnify and the
    # fragments stay plain crops.
    assert processed.source_data is None
    assert all(view.image.height == processed.height for view in views)
    assert [view.image.width for view in views] == [720, 624, 480]
    assert all(view.image.grid == processed.grid for view in views)
    assert "увеличенный фрагмент" in views[0].image.transformations[-1]


def test_rhythm_views_are_magnified_when_the_sheet_was_downscaled() -> None:
    processed = process_ecg_image(make_ecg(4000, 2600), "claude-opus-4-6", 20 * 1024 * 1024)
    profile = profile_for_model("claude-opus-4-6")

    views = create_rhythm_views(processed)

    assert processed.source_data is not None
    assert processed.source_width > processed.width
    # A crop of the delivered sheet would be 60% of its width. The fragment is cut from
    # the retained source instead, so it carries more pixels per millimetre than the
    # full sheet the model sees.
    assert views[0].image.width > round(processed.width * 0.6)
    assert "детализация ×" in views[0].image.transformations[-1]
    assert all(
        view.image.estimated_visual_tokens <= profile.visual_token_budget for view in views
    )
    if processed.grid is not None:
        assert views[0].image.grid is not None
        assert views[0].image.grid.small_cell_px > processed.grid.small_cell_px


def test_timing_prepass_finds_abrupt_short_rr_intervals() -> None:
    width, height = 1200, 800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 20):
        draw.line((x, 0, x, height), fill=(235, 210, 210), width=1)
    for y in range(0, height, 20):
        draw.line((0, y, width, y), fill=(235, 210, 210), width=1)
    peaks = [150, 250, 350, 430, 550, 630, 750, 850, 950, 1030]
    for y in (500, 555, 610, 665, 720):
        draw.line((50, y, 1150, y), fill="black", width=2)
        for x in peaks:
            draw.line((x, y - 25, x, y + 25), fill="black", width=3)
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    processed = process_ecg_image(
        raw.getvalue(), "gpt-5.6-sol", 20 * 1024 * 1024
    )

    hints = detect_rhythm_timing_hints(processed)

    assert [hint.x for hint in hints] == [359, 525, 859]
    assert all(hint.coupling_ratio == pytest.approx(0.8) for hint in hints)
    assert all(hint.confidence == "high" for hint in hints)
    assert all(hint.kind == "premature" for hint in hints)


def test_timing_prepass_flags_an_abruptly_long_interval_as_a_pause() -> None:
    width, height = 1200, 800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 20):
        draw.line((x, 0, x, height), fill=(235, 210, 210), width=1)
    for y in range(0, height, 20):
        draw.line((0, y, width, y), fill=(235, 210, 210), width=1)
    # One beat is missing from an otherwise metronomic sequence, which no premature-only
    # scan would notice.
    peaks = [150, 250, 350, 450, 550, 750, 850, 950, 1050]
    for y in (500, 555, 610, 665, 720):
        draw.line((50, y, 1150, y), fill="black", width=2)
        for x in peaks:
            draw.line((x, y - 25, x, y + 25), fill="black", width=3)
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    processed = process_ecg_image(raw.getvalue(), "gpt-5.6-sol", 20 * 1024 * 1024)

    hints = detect_rhythm_timing_hints(processed)

    assert [hint.kind for hint in hints] == ["pause"]
    assert hints[0].coupling_ratio == pytest.approx(2.0)
    assert hints[0].confidence == "high"
    # The candidate points at the gap itself, between the beats on either side.
    assert hints[0].x == pytest.approx(543, abs=25)


def test_panorama_is_cropped_to_ecg_band_and_timing_scan_follows_it() -> None:
    width, height = 1200, 800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    band_top, band_bottom = 260, 500
    for x in range(0, width, 20):
        draw.line((x, band_top, x, band_bottom), fill=(235, 210, 210), width=1)
    for y in range(band_top, band_bottom + 1, 20):
        draw.line((0, y, width, y), fill=(235, 210, 210), width=1)

    peaks = [150, 250, 350, 430, 550, 630, 750, 850, 950, 1030]
    for y in (330, 375, 420, 465):
        draw.line((50, y, 1150, y), fill="black", width=2)
        for x in peaks:
            draw.line((x, y - 22, x, y + 22), fill="black", width=3)
    # A short photographed page edge must not be mistaken for the ECG content band.
    draw.rectangle((0, 770, 500, 799), fill=(80, 80, 80))

    raw = io.BytesIO()
    image.save(raw, format="PNG")
    processed = process_ecg_image(
        raw.getvalue(), "gpt-5.6-sol", 20 * 1024 * 1024
    )
    hints = detect_rhythm_timing_hints(processed)

    assert processed.height < 400
    assert "удаление пустых полей" in processed.transformations
    assert [hint.x for hint in hints] == [359, 525, 859]


def test_badge_stays_inside_the_image_for_a_box_at_the_top_edge() -> None:
    processed = process_ecg_image(make_ecg(1200, 800), "gpt-5.6-sol", 20 * 1024 * 1024)
    finding = LocalizedFinding.model_validate(
        {
            "finding": "Комплекс у верхнего края",
            "related_observation": "rhythm",
            "box": {"top": 0, "left": 50, "bottom": 200, "right": 500},
        }
    )

    annotated = annotate_findings(processed, [finding])
    with Image.open(io.BytesIO(annotated)) as image:
        pixels = np.asarray(image.convert("RGB"))

    # The badge is drawn inside the box, so the numeral's white pixels must sit below
    # the top edge rather than being clipped away against it.
    left = round(50 * (processed.width - 1) / 1000)
    badge_region = pixels[: processed.height // 4, left : left + 80]
    assert (badge_region > 240).all(axis=2).any()


def test_annotate_findings_draws_numbered_boxes_without_resizing() -> None:
    processed = process_ecg_image(make_ecg(1200, 800), "gpt-5.6-sol", 20 * 1024 * 1024)
    finding = LocalizedFinding.model_validate(
        {
            "finding": "Тестовый комплекс",
            "related_observation": "rhythm",
            "box": {"top": 250, "left": 300, "bottom": 650, "right": 700},
        }
    )

    annotated = annotate_findings(processed, [finding])
    with Image.open(io.BytesIO(annotated)) as image:
        assert image.size == (processed.width, processed.height)
        colors = image.convert("RGB").getcolors(maxcolors=image.width * image.height)

    assert annotated.startswith(b"\x89PNG")
    assert annotated != processed.data
    assert colors is not None
    assert any(red > 180 and green < 100 for _, (red, green, _) in colors)
