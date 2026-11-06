from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from miocard.imaging import (
    annotate_findings,
    estimate_visual_tokens,
    measure_grid,
    process_ecg_image,
    profile_for_model,
)
from miocard.models import LocalizedFinding
from miocard.pipeline import signals_to_csv, to_json
from miocard.pipeline.state import CLASS_NAMES


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
    assert (
        "light local contrast enhancement" in result.transformations
        or "contrast preserved" in result.transformations
    )


def test_grid_pitch_is_measured_and_reported_to_the_model() -> None:
    result = process_ecg_image(make_ecg(1800, 1200), "gpt-5.6-sol", 20 * 1024 * 1024)

    assert result.grid is not None
    # make_ecg draws the grid every 20 px and the sheet is not resized at this size.
    assert result.grid.small_cell_px == pytest.approx(20.0, abs=0.5)
    assert result.grid.large_cell_px == pytest.approx(100.0, abs=2.5)
    assert result.grid.confidence in {"high", "medium"}

    metadata = result.metadata_text()
    assert "Grid pitch: small cell ~ 20.0 px" in metadata
    assert "large (5 mm) ~ 100.0 px" in metadata


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

    assert "Grid measurement skipped: not confident enough" in result.metadata_text()


def test_badge_stays_inside_the_image_for_a_box_at_the_top_edge() -> None:
    processed = process_ecg_image(make_ecg(1200, 800), "gpt-5.6-sol", 20 * 1024 * 1024)
    finding = LocalizedFinding.model_validate(
        {
            "finding": "Complaints in upper lead",
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
            "finding": "Test finding",
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


EXAMPLE_IMAGES = tuple((Path(__file__).resolve().parents[1] / "examples").glob("*.jpg"))


def test_examples_are_processed_with_pipeline_metadata() -> None:
    assert EXAMPLE_IMAGES, "No image files found in examples."
    for image_path in EXAMPLE_IMAGES:
        raw = image_path.read_bytes()
        result = process_ecg_image(raw, "gpt-5.6-sol", 20 * 1024 * 1024)

        assert result.pipeline is not None
        assert result.pipeline.quality_grade in {"diagnostic", "limited", "non_diagnostic"}
        assert len(result.pipeline.stages) == 15
        assert result.pipeline.image_shape == (result.height, result.width, 3)
        metadata = result.metadata_text()
        assert "MyocardBot2" in metadata


def test_pipeline_stage_probabilities_are_bounded_and_ordered() -> None:
    image_paths = tuple(EXAMPLE_IMAGES)
    assert image_paths, "No image files found in examples."
    result = process_ecg_image(image_paths[0].read_bytes(), "gpt-5.6-sol", 20 * 1024 * 1024)

    pipeline = result.pipeline
    assert pipeline is not None
    assert all(0.0 <= stage.probability <= 1.0 for stage in pipeline.stages)
    assert all(stage.stage for stage in pipeline.stages)
    assert pipeline.stages[0].stage == "Image loader"
    assert pipeline.stages[-1].stage == "Export"
    assert set(pipeline.cost_map.keys()) >= {"grid_probability", "noise_probability", "signal_probability", "artifact_probability"}
    assert "row_coverage" in pipeline.layout


def test_pipeline_returns_probability_maps_for_key_classes() -> None:
    image_paths = tuple(EXAMPLE_IMAGES)
    assert image_paths, "No image files found in examples."
    result = process_ecg_image(image_paths[0].read_bytes(), "gpt-5.6-sol", 20 * 1024 * 1024)

    pipeline = result.pipeline
    assert pipeline is not None
    maps = pipeline.probability_maps
    # MyocardBot2.md asks the cost map to store P(trace), P(text), P(grid),
    # P(noise) and P(artifact); background is what lets those five sum to one.
    assert set(CLASS_NAMES) == {"trace", "grid", "text", "noise", "artifact", "background"}
    assert set(CLASS_NAMES).issubset(maps.keys())
    for name in CLASS_NAMES:
        map_data = maps[name]
        assert map_data.shape == (result.height, result.width)
        assert map_data.dtype == np.float32
        assert np.isfinite(map_data).all()
        assert float(map_data.min()) >= 0.0
        assert float(map_data.max()) <= 1.0

    class_probability_sum = sum(maps[name] for name in CLASS_NAMES)
    assert np.isfinite(class_probability_sum).all()
    assert float(class_probability_sum.min()) >= 0.9999
    assert float(class_probability_sum.max()) <= 1.0001


def test_text_is_penalised_rather_than_erased() -> None:
    """The plan forbids destroying information: glyphs stay in the picture."""
    image_paths = tuple(EXAMPLE_IMAGES)
    assert image_paths, "No image files found in examples."
    raw = image_paths[0].read_bytes()
    result = process_ecg_image(raw, "gpt-5.6-sol", 20 * 1024 * 1024)
    pipeline = result.pipeline
    assert pipeline is not None
    assert pipeline.text_objects, "No semantic text objects were detected."

    processed = np.asarray(Image.open(io.BytesIO(result.data)).convert("L"), dtype=np.float32)
    text_map = pipeline.probability_maps["text"]
    inked = text_map > 0.2
    assert inked.any(), "Text class carries no probability mass."
    # Wherever the pipeline says 'text', the image is still darker than paper.
    assert float(processed[inked].mean()) < float(processed.mean())

    unary = pipeline.unary_cost
    assert unary is not None
    # And that text costs the solver something to walk through.
    assert float(unary[inked].mean()) > float(np.percentile(unary, 5))


def test_reconstruction_produces_one_path_per_lead_row() -> None:
    image_paths = tuple(EXAMPLE_IMAGES)
    assert image_paths, "No image files found in examples."
    result = process_ecg_image(image_paths[0].read_bytes(), "gpt-5.6-sol", 20 * 1024 * 1024)
    pipeline = result.pipeline
    assert pipeline is not None

    assert len(pipeline.bands) >= 3
    assert len(pipeline.paths) == len(pipeline.bands)
    for path in pipeline.paths:
        assert path.y.shape == (result.width,)
        assert np.isfinite(path.y).all()
        assert float(path.y.min()) >= 0 and float(path.y.max()) <= result.height - 1
        assert path.x_start < path.x_end
        assert path.confidence is not None and path.confidence.shape == (result.width,)

    # The solved paths must sit on ink, not wander across blank paper.
    trace = pipeline.probability_maps["trace"]
    for path in pipeline.paths:
        rows = np.clip(np.round(path.y).astype(int), 0, result.height - 1)
        columns = np.arange(result.width)
        on_path = trace[rows, columns][path.x_start : path.x_end]
        assert float(on_path.mean()) > float(trace.mean()) * 3


def test_signals_are_digitised_and_exportable() -> None:
    image_paths = tuple(EXAMPLE_IMAGES)
    assert image_paths, "No image files found in examples."
    result = process_ecg_image(image_paths[0].read_bytes(), "gpt-5.6-sol", 20 * 1024 * 1024)
    pipeline = result.pipeline
    assert pipeline is not None

    assert pipeline.lead_signals
    for signal in pipeline.lead_signals:
        assert np.isfinite(signal.values_mv).all()
        assert signal.values_mv.shape == (result.width,)
        assert float(signal.confidence.min()) >= 0.0
        assert float(signal.confidence.max()) <= 1.0

    payload = json.loads(to_json(pipeline))
    assert payload["schema"] == "miocard.pipeline/1"
    assert len(payload["signals"]["leads"]) == len(pipeline.lead_signals)
    assert payload["quality"]["grade"] in {"diagnostic", "limited", "non_diagnostic"}
    assert len(payload["stages"]) == 15

    csv = signals_to_csv(pipeline)
    assert csv.startswith("sample,seconds,")
    assert len(csv.strip().splitlines()) == result.width + 1


def test_measured_rate_is_only_reported_when_it_is_trustworthy() -> None:
    """A bpm from a low-confidence reconstruction must not reach the model."""
    image_paths = tuple(EXAMPLE_IMAGES)
    assert image_paths, "No image files found in examples."
    reported = 0
    for image_path in image_paths:
        result = process_ecg_image(image_path.read_bytes(), "gpt-5.6-sol", 20 * 1024 * 1024)
        pipeline = result.pipeline
        assert pipeline is not None
        rhythm = json.loads(to_json(pipeline))["rhythm"]
        text = result.metadata_text()
        if rhythm["measured"]:
            reported += 1
            assert 20.0 <= rhythm["heart_rate_bpm"] <= 300.0
            assert "Reconstructed rate" in text
        else:
            assert "was not measured reliably" in text
    assert reported >= 1, "No example produced a reportable rate."


def test_grade_never_claims_diagnostic_without_calibration() -> None:
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, (900, 1300, 3), dtype=np.uint8)
    result = process_ecg_image(_encode_png(noise), "gpt-5.6-sol", 20 * 1024 * 1024)
    pipeline = result.pipeline
    assert pipeline is not None
    assert pipeline.quality_grade in {"limited", "non_diagnostic"}
    assert float(pipeline.quality["probability"]) <= 0.72



def _pipelines() -> list:
    assert EXAMPLE_IMAGES, "No image files found in examples."
    return [
        process_ecg_image(path.read_bytes(), "gpt-5.6-sol", 20 * 1024 * 1024).pipeline
        for path in EXAMPLE_IMAGES
    ]


def test_every_lead_row_is_found_on_the_example_sheets() -> None:
    """All five examples are 12-lead prints; the detector must see the rows.

    One example is a curled page whose header sits across the first lead and
    whose chest leads overlap by more than a row pitch. Ordered label anchors
    recover its full row set when the ordinary ink tracker comes up short.
    """
    counts = [int(pipeline.layout["row_count"]) for pipeline in _pipelines()]
    assert counts == [12] * len(counts), f"lost lead rows: {counts}"


def test_lead_names_are_only_claimed_when_all_rows_were_found() -> None:
    for pipeline in _pipelines():
        named = [band.label for band in pipeline.bands if not band.label.startswith("row ")]
        if len(pipeline.bands) == 12:
            assert named[:3] == ["I", "II", "III"]
        else:
            assert not named, "named leads on a sheet with rows missing"


def test_lead_labels_are_located_for_most_rows() -> None:
    """The names are known; the algorithm has to find where each one is printed."""
    for pipeline in _pipelines():
        found = int(pipeline.text["labels_found"])
        assert found >= 0.6 * len(pipeline.bands), (
            f"only {found} of {len(pipeline.bands)} lead labels located"
        )


def test_margin_furniture_is_excluded_from_the_traces() -> None:
    """A lead's own name and its calibration step precede the recording."""
    for pipeline in _pipelines():
        edges = pipeline.text.get("margin_edges", {})
        for path in pipeline.paths:
            edge = int(edges.get(path.index, 0))
            if edge:
                assert path.x_start >= edge, (
                    f"{path.label} starts at {path.x_start}, inside margin furniture ending {edge}"
                )


def test_paths_never_reach_a_neighbouring_lead() -> None:
    """The failure this guards against is a path adopting the next lead's trace."""
    for pipeline in _pipelines():
        pitch = float(pipeline.layout["row_pitch_px"])
        assert pitch > 0
        for path in pipeline.paths:
            traced = path.y[path.x_start : path.x_end]
            excursion = float(np.abs(traced - path.band.baseline).max()) / pitch
            assert excursion < 0.9, (
                f"{path.label} strayed {excursion:.2f} row pitches from its baseline"
            )


def test_lead_baselines_are_ordered_and_separated() -> None:
    for pipeline in _pipelines():
        baselines = [band.baseline for band in pipeline.bands]
        assert baselines == sorted(baselines)
        spacing = np.diff(baselines)
        assert float(spacing.min()) > 0.4 * float(np.median(spacing))


def test_line_start_is_reported_with_a_confidence() -> None:
    for pipeline in _pipelines():
        for path in pipeline.paths:
            assert 0.0 <= path.start_probability <= 1.0
            assert 0.0 <= path.end_probability <= 1.0
        # A printed lead begins abruptly on blank margin, so most starts are sharp.
        sharp = sum(path.start_probability > 0.5 for path in pipeline.paths)
        assert sharp >= len(pipeline.paths) * 0.6


def test_text_lines_cover_the_lead_labels_and_footer() -> None:
    for pipeline in _pipelines():
        kinds = {obj.kind for obj in pipeline.text_objects}
        assert "lead_label" in kinds, f"no lead labels detected, found {kinds}"
        labels = [obj for obj in pipeline.text_objects if obj.kind == "lead_label"]
        # Labels sit in the left margin, one per lead row at most.
        assert len(labels) <= len(pipeline.bands) + 2
        width = pipeline.image_shape[1]
        assert all(obj.left <= 0.2 * width for obj in labels)
