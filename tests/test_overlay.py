"""The marked-up sheet the user gets back, and the border trim it depends on."""

from __future__ import annotations

import cv2
import numpy as np
from test_imaging import make_ecg

from miocard.imaging import process_ecg_image
from miocard.overlay import (
    BANNER_HEIGHT,
    LEGEND_HEIGHT,
    annotate_digitisation,
    digitisation_summary,
    render_digitisation,
)
from miocard.pipeline.loader import load_image, trim_dead_border
from miocard.pipeline.state import ECGPipelineState


def _decoded(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def test_overlay_covers_the_sheet_and_adds_its_own_furniture() -> None:
    result = process_ecg_image(make_ecg(), "gpt-5.6-sol", 20 * 1024 * 1024)
    processed = _decoded(result.data)
    panel = render_digitisation(processed, result.pipeline)

    assert panel.shape[1] == processed.shape[1]
    assert panel.shape[0] == processed.shape[0] + BANNER_HEIGHT + LEGEND_HEIGHT


def test_overlay_draws_on_the_traces_it_reports() -> None:
    result = process_ecg_image(make_ecg(), "gpt-5.6-sol", 20 * 1024 * 1024)
    processed = _decoded(result.data)
    panel = render_digitisation(processed, result.pipeline)[
        BANNER_HEIGHT : BANNER_HEIGHT + processed.shape[0]
    ]

    # Green is only ever the measured pen, so its rows are where the drawing
    # claims the leads are.
    blue, green, red = panel[..., 0], panel[..., 1], panel[..., 2]
    measured = (green > 140) & (blue < 120) & (red < 120)
    assert measured.any()
    drawn_rows = np.flatnonzero(measured.any(axis=1))
    for path in result.pipeline.paths:
        assert np.any(np.abs(drawn_rows - path.band.baseline) < path.band.height)


def test_nothing_is_drawn_when_nothing_was_read() -> None:
    empty = ECGPipelineState(image_shape=(64, 64, 3))
    assert annotate_digitisation_of(empty) is None


def annotate_digitisation_of(state: ECGPipelineState) -> bytes | None:
    class _Stub:
        pipeline = state
        data = b""

    return annotate_digitisation(_Stub())  # type: ignore[arg-type]


def test_overlay_encodes_to_png() -> None:
    result = process_ecg_image(make_ecg(), "gpt-5.6-sol", 20 * 1024 * 1024)
    encoded = annotate_digitisation(result)
    assert encoded is not None
    assert encoded.startswith(b"\x89PNG")


def test_summary_states_what_was_read() -> None:
    result = process_ecg_image(make_ecg(), "gpt-5.6-sol", 20 * 1024 * 1024)
    summary = digitisation_summary(result.pipeline)
    assert "leads digitised" in summary
    assert "quality:" in summary


def test_uniform_black_border_line_is_dropped() -> None:
    image = np.full((300, 400, 3), 240, dtype=np.uint8)
    image[-1, :, :] = 0
    trimmed, origin = trim_dead_border(image)
    assert trimmed.shape[0] == 299
    assert origin == (0, 0)


def test_textured_dark_border_is_kept() -> None:
    rng = np.random.default_rng(7)
    image = np.full((300, 400, 3), 240, dtype=np.uint8)
    # A photographed dark edge - a table, a shadow - has texture, and throwing
    # it away would start eating the picture.
    image[-1, :, :] = rng.integers(0, 90, size=(400, 3), dtype=np.uint8)
    trimmed, origin = trim_dead_border(image)
    assert trimmed.shape[0] == 300
    assert origin == (0, 0)


def test_trim_is_recorded_so_coordinates_still_refer_to_the_upload() -> None:
    raw = make_ecg()
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    decoded[:, 0, :] = 0
    success, encoded = cv2.imencode(".png", decoded)
    assert success

    loaded = load_image(encoded.tobytes(), 20 * 1024 * 1024)
    assert loaded.origin == (1, 0)
    assert loaded.source_width == decoded.shape[1]
    assert "trimmed_dead_border" in loaded.notes


def test_source_transform_maps_the_upload_onto_the_analysed_picture() -> None:
    raw = make_ecg()
    result = process_ecg_image(raw, "gpt-5.6-sol", 20 * 1024 * 1024)
    transform = result.pipeline.source_transform
    assert transform is not None

    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    corners = np.array(
        [[0.0, 0.0], [float(decoded.shape[1]), 0.0], [0.0, float(decoded.shape[0])]]
    )
    mapped = corners @ transform[:, :2].T + transform[:, 2]
    # The analysed picture is a deskewed, trimmed, rescaled crop of the upload,
    # so the upload's corners land at or outside its edges, never inside it.
    assert mapped[0, 0] <= 1.0 and mapped[0, 1] <= 1.0
    assert mapped[1, 0] >= result.width - 1
    assert mapped[2, 1] >= result.height - 1
