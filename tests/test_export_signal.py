"""The digitised waveform: what goes to the model, and what goes in the record."""

from __future__ import annotations

import json

import numpy as np
from test_imaging import make_ecg

from miocard.imaging import process_ecg_image
from miocard.pipeline import median_beats, signal_text, signals_to_csv, to_json
from miocard.pipeline.export import BEAT_POINTS, BEAT_WINDOW_S
from miocard.pipeline.state import ECGPipelineState


def _result():
    return process_ecg_image(make_ecg(), "gpt-5.6-sol", 20 * 1024 * 1024)


def test_median_beat_is_one_complex_per_lead_on_a_common_grid() -> None:
    beats = median_beats(_result().pipeline)
    assert beats
    for beat in beats:
        assert beat.values.size == BEAT_POINTS
        assert beat.beats_used >= 3
        assert np.isfinite(beat.values).all()


def test_median_beat_extremes_are_taken_before_resampling() -> None:
    # Point sampling a narrow QRS clips it, so the reported peak has to come
    # from the full-resolution median or every amplitude reads low.
    for beat in median_beats(_result().pipeline):
        assert beat.peak_mv >= float(beat.values.max()) - 1e-6
        assert beat.trough_mv <= float(beat.values.min()) + 1e-6


def test_signal_block_states_the_scale_and_its_provenance() -> None:
    text = signal_text(_result().pipeline)
    assert text
    assert "mV" in text
    assert "not a recording" in text
    # The gain is assumed on these sheets, and saying so is the whole point.
    assert "assumed" in text or "read from the sheet" in text
    start, end = BEAT_WINDOW_S
    assert f"{start:+.2f} s" in text
    assert f"{end:+.2f} s" in text


def test_signal_block_is_withheld_when_millivolts_are_meaningless() -> None:
    state = ECGPipelineState(image_shape=(64, 64, 3))
    state.signals = {"units": "px", "calibrated": False}
    assert signal_text(state) == ""


def test_model_metadata_carries_the_waveform() -> None:
    result = _result()
    metadata = result.metadata_text()
    assert "Digitised signal" in metadata
    for beat in median_beats(result.pipeline):
        assert f"{beat.label} [" in metadata


def test_record_describes_where_each_lead_was_read_from() -> None:
    result = _result()
    payload = json.loads(to_json(result.pipeline))
    assert payload["source_transform"] is not None
    assert len(payload["source_transform"]) == 6
    assert "text_objects" in payload
    for lead in payload["signals"]["leads"]:
        assert lead["band"] is not None
        assert lead["band"]["top"] < lead["band"]["bottom"]
        assert isinstance(lead["r_peak_columns"], list)
        assert len(lead["samples"]) == len(lead["sample_confidence"])
        assert len(lead["samples"]) == len(lead["sample_status"])
        assert set(lead["sample_status"]) <= {
            "measured",
            "interpolated",
            "unreadable",
            "outside",
        }


def test_csv_keeps_sample_provenance_next_to_every_signal() -> None:
    result = _result()
    rows = signals_to_csv(result.pipeline).splitlines()
    header = rows[0].split(",")
    assert "I_confidence" in header
    assert "I_status" in header
    assert len(rows[1].split(",")) == len(header)


def test_record_stays_json_serialisable() -> None:
    # Every value has to survive a round trip, or the file the user receives is
    # not the record we think we exported.
    payload = to_json(_result().pipeline)
    assert json.loads(payload)["schema"] == "miocard.pipeline/1"
