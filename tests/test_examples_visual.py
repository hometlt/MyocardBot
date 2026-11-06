"""Visual verification over the real photographs, not synthetic ones.

``test_pipeline_stages_visual`` draws every stage on a generated ECG, which
proves the renderers run but says nothing about a phone photo of a curled sheet.
This module builds the per-example verification sheet - overlay, checks against
what the recorder printed, and the median complex sent to the model - and
asserts it is a picture someone could actually read.

The sheets are written to ``tmp-test-results/<timestamp>_examples/`` so a failure
can be looked at rather than guessed at.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from verify_examples import build, checks, render_median_beats  # noqa: E402

from miocard.imaging import process_ecg_image  # noqa: E402
from miocard.pipeline.state import ECGPipelineState  # noqa: E402

EXAMPLES = ROOT_DIR / "examples"
GROUND_TRUTH = EXAMPLES / "ground_truth.json"


def _available() -> list[str]:
    if not GROUND_TRUTH.exists():
        return []
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    return [r["file"] for r in truth["records"] if (EXAMPLES / r["file"]).exists()]


needs_examples = pytest.mark.skipif(
    not _available(), reason="example photographs are not present in this checkout"
)


@pytest.fixture(scope="module")
def sheets(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT_DIR / "tmp-test-results" / f"{stamp}_examples"
    return build(GROUND_TRUTH, EXAMPLES, output, "gpt-5.6-sol")


@needs_examples
def test_one_sheet_per_example_plus_a_summary(sheets: list[Path]) -> None:
    assert len(sheets) == len(_available()) + 1
    assert sheets[0].name == "000_summary.png"


@needs_examples
def test_sheets_are_readable_pictures(sheets: list[Path]) -> None:
    for path in sheets:
        image = cv2.imread(str(path))
        assert image is not None, path
        assert image.shape[1] == 1500
        # The summary is one table; an example sheet carries the checks, the
        # whole overlay and twelve beat cards, so it cannot be short.
        assert image.shape[0] > (150 if path.name.startswith("000") else 800), path
        # A sheet that came out uniform means a renderer silently produced a
        # blank panel, which no dimension check would catch.
        assert float(image.std()) > 10.0, path


@needs_examples
def test_every_check_reaches_a_verdict() -> None:
    """No check may quietly go missing, and none may be left uncoloured."""
    from benchmark import measure

    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    record = next(r for r in truth["records"] if (EXAMPLES / r["file"]).exists())
    result = process_ecg_image(
        (EXAMPLES / record["file"]).read_bytes(), "gpt-5.6-sol", 20 * 1024 * 1024
    )
    processed = cv2.imdecode(np.frombuffer(result.data, dtype=np.uint8), cv2.IMREAD_COLOR)

    entries = checks(measure(record, result, processed))
    assert len(entries) == 9
    for name, value, color in entries:
        assert name and value
        assert len(color) == 3


def test_median_beat_panel_says_so_when_there_is_nothing_to_draw() -> None:
    # An uncalibrated sheet has no millivolts, and a panel of flat lines would
    # look like a reading rather than the absence of one.
    empty = ECGPipelineState(image_shape=(64, 64, 3))
    panel = render_median_beats(empty, 600)
    assert panel.shape[1] == 600
    assert float(panel.std()) > 1.0
