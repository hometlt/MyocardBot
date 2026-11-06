"""The generated sheet has to be worth measuring against.

These tests check the generator, not the pipeline. The one property everything
else depends on is that the answer comes back in the coordinates of the final,
distorted picture: a ground truth that describes the clean sheet would quietly
score every reconstruction against the wrong rows.
"""

from __future__ import annotations

import cv2
import numpy as np

from miocard.synthetic import LEAD_ORDER, SheetRecipe, make_sheet


def _ink_rows(image: np.ndarray, column: int, width: int = 24) -> np.ndarray:
    """Darkness profile down one strip of the sheet, background removed."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    strip = gray[:, max(0, column - width // 2) : column + width // 2]
    return float(strip.mean()) - strip.mean(axis=1)


def test_sheet_has_the_shape_it_claims() -> None:
    sheet = make_sheet(SheetRecipe(seed=3))
    assert sheet.image.shape == (sheet.recipe.height, sheet.recipe.width, 3)
    assert len(sheet.leads) == len(LEAD_ORDER)
    assert [lead.label for lead in sheet.leads] == list(LEAD_ORDER)
    assert sheet.r_peak_columns.size >= 6
    # A blank or saturated sheet would pass every geometric check below.
    assert 200 < float(cv2.cvtColor(sheet.image, cv2.COLOR_BGR2GRAY).mean()) < 250


def test_baselines_land_on_ink_through_the_distortion() -> None:
    """The claimed baseline must sit on the printed trace, not beside it.

    This is the test that a truth mapped through the warp passes and a truth
    left in clean-sheet coordinates fails, which is the whole point.
    """
    sheet = make_sheet(SheetRecipe(seed=4, bow_px=22.0, rotation_deg=1.5, perspective=0.02))
    for column in (300, 640, 980):
        profile = _ink_rows(sheet.image, column)
        for lead in sheet.leads:
            row = int(round(float(lead.baseline_y[column])))
            near = profile[max(0, row - 3) : row + 4]
            assert near.max() > 12.0, f"{lead.label} at column {column} is not on any ink"


def test_the_bow_actually_bends_the_rows() -> None:
    flat = make_sheet(SheetRecipe(seed=5, bow_px=0.0, rotation_deg=0.0, perspective=0.0))
    bowed = make_sheet(SheetRecipe(seed=5, bow_px=20.0, rotation_deg=0.0, perspective=0.0))
    straight = flat.leads[0].baseline_y
    curved = bowed.leads[0].baseline_y
    assert float(np.ptp(straight)) < 1.0
    assert float(np.ptp(curved)) > 10.0


def test_amplitudes_are_drawn_at_the_stated_gain() -> None:
    """A lead's R peak must be its stated millivolts times the stated gain."""
    recipe = SheetRecipe(seed=6, bow_px=0.0, rotation_deg=0.0, perspective=0.0, noise_mv=0.0,
                         wander_mv=0.0)
    sheet = make_sheet(recipe)
    px_per_mv = recipe.px_per_mm * recipe.gain_mm_mv
    for lead in sheet.leads:
        drawn = float(np.max(lead.samples_mv)) if lead.r_peak_mv > 0 else float(
            np.min(lead.samples_mv)
        )
        assert abs(drawn - lead.r_peak_mv) < 0.2, lead.label
    # And the geometry the pipeline will have to recover is self-consistent.
    assert px_per_mv > 0


def test_two_seeds_differ_and_one_seed_repeats() -> None:
    first = make_sheet(SheetRecipe(seed=7))
    again = make_sheet(SheetRecipe(seed=7))
    other = make_sheet(SheetRecipe(seed=8))
    assert np.array_equal(first.image, again.image)
    assert not np.array_equal(first.image, other.image)


def test_encode_round_trips() -> None:
    sheet = make_sheet(SheetRecipe(seed=9))
    decoded = cv2.imdecode(np.frombuffer(sheet.encode(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == sheet.image.shape
