from __future__ import annotations

import numpy as np

from miocard.pipeline.solver import _active_span


def test_active_span_ignores_a_short_prefix_before_a_long_blank_margin() -> None:
    evidence = np.full(600, 0.05, dtype=np.float32)
    evidence[:15] = 0.8
    evidence[246:580] = 0.8

    start, end, start_probability, _end_probability = _active_span(
        evidence,
        threshold=0.15,
        run_px=9,
    )

    assert start == 246
    assert end >= 580
    assert start_probability > 0.8
