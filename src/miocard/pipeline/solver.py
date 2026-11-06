"""Stage 9 - Path solver.

The pen position is recovered as the minimum-cost path across the cost map: one
row per column, paying the stage 8 unary cost where it goes plus a linear
penalty for how far it moves between neighbouring columns. That is a shortest
path, so the result is the globally best explanation of the whole row rather
than a chain of locally darkest pixels - which is exactly what lets it walk
under a lead label or through a grid line instead of derailing there.

The transition penalty is L1, and the lower envelope of an L1 penalty is a
running minimum, so the whole dynamic program is two prefix scans per column.
That keeps an exact solve inside a couple of numpy passes and lets every lead
row be solved at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from miocard.pipeline.config import PipelineConfig
from miocard.pipeline.state import LeadBand, LeadPath, PipelineStageResult, clamp01

# Cost charged when a path strays a full band height from its own baseline -
# that is, when it reaches the neighbouring lead. It has to outweigh the pull of
# the neighbour's ink, which is worth about -log(P(trace)) ~ 6 cost units, or
# two overlapping leads end up sharing one trace and the other is lost.
#
# Confining the path this tightly costs no amplitude: a vertical QRS stroke
# inks the whole column, and stage 13 reads its height off the ink envelope
# rather than off how far the path climbed.
WALL_WEIGHT = 12.0
ENVELOPE_FRACTION = 0.45
# Above this the classifier is calling a pixel grid rather than pen, and the
# envelope must not walk through it.
GRID_IN_ENVELOPE = 0.5
# How long ink must run horizontally to count as a trace rather than a stroke,
# in millimetres. A QRS upstroke spans a fraction of this.
HORIZONTAL_SPAN_MM = 2.0
# Longest break along a column that is treated as ink flow rather than as the
# end of a stroke. See _closed_ink.
INK_BREAK_BRIDGE_PX = 3
# Confidence in a start that was placed at the end of the row's own printed
# label rather than inferred from ink appearing.
ANCHORED_START_CONFIDENCE = 0.85


@dataclass(frozen=True)
class SolverReport:
    paths: tuple[LeadPath, ...]
    probability: float
    statistics: dict[str, float]


def _lower_envelope(previous: np.ndarray, ramp: np.ndarray) -> np.ndarray:
    """min over y' of previous[y'] + |ramp[y] - ramp[y']|, for every y at once.

    ``ramp`` must be non-decreasing in y. Rewriting the forward half as
    ``ramp[y] + min_{y'<=y}(previous[y'] - ramp[y'])`` turns it into a prefix
    minimum, and the backward half is the mirror image of the same scan - so the
    whole transition is two linear passes regardless of how the penalty is
    shaped along the column.
    """
    forward = ramp + np.minimum.accumulate(previous - ramp, axis=-1)
    backward = -ramp + np.minimum.accumulate((previous + ramp)[..., ::-1], axis=-1)[..., ::-1]
    return np.minimum(forward, backward)


def _travel_ramp(
    ink: np.ndarray,
    continuity: float,
    gap_penalty: float,
) -> np.ndarray:
    """Cost coordinate along each column: distance moved, plus emptiness crossed.

    A pen can only be somewhere it left ink. Moving from one row to another
    within a column is free of suspicion where the column is inked all the way -
    that is a genuine vertical stroke through a QRS - but crossing blank paper to
    reach a trace on the other side is exactly the jump that steals a
    neighbouring lead, and it is charged per empty pixel traversed.

    Both terms grow monotonically down the column, so their sum is a monotone
    coordinate ``u(y)`` and the transition cost between any two rows is simply
    ``|u(y2) - u(y1)|``. That keeps the exact prefix-minimum solve intact while
    making the penalty depend on what lies between the two rows.
    """
    rows = ink.shape[-2]
    distance = continuity * np.arange(rows, dtype=np.float32)
    # Cumulative count of empty pixels above each row, per column.
    emptiness = gap_penalty * np.cumsum(1.0 - ink, axis=-2, dtype=np.float32)
    return distance[None, :, None] + emptiness


def _band_windows(
    bands: tuple[LeadBand, ...],
    height: int,
    margin_fraction: float,
) -> tuple[np.ndarray, int]:
    """Top row of each lead's search window, plus the shared window height."""
    heights = [band.height for band in bands]
    margin = int(round(margin_fraction * float(np.median(heights))))
    window = int(np.median(heights)) + 2 * margin
    window = int(np.clip(window, 8, height))
    tops = []
    for band in bands:
        centre = band.baseline
        top = int(round(centre - window / 2))
        tops.append(int(np.clip(top, 0, max(0, height - window))))
    return np.asarray(tops, dtype=np.int32), window


def _own_stroke(
    ink: np.ndarray,
    horizontal: np.ndarray,
    band: tuple[int, int],
    baseline: int,
) -> np.ndarray:
    """Rows reachable from this lead's own trace along a stroke, per column.

    A pen that draws a tall complex leaves its own strip, and the wall is there
    to stop a *different* thing: adopting the neighbour's trace. The two are
    distinguishable on the paper. What the pen leaves behind on its way out is a
    near-vertical stroke continuous with its own trace; what the neighbour leaves
    is a horizontally continuous line.

    The anchor is found per column as the ink nearest the baseline *inside the
    strip*, not the baseline row itself: a baseline that has wandered a few
    pixels leaves its nominal row blank, and anchoring there finds nothing at
    exactly the columns that matter - measured on examples/photo_293, where at
    the crossing column the pen is at 860 and the nominal baseline 870 is bare.
    """
    window, width = ink.shape
    rows = np.arange(window, dtype=np.int32)[:, None]
    inside = (rows >= band[0]) & (rows < band[1])
    far = window * 2
    distance = np.where(ink & inside, np.abs(rows - baseline), far)
    anchor = np.argmin(distance, axis=0)
    columns = np.arange(width)
    found = distance[anchor, columns] < far
    gaps = np.cumsum(~ink, axis=0, dtype=np.int32)
    return (gaps == gaps[anchor, columns][None, :]) & ink & ~horizontal & found[None, :]


def _stacked_cost(
    unary: np.ndarray,
    bands: tuple[LeadBand, ...],
    tops: np.ndarray,
    window: int,
) -> np.ndarray:
    """Cost slices for every lead, stacked into one (bands, window, width) array."""
    stacked = np.empty((len(bands), window, unary.shape[1]), dtype=np.float32)
    rows = np.arange(window, dtype=np.float32)
    for index, band in enumerate(bands):
        top = int(tops[index])
        stacked[index] = unary[top : top + window]
        # Soft wall: free inside the lead's own strip, quadratic outside it, and
        # normalised so straying a full band height from the baseline - the
        # distance to the neighbouring lead - costs exactly WALL_WEIGHT.
        half = max(band.height * 0.5, 1.0)
        offset = np.abs(rows + top - band.baseline) - half
        wall = np.clip(offset, 0.0, None) ** 2 / (half * half)
        stacked[index] += (WALL_WEIGHT * wall).astype(np.float32)[:, None]
    return stacked


def _solve_stack(cost: np.ndarray, ramp: np.ndarray) -> np.ndarray:
    """Minimum-cost row per column for every stacked lead."""
    count, _window, width = cost.shape
    accumulated = np.empty_like(cost)
    accumulated[:, :, 0] = cost[:, :, 0]
    for x in range(1, width):
        envelope = _lower_envelope(accumulated[:, :, x - 1], ramp[:, :, x])
        accumulated[:, :, x] = cost[:, :, x] + envelope

    leads = np.arange(count)
    path = np.empty((count, width), dtype=np.int32)
    path[:, width - 1] = np.argmin(accumulated[:, :, width - 1], axis=1)
    for x in range(width - 2, -1, -1):
        # Backtrack against the same metric the forward pass used.
        arrival = ramp[leads, path[:, x + 1], x + 1][:, None]
        step = np.abs(ramp[:, :, x + 1] - arrival)
        path[:, x] = np.argmin(accumulated[:, :, x] + step, axis=1)
    return path


def _run_extent(mask: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extent of the contiguous ink run the path sits in, per column.

    A vertical QRS stroke covers many rows in a single column, and its extremes
    carry the amplitude that a one-row-per-column path cannot represent.
    """
    count, window, width = mask.shape
    gaps = np.cumsum(~mask, axis=1)
    at_path = np.take_along_axis(gaps, path[:, None, :], axis=1)
    same_run = (gaps == at_path) & mask
    rows = np.arange(window, dtype=np.int32)[None, :, None]
    high = np.where(same_run, rows, window)
    low = np.where(same_run, rows, -1)
    top = high.min(axis=1)
    bottom = low.max(axis=1)
    top = np.where(top >= window, path, top)
    bottom = np.where(bottom < 0, path, bottom)
    return top.astype(np.int32), bottom.astype(np.int32)


def _closed_ink(mask: np.ndarray) -> np.ndarray:
    """Bridge one- and two-pixel breaks along a column before measuring a run.

    The pen moves fastest at the tip of a spike and lays down least ink there,
    so the last few pixels of a QRS come out faint and broken - measured on
    examples/photo_293, where P(trace) at a tip alternates 0.16, 0.09, 0.15,
    0.38 either side of the 0.135 cut. The run then stops short and the peak is
    missing from the reconstruction, which is what a reader notices first.

    Closing repairs that without lengthening anything: it can only join ink to
    ink across a gap narrower than the kernel, never extend a run past its real
    end. The size is the largest that leaves the amplitudes alone - at five
    pixels the runs start reaching a neighbouring trace and the millivolts
    measured against the recorder's own print go from 0.03 mV out to 0.21.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, INK_BREAK_BRIDGE_PX))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def _sheet_run_extent(mask: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extent of the ink run through each lead's row, measured on the whole sheet.

    The windowed version of this stops at the edge of the band's solve window,
    which is fine for a QRS drawn as a vertical stroke - the stroke lives inside
    the window. It is wrong for a stroke the pen drew *fast and slanted*:
    measured on examples/photo_296 the pen covers about 130 px of paper in five
    columns, so the stroke leaves the window entirely and the envelope stops a
    few pixels above the baseline. The peak is then missing from the
    reconstruction, and the digitised complex is a fraction of the printed one.

    Following the run over the full height fixes that. It is safe because the
    path itself is still band-bound - this only asks how far the ink the path is
    standing on actually reaches - and because :func:`_split_shared_runs` still
    refuses to claim ink from beyond the neighbouring pen.
    """
    height, width = mask.shape
    count = rows.shape[0]
    # int32 and argmax rather than int64 and np.where: this runs once per lead
    # over the whole sheet, and the obvious formulation allocates several
    # full-size 64-bit temporaries per lead - enough to exhaust memory on a
    # large upload.
    gaps = np.cumsum(~mask, axis=0, dtype=np.int32)
    columns = np.arange(width)
    top = np.empty((count, width), dtype=np.int32)
    bottom = np.empty((count, width), dtype=np.int32)
    for index in range(count):
        seed = np.clip(rows[index], 0, height - 1)
        same_run = (gaps == gaps[seed, columns][None, :]) & mask
        present = same_run.any(axis=0)
        first = np.argmax(same_run, axis=0).astype(np.int32)
        last = (height - 1 - np.argmax(same_run[::-1], axis=0)).astype(np.int32)
        # A seed on blank paper belongs to no run; keep the path row itself.
        top[index] = np.where(present, first, seed)
        bottom[index] = np.where(present, last, seed)
    return top, bottom


def _split_shared_runs(
    paths: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    own_top: np.ndarray,
    own_bottom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stop a lead's ink run where the neighbour's own stroke begins.

    Where two leads are drawn close together and both are mid-complex, their
    strokes touch and the run in that column is one unbroken mass of ink from
    one baseline to the next. Measured on examples/photo_244: the column is
    solidly dark for 85 px across a 73 px row pitch, so there is no gap to find
    and nothing *in that column alone* says where one pen stopped and the other
    began.

    What does say it is the neighbour's own stroke - the ink continuous with the
    neighbour's trace and not part of any horizontal line. That is a structure
    on the paper, so it stays put whether or not the neighbour's path chose to
    climb it. Using the neighbour's path row instead reads the sheet wrong
    whenever the path stayed on its baseline through its own complex, which is
    the normal case: the solver has no reason to climb a stroke it is already
    standing at the foot of. Measured on examples/photo_293, column 291, where
    V5 claimed ink 47 px into V6's upstroke because V6's path had not moved.

    The rule stays symmetric, and a lead's own row always remains inside its
    run, so a lead whose neighbour has wandered across it keeps a valid envelope.
    """
    if paths.shape[0] < 2:
        return top, bottom
    top = top.copy()
    bottom = bottom.copy()
    # The lead above ends at the bottom of its own stroke; below that the ink
    # belongs to whoever is underneath.
    top[1:] = np.maximum(top[1:], np.minimum(own_bottom[:-1] + 1.0, paths[1:]))
    bottom[:-1] = np.minimum(bottom[:-1], np.maximum(own_top[1:] - 1.0, paths[:-1]))
    return top, bottom


def _active_span(
    evidence: np.ndarray,
    threshold: float,
    run_px: int,
    floor: int = 0,
) -> tuple[int, int, float, float]:
    """Where the printed line starts and ends, and how sharply.

    A single column above threshold is noise; a printed lead begins where the
    evidence *stays* up for a couple of millimetres. The confidence compares the
    evidence just inside the span against the paper just outside it, so a line
    that starts abruptly on blank margin scores near one and a line that fades
    in from nothing scores low.
    """
    width = int(evidence.size)
    if width == 0:
        return 0, 0, 0.0, 0.0
    run = int(np.clip(run_px, 1, max(1, width // 4)))
    strong = (evidence > threshold).astype(np.float32)
    # Ink to the left of the row's printed name belongs to the margin, not to
    # the recording.
    strong[: int(np.clip(floor, 0, width))] = 0.0
    sustained = np.convolve(strong, np.ones(run, dtype=np.float32), mode="valid")
    held = np.flatnonzero(sustained >= run)
    if held.size == 0:
        active = np.flatnonzero(strong)
        if active.size == 0:
            return 0, width, 0.0, 0.0
        return int(active[0]), int(active[-1]) + 1, 0.15, 0.15

    # One isolated dark stroke near the margin can itself be ``run`` columns
    # wide and used to win here.  On a curled sheet that made lead I begin on a
    # header/calibration fragment, followed by almost 200 columns of bare
    # paper, while the real recording started later.  Require the *starts* of
    # qualifying windows to remain consecutive for another ``run`` columns.
    # This is equivalent to seeing roughly twice the minimum ink duration, but
    # it does not bridge a long blank interval and therefore cannot turn a
    # lone printed character into the beginning of a lead.
    breaks = np.flatnonzero(np.diff(held) > 1) + 1
    groups = np.split(held, breaks)
    stable = [group for group in groups if group.size >= run]
    start = int(stable[0][0] if stable else held[0])
    end = int(held[-1]) + run

    def sharpness(inside: np.ndarray, outside: np.ndarray) -> float:
        if inside.size == 0:
            return 0.0
        within = float(inside.mean())
        beyond = float(outside.mean()) if outside.size else 0.0
        return clamp01((within - beyond) / max(within, 1e-6))

    start_probability = sharpness(
        evidence[start : start + run], evidence[max(0, start - run) : start]
    )
    end_probability = sharpness(evidence[end - run : end], evidence[end : end + run])
    return start, end, start_probability, end_probability


def solve_paths(
    unary: np.ndarray,
    trace_probability: np.ndarray,
    bands: tuple[LeadBand, ...],
    config: PipelineConfig,
    px_per_mm: float = 0.0,
    margin_edges: dict[int, int] | None = None,
    grid_probability: np.ndarray | None = None,
) -> SolverReport:
    height, width = unary.shape
    if not bands:
        bands = (
            LeadBand(
                index=0,
                top=0,
                bottom=height,
                baseline=height / 2.0,
                ink=float(trace_probability.mean()),
                label="row 1",
            ),
        )

    # Wide sheets are solved on a decimated column grid and interpolated back;
    # stage 13 refines the result against the full-resolution cost map.
    decimation = max(1, int(np.ceil(width / max(config.max_solver_width, 64))))
    solve_width = width // decimation
    if decimation > 1:
        solve_cost = cv2.resize(unary, (solve_width, height), interpolation=cv2.INTER_AREA)
        solve_trace = cv2.resize(
            trace_probability, (solve_width, height), interpolation=cv2.INTER_AREA
        )
        solve_grid = (
            cv2.resize(grid_probability, (solve_width, height), interpolation=cv2.INTER_AREA)
            if grid_probability is not None
            else None
        )
    else:
        solve_cost, solve_trace = unary, trace_probability
        solve_grid = grid_probability

    pen = solve_trace > (config.gap_evidence_threshold * ENVELOPE_FRACTION)
    if solve_grid is not None:
        # The millimetre grid is ink too, and at this threshold a shadowed patch
        # of it passes. Left in, it bridges the blank paper between a stroke tip
        # and the next trace, and the run - which is what the amplitude is read
        # from - grows by up to 92 px (measured on examples/photo_296; 7.5-9.7 %
        # of the mask is grid on every sheet). The path already pays a grid
        # penalty through the cost map; the envelope had no such guard.
        pen &= solve_grid < GRID_IN_ENVELOPE
    sheet_ink = _closed_ink(pen)
    span = max(9, int(round(HORIZONTAL_SPAN_MM * px_per_mm))) if px_per_mm > 1 else 9
    horizontal = (
        cv2.morphologyEx(
            sheet_ink.astype(np.uint8),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (span, 1)),
        )
        > 0
    )

    tops, window = _band_windows(bands, height, config.band_margin_fraction)
    stacked = _stacked_cost(solve_cost, bands, tops, window)
    trace_stack = np.stack([solve_trace[int(top) : int(top) + window] for top in tops], axis=0)
    ink_mask = trace_stack > (config.gap_evidence_threshold * ENVELOPE_FRACTION)

    ramp = _travel_ramp(
        ink_mask.astype(np.float32),
        float(config.cost_weights.continuity),
        float(config.cost_weights.gap_crossing),
    )
    rows_solved = _solve_stack(stacked, ramp)
    absolute_rows = rows_solved + tops[:, None]
    top_env, bottom_env = _sheet_run_extent(sheet_ink, absolute_rows)

    own_top = np.empty_like(absolute_rows, dtype=np.float32)
    own_bottom = np.empty_like(absolute_rows, dtype=np.float32)
    for index, band in enumerate(bands):
        top = int(tops[index])
        own = _own_stroke(
            sheet_ink[top : top + window],
            horizontal[top : top + window],
            (band.top - top, band.bottom - top),
            int(round(band.baseline)) - top,
        )
        present = own.any(axis=0)
        first = np.argmax(own, axis=0) + top
        last = window - 1 - np.argmax(own[::-1], axis=0) + top
        own_top[index] = np.where(present, first, absolute_rows[index])
        own_bottom[index] = np.where(present, last, absolute_rows[index])

    absolute_top, absolute_bottom = _split_shared_runs(
        absolute_rows.astype(np.float32),
        top_env.astype(np.float32),
        bottom_env.astype(np.float32),
        own_top,
        own_bottom,
    )

    columns = np.arange(rows_solved.shape[1])
    paths: list[LeadPath] = []
    costs: list[float] = []
    for index, band in enumerate(bands):
        offset = float(tops[index])
        y_solved = rows_solved[index].astype(np.float32) + offset
        evidence = trace_stack[index][rows_solved[index], columns].astype(np.float32)
        top_line = absolute_top[index]
        bottom_line = absolute_bottom[index]
        if decimation > 1:
            target = np.linspace(0, y_solved.size - 1, width, dtype=np.float32)
            source = np.arange(y_solved.size, dtype=np.float32)
            y_full = np.interp(target, source, y_solved).astype(np.float32)
            evidence = np.interp(target, source, evidence).astype(np.float32)
            top_line = np.interp(target, source, top_line).astype(np.float32)
            bottom_line = np.interp(target, source, bottom_line).astype(np.float32)
        else:
            y_full = y_solved

        run_px = max(3, int(round(2.0 * px_per_mm))) if px_per_mm > 1 else 5
        # The recording begins after the row's own name and calibration step, so
        # that is where the search for the start of the line begins too.
        floor = int((margin_edges or {}).get(index, 0))
        start, end, start_probability, end_probability = _active_span(
            evidence, config.gap_evidence_threshold * 0.5, run_px, floor
        )
        if floor and start <= floor + run_px:
            # The start was not inferred from ink fading in - it was placed
            # where this row's own printed name ends, which is a stronger
            # statement about where the recording begins than any contrast is.
            start_probability = max(start_probability, ANCHORED_START_CONFIDENCE)
        mean_cost = float(
            unary[np.clip(np.round(y_full).astype(np.int32), 0, height - 1), np.arange(width)][
                start:end
            ].mean()
            if end > start
            else float(unary.mean())
        )
        costs.append(mean_cost)
        paths.append(
            LeadPath(
                index=index,
                label=band.label,
                band=band,
                y=y_full,
                evidence=evidence,
                top_envelope=top_line,
                bottom_envelope=bottom_line,
                x_start=start,
                x_end=end,
                mean_cost=mean_cost,
                start_probability=start_probability,
                end_probability=end_probability,
            )
        )

    coverage = float(np.mean([(path.evidence > config.gap_evidence_threshold).mean() for path in paths]))
    span = float(np.mean([(path.x_end - path.x_start) / max(width, 1) for path in paths]))
    starts = [path.start_probability for path in paths]
    ends = [path.end_probability for path in paths]
    statistics = {
        "mean_cost": float(np.mean(costs)) if costs else 0.0,
        "evidence_coverage": coverage,
        "horizontal_span": span,
        "line_start_probability": float(np.mean(starts)) if starts else 0.0,
        "line_end_probability": float(np.mean(ends)) if ends else 0.0,
        "decimation": float(decimation),
        "window_px": float(window),
    }
    edges = float(np.mean(starts + ends)) if starts else 0.0
    probability = clamp01(
        0.05 + 0.50 * coverage + 0.25 * clamp01((span - 0.5) / 0.45) + 0.20 * edges
    )
    return SolverReport(paths=tuple(paths), probability=probability, statistics=statistics)


def solver_stage(report: SolverReport) -> PipelineStageResult:
    return PipelineStageResult(
        "Path solver",
        report.probability,
        f"{len(report.paths)} lead traces reconstructed by minimum-cost path "
        f"over {report.statistics['horizontal_span'] * 100:.0f}% of the sheet width.",
        evidence=(
            f"evidence_coverage={report.statistics['evidence_coverage']:.3f}",
            f"mean_path_cost={report.statistics['mean_cost']:.2f}",
            f"line_start_probability={report.statistics['line_start_probability']:.2f}",
            f"line_end_probability={report.statistics['line_end_probability']:.2f}",
            f"search_window_px={report.statistics['window_px']:.0f}",
            f"column_decimation={report.statistics['decimation']:.0f}",
        ),
    )
