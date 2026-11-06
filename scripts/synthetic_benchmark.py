#!/usr/bin/env python

"""Score the pipeline against sheets whose answer is known exactly.

``scripts/benchmark.py`` measures the five real photographs, where the only
external truth is what the recorder happened to print - one heart rate and four
pairs of amplitudes in total. That is enough to catch a gross error and not
enough to tell a real improvement from luck; twice during development a change
moved that score the right way and was wrong.

Here every sheet comes with its own signal in millivolts, its row positions and
its rate, so the same questions get answered on as many sheets as you care to
generate, and the answers are exact.

    python scripts/synthetic_benchmark.py --count 12
    python scripts/synthetic_benchmark.py --count 40 --hard --out runs/synth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from miocard.imaging import process_ecg_image  # noqa: E402
from miocard.pipeline import map_source_points, median_beats  # noqa: E402
from miocard.synthetic import SheetRecipe, SyntheticSheet, make_sheet  # noqa: E402

# A row is matched when a detected band sits within this fraction of the pitch
# of the printed baseline. Half a pitch is the widest that cannot reach the
# neighbouring lead.
ROW_TOLERANCE = 0.5


def recipes(count: int, hard: bool) -> list[SheetRecipe]:
    """A spread of sheets, from cleanly photographed to barely legible."""
    rng = np.random.default_rng(20260811)
    out: list[SheetRecipe] = []
    for index in range(count):
        severity = (index / max(count - 1, 1)) if hard else 0.35
        out.append(
            SheetRecipe(
                seed=index,
                px_per_mm=float(rng.uniform(4.4, 5.4)),
                heart_rate_bpm=float(rng.uniform(48, 110)),
                row_pitch_px=float(rng.uniform(68, 84)),
                rotation_deg=float(rng.uniform(-1.0, 1.0) * (0.5 + 2.0 * severity)),
                perspective=float(rng.uniform(0.0, 0.03) * (0.3 + severity)),
                bow_px=float(rng.uniform(0.0, 26.0) * severity),
                glare=float(rng.uniform(0.0, 0.18) * severity),
                blur_px=float(0.4 + 1.2 * severity),
                illumination=float(0.05 + 0.25 * severity),
                noise_mv=float(0.01 + 0.05 * severity),
                wander_mv=float(0.02 + 0.14 * severity),
                jpeg_quality=int(round(92 - 30 * severity)),
                ink=int(round(30 + 40 * severity)),
            )
        )
    return out


def score(sheet: SyntheticSheet, model: str, max_bytes: int) -> dict[str, object]:
    result = process_ecg_image(sheet.encode(), model, max_bytes)
    state = result.pipeline
    if state is None or not state.bands:
        return {"seed": sheet.recipe.seed, "error": "no layout"}

    pitch = float(state.layout.get("row_pitch_px", 0.0) or 0.0)
    matrix = state.source_transform
    baselines = np.array([band.baseline for band in state.bands])

    matched = 0
    named = 0
    offsets: list[float] = []
    for lead in sheet.leads:
        middle = sheet.image.shape[1] / 2.0
        truth = map_source_points(matrix, np.array([[middle, float(lead.baseline_y[int(middle)])]]))
        row = float(truth[0, 1])
        index = int(np.argmin(np.abs(baselines - row)))
        if pitch > 1 and abs(baselines[index] - row) <= ROW_TOLERANCE * pitch:
            matched += 1
            offsets.append(float(baselines[index] - row))
            if state.bands[index].label == lead.label:
                named += 1

    # Amplitude: the R peak of the averaged complex, against the millivolts the
    # generator drew. Only leads that were both found and named can be compared,
    # since an unnamed row cannot be attributed to a lead.
    truth_r = {lead.label: lead.r_peak_mv for lead in sheet.leads}
    errors: list[float] = []
    for beat in median_beats(state):
        expected = truth_r.get(beat.label)
        if expected is None:
            continue
        measured = float(np.max(beat.values)) if expected > 0 else float(np.min(beat.values))
        errors.append(abs(measured - expected))

    rate = state.beats.heart_rate_bpm if state.beats is not None else None
    return {
        "seed": sheet.recipe.seed,
        "rows": len(state.bands),
        "leads_matched": matched,
        "leads_named": named,
        "row_offset_px": round(float(np.median(np.abs(offsets))), 1) if offsets else None,
        "row_spread_px": round(float(np.std(offsets)), 1) if len(offsets) > 1 else None,
        "amplitude_error_mv": round(float(np.mean(errors)), 3) if errors else None,
        "amplitude_leads": len(errors),
        "rate_bpm": round(float(rate), 1) if rate is not None else None,
        "rate_error_bpm": (
            round(float(rate) - sheet.recipe.heart_rate_bpm, 1) if rate is not None else None
        ),
        "px_per_mm": round(float(state.grid.small_cell_px), 2) if state.grid else None,
        "px_per_mm_error": (
            round(float(state.grid.small_cell_px) - sheet.recipe.px_per_mm, 3)
            if state.grid
            else None
        ),
        "quality_grade": state.quality_grade,
    }


COLUMNS = (
    ("seed", "sheet", 6, "d"),
    ("rows", "rows", 5, "d"),
    ("leads_matched", "found", 6, "d"),
    ("leads_named", "named", 6, "d"),
    ("row_offset_px", "row px", 7, ".1f"),
    ("px_per_mm_error", "px/mm", 7, "+.2f"),
    ("amplitude_error_mv", "mV err", 7, ".3f"),
    ("rate_error_bpm", "bpm err", 8, "+.1f"),
    ("quality_grade", "grade", 9, "s"),
)


def report(rows: list[dict]) -> None:
    header = "  ".join(name.rjust(width) for _, name, width, _ in COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        if "error" in row:
            print(f"{row['seed']:>6}  {row['error']}")
            continue
        cells = []
        for key, _, width, spec in COLUMNS:
            value = row.get(key)
            text = "-" if value is None else (str(value) if spec == "s" else format(value, spec))
            cells.append(text[:width].rjust(width))
        print("  ".join(cells))

    scored = [row for row in rows if "error" not in row]
    if not scored:
        return
    total = 12 * len(scored)
    amplitude = [row["amplitude_error_mv"] for row in scored if row["amplitude_error_mv"]]
    rate = [row["rate_error_bpm"] for row in scored if row["rate_error_bpm"] is not None]
    scale = [row["px_per_mm_error"] for row in scored if row["px_per_mm_error"] is not None]
    print()
    print(f"sheets              {len(scored)} of {len(rows)}")
    print(f"leads located       {sum(r['leads_matched'] for r in scored)}/{total}")
    print(f"leads named right   {sum(r['leads_named'] for r in scored)}/{total}")
    if scale:
        print(f"grid pitch error    {np.mean(np.abs(scale)):.3f} px/mm")
    if amplitude:
        print(f"R amplitude error   {np.mean(amplitude):.3f} mV over {len(amplitude)} sheet(s)")
    if rate:
        print(f"rate error          {np.mean(np.abs(rate)):.1f} bpm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--hard", action="store_true", help="Sweep from clean to barely legible.")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--max-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = [score(make_sheet(recipe), args.model, args.max_bytes)
            for recipe in recipes(args.count, args.hard)]
    report(rows)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
