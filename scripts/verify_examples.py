#!/usr/bin/env python

"""One verification sheet per example photograph.

The benchmark prints numbers and the overlay draws lines; neither on its own
answers "is this reading any good?". This puts them on one page: what the
pipeline drew over the sheet, what it measured against what the recorder itself
printed, and the median complex per lead - which is exactly what gets sent to
the model, and the one artefact that is otherwise invisible.

A check is green when it agrees with something printed on the paper, amber when
it is merely plausible, and red when it contradicts the print. Grey means the
sheet carries nothing to check against, which on this sample is common and is
not a failure.

    python scripts/verify_examples.py
    python scripts/verify_examples.py --output-dir tmp-verify/today
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (str(SRC_DIR), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark import measure  # noqa: E402

from miocard.imaging import process_ecg_image  # noqa: E402
from miocard.overlay import render_digitisation  # noqa: E402
from miocard.pipeline import median_beats  # noqa: E402

DEFAULT_GROUND_TRUTH = ROOT_DIR / "examples" / "ground_truth.json"
DEFAULT_IMAGE_DIR = ROOT_DIR / "examples"

SHEET_WIDTH = 1500
CARD_HEIGHT = 140
CARD_COLUMNS = 6

OK = (60, 170, 60)
WARN = (0, 165, 235)
BAD = (60, 60, 220)
NEUTRAL = (150, 150, 150)
INK = (35, 35, 35)


def _text(
    panel: np.ndarray,
    value: str,
    origin: tuple[int, int],
    scale: float = 0.46,
    color: tuple[int, int, int] = INK,
    thickness: int = 1,
) -> None:
    cv2.putText(
        panel, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def checks(row: dict) -> list[tuple[str, str, tuple[int, int, int]]]:
    """Every check the sheet supports, with the verdict spelled out.

    Written as (what was asked, what came back, verdict) so a reader does not
    have to know the metric to see whether it passed.
    """
    result: list[tuple[str, str, tuple[int, int, int]]] = []

    found, expected = row["leads_found"], row["leads_expected"]
    result.append(
        (
            "printed leads located",
            f"{found}/{expected}  (rows detected: {row['rows']})",
            OK if found == expected else BAD,
        )
    )
    named = row["leads_named"]
    result.append(
        (
            "leads named correctly",
            f"{named}/{expected}" + (" - rows unnamed" if named == 0 else ""),
            OK if named == expected else (BAD if named == 0 else WARN),
        )
    )
    residual = row["lead_residual_px"]
    pitch = row["row_pitch_px"] or 1.0
    result.append(
        (
            "worst row offset vs print",
            f"{residual:.1f} px ({residual / pitch:.2f} of a row pitch)",
            OK if residual < 0.3 * pitch else WARN,
        )
    )

    printed_rate = row["rate_printed_bpm"]
    if printed_rate:
        error = row["rate_error_bpm"]
        result.append(
            (
                "rate vs printed rate",
                f"{row['rate_bpm']:.1f} vs {printed_rate} bpm  ({error:+.1f})",
                OK if abs(error) <= 3 else (WARN if abs(error) <= 8 else BAD),
            )
        )
    else:
        result.append(("rate vs printed rate", "not printed on this sheet", NEUTRAL))

    if row["sv1_printed_mv"] is not None and row["sv1_mv"] is not None:
        error = row["amplitude_error_mv"]
        result.append(
            (
                "S in V1 / R in V5 vs print",
                f"{row['sv1_mv']:.2f}/{row['rv5_mv']:.2f} vs "
                f"{row['sv1_printed_mv']:.2f}/{row['rv5_printed_mv']:.2f} mV "
                f"(mean error {error:.3f})",
                OK if error <= 0.10 else (WARN if error <= 0.25 else BAD),
            )
        )
    elif row["sv1_printed_mv"] is not None:
        result.append(
            ("S in V1 / R in V5 vs print", "printed, but those leads were not named", BAD)
        )
    else:
        result.append(("S in V1 / R in V5 vs print", "not printed on this sheet", NEUTRAL))

    gain = row["gain_printed_mm_mv"]
    chest = row["chest_gain_printed_mm_mv"]
    if chest:
        result.append(
            (
                "gain",
                f"sheet prints {gain:.0f}/{chest:.0f} mm/mV; "
                f"pipeline uses one gain of {row['gain_mm_mv']:.0f} - chest leads read double",
                BAD,
            )
        )
    else:
        agrees = gain is not None and abs(row["gain_mm_mv"] - float(gain)) < 0.1
        result.append(
            (
                "gain",
                f"{row['gain_mm_mv']:.0f} mm/mV assumed, {gain:.0f} printed"
                if gain
                else f"{row['gain_mm_mv']:.0f} mm/mV assumed, nothing printed",
                OK if agrees else WARN,
            )
        )

    result.append(
        (
            "pen stands on ink",
            f"{row['ink_hit_rate']:.3f} overall, {row['worst_lead_ink']:.3f} on "
            f"{row['worst_lead']} (worst lead)",
            OK if row["worst_lead_ink"] >= 0.9 else WARN,
        )
    )
    result.append(
        (
            "path along the picture border",
            f"{row['edge_rate']:.3f}",
            OK if row["edge_rate"] < 0.01 else BAD,
        )
    )
    result.append(
        (
            "neighbouring leads collide",
            f"{row['collision_rate']:.3f} of columns",
            OK if row["collision_rate"] < 0.02 else WARN,
        )
    )
    return result


def render_checks(row: dict, width: int) -> np.ndarray:
    entries = checks(row)
    panel = np.full((34 + 24 * len(entries), width, 3), 250, dtype=np.uint8)
    _text(panel, "Checks against what is printed on the paper", (14, 22), 0.52, INK, 1)
    y = 46
    for name, value, color in entries:
        cv2.circle(panel, (22, y - 4), 5, color, -1)
        _text(panel, name, (38, y), 0.45, INK)
        _text(panel, value, (330, y), 0.45, color if color != NEUTRAL else (120, 120, 120))
        y += 24
    return panel


def render_median_beats(state, width: int) -> np.ndarray:
    """The median complex per lead - the numbers the model is actually given."""
    beats = median_beats(state)
    if not beats:
        panel = np.full((60, width, 3), 250, dtype=np.uint8)
        _text(panel, "No median beat: the sheet was not calibrated.", (14, 36), 0.5, BAD)
        return panel

    # A lead that produced no median beat has no card, and a missing card is
    # invisible unless you count them. The model is told the same thing.
    drawn = {beat.label for beat in beats}
    silent = [signal.label for signal in state.lead_signals if signal.label not in drawn]
    rows = int(np.ceil(len(beats) / CARD_COLUMNS))
    card_w = width // CARD_COLUMNS
    footer = 24 if silent else 0
    panel = np.full((30 + rows * CARD_HEIGHT + footer, width, 3), 250, dtype=np.uint8)
    if silent:
        _text(
            panel,
            f"No median beat, and therefore not sent to the model: {', '.join(silent)} "
            "(fewer than three complexes fitted inside the traced span).",
            (14, 30 + rows * CARD_HEIGHT + 16),
            0.46,
            WARN,
        )
    _text(
        panel,
        "Median complex per lead, mV, -0.25 s to +0.45 s around R - this is what goes to the model. "
        "Each card is scaled to its own lead; the peak and trough in mV are printed on it.",
        (14, 20),
        0.5,
        INK,
    )
    for position, beat in enumerate(beats):
        # Scaling each card to its own lead rather than to the record: on a
        # sheet where the chest leads are five times the limb leads, a shared
        # scale flattens half the panel into a straight line and the morphology
        # that this page exists to show disappears. The amplitudes are printed.
        span = float(np.max(np.abs(beat.values))) or 1.0
        row, column = divmod(position, CARD_COLUMNS)
        x0, y0 = column * card_w, 30 + row * CARD_HEIGHT
        card = panel[y0 : y0 + CARD_HEIGHT, x0 : x0 + card_w]
        cv2.rectangle(card, (0, 0), (card_w - 2, CARD_HEIGHT - 2), (225, 225, 225), 1)
        centre = CARD_HEIGHT // 2 + 8
        cv2.line(card, (6, centre), (card_w - 8, centre), (232, 232, 232), 1)
        ys = centre - beat.values / span * (CARD_HEIGHT * 0.32)
        xs = np.linspace(8, card_w - 10, beat.values.size)
        cv2.polylines(
            card,
            [np.stack((xs, ys), axis=1).astype(np.int32)],
            False,
            (150, 60, 40),
            1,
            cv2.LINE_AA,
        )
        _text(card, beat.label, (8, 16), 0.44, INK)
        # An unnamed row is called "row 11", not "V5", so the amplitudes have to
        # start after whatever the label turned out to be.
        (label_width, _), _ = cv2.getTextSize(beat.label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        _text(
            card,
            f"{beat.peak_mv:+.2f}/{beat.trough_mv:+.2f}  n={beat.beats_used}",
            (14 + label_width, 16),
            0.36,
            (110, 110, 110),
        )
    return panel


def render_sheet(name: str, row: dict, overlay: np.ndarray, state) -> np.ndarray:
    scale = SHEET_WIDTH / overlay.shape[1]
    picture = cv2.resize(
        overlay,
        (SHEET_WIDTH, max(1, int(overlay.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    title = np.full((42, SHEET_WIDTH, 3), 28, dtype=np.uint8)
    _text(title, name, (14, 28), 0.62, (240, 240, 240))
    _text(
        title,
        f"grade {row['quality_grade']}   |   {row['px_per_mm']:.2f} px/mm "
        f"({row['grid_confidence']})   |   pitch {row['row_pitch_px']:.0f} px",
        (330, 28),
        0.5,
        (170, 220, 170),
    )
    return np.vstack(
        (
            title,
            render_checks(row, SHEET_WIDTH),
            picture,
            render_median_beats(state, SHEET_WIDTH),
        )
    )


def render_index(rows: list[dict]) -> np.ndarray:
    header = ("sheet", "leads", "named", "resid", "on-ink", "bpm vs print", "mV err", "grade")
    widths = (200, 90, 90, 90, 90, 170, 100, 120)
    panel = np.full((60 + 26 * len(rows), SHEET_WIDTH, 3), 250, dtype=np.uint8)
    _text(panel, "Verification summary", (14, 26), 0.6, INK)
    x = 14
    for name, width in zip(header, widths, strict=True):
        _text(panel, name, (x, 50), 0.44, (110, 110, 110))
        x += width
    y = 76
    for row in rows:
        cells = (
            row["file"].split("@")[0],
            f"{row['leads_found']}/{row['leads_expected']}",
            f"{row['leads_named']}/{row['leads_expected']}",
            f"{row['lead_residual_px']:.1f}",
            f"{row['ink_hit_rate']:.3f}",
            (
                f"{row['rate_bpm']:.0f} vs {row['rate_printed_bpm']}"
                if row["rate_printed_bpm"]
                else "-"
            ),
            f"{row['amplitude_error_mv']:.3f}" if row["amplitude_error_mv"] is not None else "-",
            row["quality_grade"],
        )
        x = 14
        for cell, width in zip(cells, widths, strict=True):
            _text(panel, cell, (x, y), 0.44, INK)
            x += width
        y += 26
    return panel


def build(ground_truth: Path, image_dir: Path, output_dir: Path, model: str) -> list[Path]:
    truth = json.loads(ground_truth.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    rows: list[dict] = []
    for record in truth["records"]:
        source = image_dir / record["file"]
        if not source.exists():
            continue
        result = process_ecg_image(source.read_bytes(), model, 20 * 1024 * 1024)
        processed = cv2.imdecode(np.frombuffer(result.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        row = measure(record, result, processed)
        if "error" in row:
            continue
        rows.append(row)

        name = record["file"].split("@")[0]
        sheet = render_sheet(
            name, row, render_digitisation(processed, result.pipeline), result.pipeline
        )
        path = output_dir / f"{name}_verify.png"
        if not cv2.imwrite(str(path), sheet):
            raise RuntimeError(f"Failed to write {path}")
        written.append(path)

    if rows:
        path = output_dir / "000_summary.png"
        if not cv2.imwrite(str(path), render_index(rows)):
            raise RuntimeError(f"Failed to write {path}")
        written.insert(0, path)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model", default="gpt-5.6-sol")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT_DIR / "tmp-verify" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    written = build(
        Path(args.ground_truth), Path(args.image_dir), output_dir, args.model
    )
    if not written:
        print("No example photographs matched; nothing was written.")
        return 2
    for path in written:
        print(f"saved: {path}")
    print(f"\nSaved in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
