"""Stage 11 - Beat detector.

R peaks are found the way they are found in a signal chain rather than in an
image: differentiate, square, integrate over a QRS-width window, then take the
largest responses subject to a refractory period. Working on the reconstructed
deflection instead of on pixels means the detector is insensitive to how dark
the print was.

Every lead is searched, and the lead whose rhythm is most self-consistent
supplies the reported rate. That avoids reading the heart rate off a lead where
the reconstruction happened to be poor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miocard.pipeline.config import PipelineConfig
from miocard.pipeline.series import median_filter, moving_average, robust_scale
from miocard.pipeline.state import BeatSet, PipelineStageResult, clamp01

MIN_PLAUSIBLE_BPM = 20.0
MAX_PLAUSIBLE_BPM = 300.0
MIN_BEATS = 3


@dataclass(frozen=True)
class BeatReport:
    beats: BeatSet | None
    per_lead: dict[int, np.ndarray]
    statistics: dict[str, float]
    probability: float


def detrend(signal: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Remove baseline wander with a median window about one beat wide."""
    window = max(3, int(round(0.60 * sample_rate_hz)))
    return (signal - median_filter(signal, window)).astype(np.float32)


def _integrated_energy(signal: np.ndarray, sample_rate_hz: float, config: PipelineConfig):
    derivative = np.gradient(signal.astype(np.float32))
    window = max(3, int(round(config.integration_seconds * sample_rate_hz)))
    return moving_average(derivative * derivative, window)


def _pick_peaks(energy: np.ndarray, refractory: int, threshold: float) -> np.ndarray:
    candidates = np.flatnonzero(energy >= threshold)
    if candidates.size == 0:
        return np.zeros(0, dtype=np.int32)
    order = candidates[np.argsort(-energy[candidates])]
    chosen: list[int] = []
    for index in order:
        if all(abs(int(index) - other) >= refractory for other in chosen):
            chosen.append(int(index))
    return np.asarray(sorted(chosen), dtype=np.int32)


def _refine(signal: np.ndarray, peaks: np.ndarray, radius: int) -> np.ndarray:
    refined = []
    for peak in peaks:
        low = max(0, int(peak) - radius)
        high = min(signal.size, int(peak) + radius + 1)
        window = np.abs(signal[low:high])
        if window.size == 0:
            refined.append(int(peak))
            continue
        refined.append(low + int(np.argmax(window)))
    return np.asarray(refined, dtype=np.int32)


def detect_r_peaks(
    signal: np.ndarray,
    sample_rate_hz: float,
    config: PipelineConfig,
) -> tuple[np.ndarray, float]:
    """Return the R-peak sample indices and how far they stand above the rest."""
    if signal.size < int(sample_rate_hz) or sample_rate_hz <= 1:
        return np.zeros(0, dtype=np.int32), 0.0
    detrended = detrend(signal, sample_rate_hz)
    energy = _integrated_energy(detrended, sample_rate_hz, config)
    reference = float(np.percentile(energy, 98))
    if reference <= 1e-9:
        return np.zeros(0, dtype=np.int32), 0.0
    threshold = 0.35 * reference
    refractory = max(2, int(round(config.refractory_seconds * sample_rate_hz)))
    peaks = _pick_peaks(energy, refractory, threshold)
    peaks = _refine(detrended, peaks, max(2, int(round(0.05 * sample_rate_hz))))
    if peaks.size == 0:
        return peaks, 0.0
    background = float(np.percentile(energy, 60)) + 1e-9
    prominence = clamp01(float(np.log10(reference / background)) / 2.0)
    return peaks, prominence


def _rhythm(peaks: np.ndarray, sample_rate_hz: float) -> tuple[float, float, np.ndarray]:
    if peaks.size < 2:
        return 0.0, 0.0, np.zeros(0, dtype=np.float32)
    rr = np.diff(peaks).astype(np.float32) / sample_rate_hz
    median_rr = float(np.median(rr))
    if median_rr <= 0:
        return 0.0, 0.0, rr
    heart_rate = 60.0 / median_rr
    dispersion = float(np.percentile(np.abs(rr - median_rr), 75)) / median_rr
    return heart_rate, clamp01(1.0 - dispersion * 4.0), rr


def detect_beats(
    signals: dict[int, np.ndarray],
    labels: dict[int, str],
    sample_rate_hz: float,
    config: PipelineConfig,
) -> BeatReport:
    per_lead: dict[int, np.ndarray] = {}
    scored: list[tuple[float, int, np.ndarray, float, float, np.ndarray]] = []
    for index, signal in signals.items():
        peaks, prominence = detect_r_peaks(signal, sample_rate_hz, config)
        per_lead[index] = peaks
        if peaks.size < MIN_BEATS:
            continue
        heart_rate, regularity, rr = _rhythm(peaks, sample_rate_hz)
        if not MIN_PLAUSIBLE_BPM <= heart_rate <= MAX_PLAUSIBLE_BPM:
            continue
        amplitude = robust_scale(detrend(signal, sample_rate_hz))
        score = 0.55 * regularity + 0.30 * prominence + 0.15 * clamp01(amplitude / 12.0)
        scored.append((score, index, peaks, heart_rate, regularity, rr))

    if not scored:
        return BeatReport(
            beats=None,
            per_lead=per_lead,
            statistics={"leads_with_beats": 0.0, "sample_rate_hz": float(sample_rate_hz)},
            probability=0.08,
        )

    score, index, peaks, heart_rate, regularity, rr = max(scored, key=lambda item: item[0])
    rates = np.asarray([item[3] for item in scored], dtype=np.float32)
    # A couple of leads reading double or half rate - a tall T wave counted as a
    # QRS - would blow up a standard deviation, so agreement is the share of
    # leads near the consensus rather than the spread of all of them.
    consensus = float(np.median(rates))
    agreement = float(np.mean(np.abs(rates - consensus) <= 0.10 * max(consensus, 1e-3)))
    probability = clamp01(
        0.10
        + 0.35 * regularity
        + 0.25 * agreement
        + 0.20 * clamp01(len(scored) / max(len(signals), 1))
        + 0.10 * clamp01(peaks.size / 8.0)
    )
    beats = BeatSet(
        r_peaks=peaks,
        rr_seconds=rr,
        heart_rate_bpm=float(heart_rate),
        regularity=float(regularity),
        probability=probability,
        source_lead=labels.get(index, f"row {index + 1}"),
    )
    statistics = {
        "leads_with_beats": float(len(scored)),
        "rate_agreement": float(agreement),
        "sample_rate_hz": float(sample_rate_hz),
        "median_rate_bpm": float(np.median(rates)),
    }
    return BeatReport(beats=beats, per_lead=per_lead, statistics=statistics, probability=probability)


def beat_stage(report: BeatReport) -> PipelineStageResult:
    if report.beats is None:
        return PipelineStageResult(
            "Beat detector",
            report.probability,
            "No repeating QRS pattern was found in any reconstructed lead.",
            evidence=(f"sample_rate_hz={report.statistics['sample_rate_hz']:.1f}",),
        )
    beats = report.beats
    return PipelineStageResult(
        "Beat detector",
        report.probability,
        f"{beats.count} beats on {beats.source_lead}: "
        f"{beats.heart_rate_bpm:.0f} bpm, regularity {beats.regularity:.2f}.",
        evidence=(
            f"heart_rate_bpm={beats.heart_rate_bpm:.1f}",
            f"beat_count={beats.count}",
            f"rr_regularity={beats.regularity:.3f}",
            f"leads_with_beats={report.statistics['leads_with_beats']:.0f}",
            f"rate_agreement={report.statistics['rate_agreement']:.2f}",
        ),
    )
