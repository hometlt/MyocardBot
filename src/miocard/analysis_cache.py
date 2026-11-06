from __future__ import annotations

import os
from pathlib import Path

from miocard.models import ECGReport, RhythmScoutResponse


class AnalysisCache:
    """Small persistent cache that stores reports, never source images."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def get(self, key: str) -> ECGReport | None:
        path = self._directory / f"{key}.json"
        try:
            return ECGReport.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key: str, report: ECGReport) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{key}.json"
        temporary = self._directory / f".{key}.{os.getpid()}.tmp"
        temporary.write_text(report.model_dump_json(), encoding="utf-8")
        temporary.replace(path)

    def get_rhythm_scout(self, key: str) -> RhythmScoutResponse | None:
        path = self._directory / f"{key}.rhythm.json"
        try:
            return RhythmScoutResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put_rhythm_scout(self, key: str, scout: RhythmScoutResponse) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{key}.rhythm.json"
        temporary = self._directory / f".{key}.{os.getpid()}.rhythm.tmp"
        temporary.write_text(scout.model_dump_json(), encoding="utf-8")
        temporary.replace(path)
