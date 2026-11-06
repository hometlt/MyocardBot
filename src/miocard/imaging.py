from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from miocard.models import LocalizedFinding


class ImageProcessingError(ValueError):
    pass


MAX_DECODED_PIXELS = 40_000_000


@dataclass(frozen=True)
class ImageProfile:
    max_edge: int
    visual_token_budget: int


# Millimetre grid periods worth searching for on a processed sheet. Below ~4 px the
# small square is indistinguishable from JPEG noise; above ~60 px it would imply a
# crop of a few centimetres rather than a whole recording.
MIN_GRID_PERIOD_PX = 4
MAX_GRID_PERIOD_PX = 60
# A single axis peak is accepted generously; the real guard against a spurious pitch is
# that both axes must independently land on the same period (see GridMetrics.confidence).
GRID_PEAK_THRESHOLD = 0.07
LARGE_CELL_SMALL_SQUARES = 5
# How much larger than the delivered sheet the retained source may be. A rhythm view
# spans about half the width, so doubling the edge is enough for a crop to fill the
# model's visual budget without holding a full-resolution photograph in memory.
VIEW_SOURCE_SCALE = 2
# Above this multiple of the local median R-R the gap is no longer sinus variability.
# It stays well clear of the ~1.3x that follows an ordinary compensatory pause.
PAUSE_RATIO = 1.6


@dataclass(frozen=True)
class GridMetrics:
    """Millimetre grid pitch measured locally, in pixels of the processed image."""

    small_cell_px: float
    strength: float
    axis_ratio: float

    @property
    def large_cell_px(self) -> float:
        return self.small_cell_px * LARGE_CELL_SMALL_SQUARES

    @property
    def confidence(self) -> str:
        # A square millimetre grid has the same pitch on both axes. A diverging ratio
        # means residual perspective or an accidental peak, and reporting nothing is
        # safer than handing the model a wrong time scale.
        if self.strength >= 0.22 and 0.97 <= self.axis_ratio <= 1.03:
            return "high"
        if self.strength >= GRID_PEAK_THRESHOLD and 0.93 <= self.axis_ratio <= 1.07:
            return "medium"
        return "low"


@dataclass(frozen=True)
class ProcessedImage:
    data: bytes
    media_type: str
    width: int
    height: int
    estimated_visual_tokens: int
    blur_score: float
    brightness: float
    contrast: float
    transformations: tuple[str, ...] = field(default_factory=tuple)
    grid: GridMetrics | None = None
    # A higher-resolution copy of the same sheet, kept only so that an enlarged fragment
    # is genuinely enlarged instead of a crop of pixels the downscale already discarded.
    # None when the source was not larger than the delivered image.
    source_data: bytes | None = None
    source_width: int = 0
    profile: ImageProfile | None = None

    def _grid_text(self) -> str:
        if self.grid is None or self.grid.confidence == "low":
            return (
                "- шаг миллиметровой сетки измерить не удалось; "
                "оценивай интервалы по видимой сетке\n"
            )
        small = self.grid.small_cell_px
        large = self.grid.large_cell_px
        # At the standard 25 mm/s one millimetre is 0.04 s, so the measured pitch turns
        # the picture into a time scale without the model having to count squares.
        seconds_per_row = self.width / (25.0 * small)
        return (
            f"- шаг миллиметровой сетки: малая клетка ≈ {small:.1f} px, "
            f"большая (5 мм) ≈ {large:.1f} px, надежность измерения: {self.grid.confidence}\n"
            f"- при скорости 25 мм/с полная ширина изображения ≈ {seconds_per_row:.1f} с, "
            f"а 300 / (число больших клеток между R–R) дает ЧСС\n"
        )

    def metadata_text(self) -> str:
        changes = ", ".join(self.transformations) if self.transformations else "нет"
        return (
            f"- размер после обработки: {self.width}x{self.height} px\n"
            f"- расчетный визуальный бюджет: ~{self.estimated_visual_tokens} токенов\n"
            f"- резкость (variance of Laplacian): {self.blur_score:.1f}\n"
            f"- яркость/контраст (0–255): {self.brightness:.1f}/{self.contrast:.1f}\n"
            + self._grid_text()
            + f"- преобразования: {changes}\n"
            "Эти показатели описывают фотографию, а не пациента и не параметры ЭКГ. "
            "Шаг сетки измерен программой по изображению: используй его для проверки "
            "интервалов, но сверяй со скоростью, напечатанной на записи."
        )


@dataclass(frozen=True)
class ECGImageView:
    """An enlarged full-height time window mapped to the processed full sheet."""

    name: str
    image: ProcessedImage
    left: int
    right: int


@dataclass(frozen=True)
class RhythmTimingHint:
    """An abnormal R-R interval measured from synchronized dark QRS columns."""

    x: int
    rr_px: float
    median_rr_px: float
    coupling_ratio: float
    confidence: str
    kind: str = "premature"


def profile_for_model(model: str) -> ImageProfile:
    if model == "claude-opus-4-6":
        return ImageProfile(max_edge=1568, visual_token_budget=1568)
    return ImageProfile(max_edge=2576, visual_token_budget=4784)


def estimate_visual_tokens(width: int, height: int) -> int:
    return math.ceil(width / 28) * math.ceil(height / 28)


def _autocorrelation(profile: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation of a 1-D profile, detrended to ignore lighting."""
    values = profile.astype(np.float64)
    window = MAX_GRID_PERIOD_PX * 2 + 1
    # Harmonics are needed to tell a real grid from noise, so look twice as far as the
    # longest accepted pitch.
    max_lag = MAX_GRID_PERIOD_PX * 2 + 2
    if values.size < max_lag * 2:
        return np.zeros(0)
    trend = cv2.blur(
        values.reshape(1, -1).astype(np.float32),
        (window, 1),
        borderType=cv2.BORDER_REPLICATE,
    ).ravel()
    centered = values - trend
    energy = float(np.dot(centered, centered))
    if energy <= 0:
        return np.zeros(0)
    padded = int(2 ** math.ceil(math.log2(values.size * 2)))
    spectrum = np.fft.rfft(centered, n=padded)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=padded)
    return correlation[: max_lag + 1] / energy


def _peak_prominence(correlation: np.ndarray, lag: int) -> float:
    """Height of a peak above the decaying autocorrelation envelope around it.

    The envelope falls steeply near zero lag, so an absolute threshold would miss a
    perfectly regular grid whose peaks ride low on that slope.
    """
    if lag <= 0 or lag + 1 >= correlation.size:
        return 0.0
    value = correlation[lag]
    if value < correlation[lag - 1] or value < correlation[lag + 1]:
        return 0.0
    half = MIN_GRID_PERIOD_PX // 2 + 1
    left = correlation[max(0, lag - half) : lag]
    right = correlation[lag + 1 : lag + 1 + half]
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(value - (float(left.min()) + float(right.min())) / 2)


def _smallest_periodic_peak(correlation: np.ndarray) -> tuple[float, float]:
    """Return the shortest repeating period and its prominence.

    A grid repeats at the millimetre pitch and at every multiple of it, so the shortest
    peak that also has a second-harmonic partner is the small square rather than the
    5 mm block or a stray waveform rhythm.
    """
    last_lag = min(MAX_GRID_PERIOD_PX, correlation.size - 2)
    for lag in range(MIN_GRID_PERIOD_PX, last_lag + 1):
        prominence = _peak_prominence(correlation, lag)
        if prominence < GRID_PEAK_THRESHOLD:
            continue
        harmonic = max(
            _peak_prominence(correlation, offset)
            for offset in (2 * lag - 1, 2 * lag, 2 * lag + 1)
        )
        if harmonic < GRID_PEAK_THRESHOLD / 2:
            continue
        # Parabolic interpolation recovers sub-pixel pitch, which compounds into whole
        # squares over the width of a row.
        left, right = correlation[lag - 1], correlation[lag + 1]
        denominator = left - 2 * correlation[lag] + right
        offset = 0.5 * (left - right) / denominator if denominator else 0.0
        return float(lag) + float(np.clip(offset, -0.5, 0.5)), prominence
    return 0.0, 0.0


def measure_grid(image: np.ndarray) -> GridMetrics | None:
    """Measure the millimetre grid pitch so intervals do not rest on eyeballed squares."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # The grid is a low-contrast red mesh; inverting makes its lines the signal peaks.
    inverted = 255.0 - gray
    vertical_period, vertical_strength = _smallest_periodic_peak(
        _autocorrelation(inverted.mean(axis=0))
    )
    horizontal_period, horizontal_strength = _smallest_periodic_peak(
        _autocorrelation(inverted.mean(axis=1))
    )
    if vertical_period <= 0 or horizontal_period <= 0:
        return None
    return GridMetrics(
        small_cell_px=(vertical_period + horizontal_period) / 2,
        strength=min(vertical_strength, horizontal_strength),
        axis_ratio=vertical_period / horizontal_period,
    )


def _decode_image(raw: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > MAX_DECODED_PIXELS:
                raise ImageProcessingError("Изображение имеет слишком большое разрешение.")
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            return cv2.cvtColor(np.asarray(normalized), cv2.COLOR_RGB2BGR)
    except ImageProcessingError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageProcessingError("Файл не распознан как корректное изображение.") from exc


def _deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detection_scale = min(1.0, 1200 / max(gray.shape))
    if detection_scale < 1:
        gray = cv2.resize(gray, None, fx=detection_scale, fy=detection_scale)
    edges = cv2.Canny(gray, 60, 160)
    min_length = max(100, int(gray.shape[1] * 0.3))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,
        threshold=100,
        minLineLength=min_length,
        maxLineGap=30,
    )
    if lines is None:
        return image, 0.0
    angles: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -7 <= angle <= 7:
            angles.append(angle)
    if len(angles) < 3:
        return image, 0.0
    angle = float(np.median(angles))
    if abs(angle) < 0.25:
        return image, 0.0
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def _main_content_row_span(gray: np.ndarray) -> tuple[int, int]:
    """Find the main horizontally extended recording band, ignoring page-edge shadows."""
    height, width = gray.shape
    if height < 40 or width < 40:
        return 0, height

    # ECG grids and traces extend over a meaningful fraction of every row in the sheet.
    # Dust and page-edge shadows form short isolated runs and must not decide the crop.
    row_activity = (gray < 240).mean(axis=1)
    active = (row_activity >= 0.02).astype(np.uint8).reshape(-1, 1)
    close_size = max(3, round(height * 0.025))
    closed = cv2.morphologyEx(
        active,
        cv2.MORPH_CLOSE,
        np.ones((close_size, 1), dtype=np.uint8),
    ).ravel().astype(bool)
    boundaries = np.flatnonzero(np.diff(np.r_[False, closed, False]))
    runs = [
        (int(start), int(stop))
        for start, stop in zip(boundaries[::2], boundaries[1::2], strict=True)
        if stop - start >= max(20, round(height * 0.06))
    ]
    if not runs:
        return 0, height
    return max(runs, key=lambda item: item[1] - item[0])


def _crop_empty_margins(image: np.ndarray) -> tuple[np.ndarray, bool]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    if not np.any(gray < 248):
        raise ImageProcessingError("Изображение выглядит пустым.")

    content_y1, content_y2 = _main_content_row_span(gray)
    content_height = content_y2 - content_y1
    pad_y = max(8, round(content_height * 0.05))
    y1 = max(0, content_y1 - pad_y)
    y2 = min(height, content_y2 + pad_y)

    # Find horizontal bounds only inside the selected recording band. This prevents a
    # single speckle in a white margin from preserving the whole photograph.
    column_activity = (gray[y1:y2] < 240).mean(axis=0)
    active_columns = np.flatnonzero(column_activity >= 0.01)
    if active_columns.size:
        content_x1 = int(active_columns.min())
        content_x2 = int(active_columns.max()) + 1
        pad_x = max(8, round((content_x2 - content_x1) * 0.015))
        x1 = max(0, content_x1 - pad_x)
        x2 = min(width, content_x2 + pad_x)
    else:
        x1, x2 = 0, width

    cropped = image[y1:y2, x1:x2]
    changed = cropped.shape[:2] != image.shape[:2]
    return cropped, changed


def _enhance_for_ecg(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(12, 12))
    enhanced_light = clahe.apply(light)
    enhanced = cv2.merge((enhanced_light, a, b))
    bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=0.8)
    return cv2.addWeighted(bgr, 1.12, blurred, -0.12, 0)


def _resize_to_profile(image: np.ndarray, profile: ImageProfile) -> tuple[np.ndarray, bool]:
    height, width = image.shape[:2]
    edge_scale = min(1.0, profile.max_edge / max(width, height))
    token_scale = min(
        1.0,
        math.sqrt(profile.visual_token_budget / max(1, estimate_visual_tokens(width, height))),
    )
    scale = min(edge_scale, token_scale)
    if scale >= 0.995:
        return image, False
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # Ceilings in the token formula can leave the first estimate a few tokens over budget.
    while estimate_visual_tokens(new_width, new_height) > profile.visual_token_budget:
        new_width -= 1
        new_height = max(1, int(height * (new_width / width)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, True


def process_ecg_image(raw: bytes, model: str, max_bytes: int) -> ProcessedImage:
    if not raw:
        raise ImageProcessingError("Получен пустой файл.")
    if len(raw) > max_bytes:
        raise ImageProcessingError(f"Файл слишком большой: максимум {max_bytes // 1024 // 1024} МБ.")

    image = _decode_image(raw)
    height, width = image.shape[:2]
    if min(width, height) < 200:
        raise ImageProcessingError("Изображение слишком маленькое; минимум 200 px по каждой стороне.")
    if max(width, height) / min(width, height) > 5:
        raise ImageProcessingError("Изображение слишком узкое; пришлите весь лист ЭКГ целиком.")

    transformations: list[str] = []
    image, angle = _deskew(image)
    if angle:
        transformations.append(f"выравнивание {angle:+.1f}°")
    image, cropped = _crop_empty_margins(image)
    if cropped:
        transformations.append("удаление пустых полей")

    gray_before = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray_before, cv2.CV_64F).var())
    brightness = float(gray_before.mean())
    contrast = float(gray_before.std())
    if contrast < 8:
        raise ImageProcessingError("На изображении недостаточно контраста для разбора ЭКГ.")

    # Preserve waveform morphology when the source already has good contrast. Non-linear
    # enhancement can make the grid/noise look like small deflections and is only useful
    # for genuinely faded or poorly exposed photographs.
    if contrast < 30 or brightness < 100 or brightness > 250:
        image = _enhance_for_ecg(image)
        transformations.append("мягкое локальное усиление контраста")
    else:
        transformations.append("контраст исходника сохранен")

    profile = profile_for_model(model)
    # Each rhythm view covers roughly half of the sheet, so a copy kept at twice the
    # delivered edge lets every fragment fill the visual budget on its own.
    source = _resize_to_profile(
        image,
        ImageProfile(
            max_edge=profile.max_edge * VIEW_SOURCE_SCALE,
            visual_token_budget=profile.visual_token_budget * VIEW_SOURCE_SCALE**2,
        ),
    )[0]
    image, resized = _resize_to_profile(image, profile)
    if resized:
        transformations.append("уменьшение под визуальный токен-бюджет")

    height, width = image.shape[:2]
    success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not success:
        raise ImageProcessingError("Не удалось закодировать обработанное изображение.")

    source_data: bytes | None = None
    source_width = width
    if source.shape[:2] != image.shape[:2]:
        source_success, encoded_source = cv2.imencode(
            ".png", source, [cv2.IMWRITE_PNG_COMPRESSION, 6]
        )
        if source_success:
            source_data = encoded_source.tobytes()
            source_width = source.shape[1]
            transformations.append("сохранен исходник для увеличенных фрагментов")

    # Measured on the final image so the pitch matches the pixels the model actually sees.
    grid = measure_grid(image)
    if grid is not None and grid.confidence != "low":
        transformations.append(f"измерена сетка {grid.small_cell_px:.1f} px/мм")
    return ProcessedImage(
        data=encoded.tobytes(),
        media_type="image/png",
        width=width,
        height=height,
        estimated_visual_tokens=estimate_visual_tokens(width, height),
        blur_score=blur_score,
        brightness=brightness,
        contrast=contrast,
        transformations=tuple(transformations),
        grid=grid,
        source_data=source_data,
        source_width=source_width,
        profile=profile,
    )


def create_rhythm_views(image: ProcessedImage) -> tuple[ECGImageView, ...]:
    """Create overlapping full-height views for independent beat inventories.

    A narrow premature complex is often distinguished by R-R timing rather than QRS
    morphology. Giving each half its own vision pass prevents the salient first outlier
    on a dense 12-lead sheet from consuming the model's rhythm attention. Twenty percent
    overlap keeps an event near the midpoint away from a crop edge. A third terminal
    view gives the last four R-R intervals an independent pass, because narrow late
    ectopy was the measured failure mode on the full sheet. Downstream code merges
    duplicate sightings in the shared intervals.

    The crop is taken from the retained higher-resolution source and only then fitted to
    the visual budget, so a fragment really does carry more detail per millimetre than
    the full sheet. Cropping the delivered image instead would isolate attention but
    hand the model the very pixels the downscale had already thrown away.
    """
    decoded = cv2.imdecode(
        np.frombuffer(image.source_data or image.data, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if decoded is None:
        raise ImageProcessingError("Не удалось открыть обработанное изображение для фрагментов.")

    height, width = decoded.shape[:2]
    # Without a stored profile (a directly constructed image) the full sheet's own size
    # is the budget, which leaves the crop at its native resolution.
    profile = image.profile or ImageProfile(
        max_edge=max(image.width, image.height),
        visual_token_budget=max(1, image.estimated_visual_tokens),
    )
    # The grid was measured on the delivered image, so it has to travel through both
    # rescalings to describe the pixels of the fragment.
    source_scale = width / max(1, image.width)
    regions = (
        ("left", 0, 600),
        ("right", 480, 1000),
        ("terminal", 600, 1000),
    )
    views: list[ECGImageView] = []
    for name, left, right in regions:
        x1 = round(left * width / 1000)
        x2 = round(right * width / 1000)
        crop, _ = _resize_to_profile(decoded[:, x1:x2], profile)
        success, encoded = cv2.imencode(".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if not success:
            raise ImageProcessingError("Не удалось закодировать увеличенный фрагмент ЭКГ.")

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_height, crop_width = crop.shape[:2]
        view_scale = crop_width / max(1, x2 - x1)
        magnification = source_scale * view_scale
        label = f"увеличенный фрагмент {left}–{right}/1000 полного листа"
        if magnification >= 1.05:
            label += f", детализация ×{magnification:.1f}"
        view_image = ProcessedImage(
            data=encoded.tobytes(),
            media_type="image/png",
            width=crop_width,
            height=crop_height,
            estimated_visual_tokens=estimate_visual_tokens(crop_width, crop_height),
            blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            brightness=float(gray.mean()),
            contrast=float(gray.std()),
            transformations=image.transformations + (label,),
            grid=None
            if image.grid is None
            else GridMetrics(
                small_cell_px=image.grid.small_cell_px * magnification,
                strength=image.grid.strength,
                axis_ratio=image.grid.axis_ratio,
            ),
            profile=profile,
        )
        views.append(ECGImageView(name=name, image=view_image, left=left, right=right))
    return tuple(views)


def detect_rhythm_timing_hints(
    image: ProcessedImage,
) -> tuple[RhythmTimingHint, ...]:
    """Measure abruptly short and abruptly long R-R intervals, without diagnosing.

    The lower precordial rows on a simultaneous 12-lead print contain tall, vertically
    aligned QRS strokes and relatively little header text. Summing only dark pixels over
    that band produces a one-dimensional synchronization profile. A short interval is a
    candidate premature complex; a long one is a candidate pause, dropped beat or
    blocked atrial complex, which no other pass in this pipeline looks for. This
    deliberately supplies timing candidates only; the vision model still validates
    morphology, multi-lead concordance, and artifact alternatives.
    """
    gray = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ImageProcessingError("Не удалось открыть ЭКГ для измерения R–R.")
    height, width = gray.shape
    content_y1, content_y2 = _main_content_row_span(gray)
    content_height = content_y2 - content_y1
    y1 = content_y1 + round(content_height * 0.35)
    y2 = content_y1 + round(content_height * 0.98)
    band = gray[y1:y2]
    if band.size == 0:
        return ()

    dark_columns = (band < 105).astype(np.float32).sum(axis=0)
    score = cv2.GaussianBlur(dark_columns.reshape(1, -1), (0, 0), 1.2).ravel()
    x1, x2 = round(width * 0.06), round(width * 0.96)
    useful = score[x1:x2]
    if useful.size < 100 or float(useful.max()) <= 0:
        return ()
    threshold = max(float(np.percentile(useful, 90)), float(useful.max()) * 0.24)
    local = np.where(
        (score[1:-1] >= score[:-2])
        & (score[1:-1] > score[2:])
        & (score[1:-1] >= threshold)
    )[0] + 1

    min_separation = max(18, round(width * 0.025))
    peaks: list[int] = []
    for x in local[np.argsort(score[local])[::-1]]:
        candidate = int(x)
        if not x1 <= candidate <= x2:
            continue
        if all(abs(candidate - existing) >= min_separation for existing in peaks):
            peaks.append(candidate)
    peaks.sort()
    if len(peaks) < 6:
        return ()

    intervals = np.diff(np.asarray(peaks, dtype=np.float64))
    median_rr = float(np.median(intervals))
    if median_rr < min_separation * 1.5 or median_rr > width * 0.2:
        return ()

    hints: list[RhythmTimingHint] = []
    for index, rr_px in enumerate(intervals, start=1):
        ratio = float(rr_px / median_rr)
        if ratio < 0.84:
            # An isolated weak column can mimic a short interval. Require the candidate
            # QRS to carry at least the robust energy threshold used for the sequence.
            peak_x = peaks[index]
            if score[peak_x] < threshold:
                continue
            kind = "premature"
            confidence = "high" if ratio <= 0.80 else "medium"
        elif ratio > PAUSE_RATIO:
            # A dropped or blocked beat is a gap rather than a complex, so the candidate
            # points at the middle of the interval instead of at either neighbour.
            peak_x = (peaks[index - 1] + peaks[index]) // 2
            kind = "pause"
            confidence = "high" if ratio >= 1.9 else "medium"
        else:
            continue
        hints.append(
            RhythmTimingHint(
                x=round(peak_x * 1000 / max(1, width - 1)),
                rr_px=float(rr_px),
                median_rr_px=median_rr,
                coupling_ratio=ratio,
                confidence=confidence,
                kind=kind,
            )
        )
    return tuple(hints)


def annotate_findings(
    image: ProcessedImage,
    findings: list[LocalizedFinding],
) -> bytes:
    """Draw numbered boxes using the same normalized coordinate space seen by the model."""
    decoded = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ImageProcessingError("Не удалось открыть изображение для разметки.")
    if not findings:
        return image.data

    height, width = decoded.shape[:2]
    palette = (
        (20, 20, 230),
        (0, 135, 255),
        (220, 90, 20),
    )
    rectangle_thickness = max(2, round(min(width, height) / 300))
    font_scale = max(0.65, min(1.8, min(width, height) / 700))
    text_thickness = max(2, rectangle_thickness)

    # A hollow outline disappears into a dense millimetre grid, so the region is tinted
    # first and the traces are then blended back over the tint.
    highlighted = decoded.copy()
    boxes: list[tuple[int, int, int, int, tuple[int, int, int]]] = []
    for marker, finding in enumerate(findings[:3], start=1):
        box = finding.box
        left = round(box.left * (width - 1) / 1000)
        top = round(box.top * (height - 1) / 1000)
        right = round(box.right * (width - 1) / 1000)
        bottom = round(box.bottom * (height - 1) / 1000)
        color = palette[(marker - 1) % len(palette)]
        boxes.append((left, top, right, bottom, color))
        cv2.rectangle(highlighted, (left, top), (right, bottom), color, -1)
    cv2.addWeighted(highlighted, 0.12, decoded, 0.88, 0, decoded)

    for marker, (left, top, right, bottom, color) in enumerate(boxes, start=1):
        cv2.rectangle(
            decoded,
            (left, top),
            (right, bottom),
            color,
            rectangle_thickness,
            cv2.LINE_AA,
        )

        label = str(marker)
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        pad = max(4, rectangle_thickness * 2)
        badge_width = text_width + pad * 2
        badge_height = text_height + baseline + pad * 2
        badge_left = min(left, max(0, width - 1 - badge_width))
        # Near the top edge there is no room above the box, and a badge clamped to zero
        # used to sit across the frame it labels. Drop it inside the box instead.
        badge_top = top - badge_height
        if badge_top < 0:
            badge_top = min(top + rectangle_thickness, max(0, height - 1 - badge_height))
        badge_right = min(width - 1, badge_left + badge_width)
        badge_bottom = min(height - 1, badge_top + badge_height)
        cv2.rectangle(
            decoded,
            (badge_left, badge_top),
            (badge_right, badge_bottom),
            color,
            -1,
        )
        text_x = badge_left + pad
        text_y = badge_top + pad + text_height
        cv2.putText(
            decoded,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    success, encoded = cv2.imencode(
        ".png",
        decoded,
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )
    if not success:
        raise ImageProcessingError("Не удалось сохранить размеченное изображение.")
    return encoded.tobytes()
