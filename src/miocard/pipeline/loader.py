"""Stage 1 - Image loader.

Decodes the upload, honours EXIF orientation and rejects payloads that cannot
carry a readable ECG sheet. Nothing here is heuristic: the stage either produces
a usable array or raises, so its confidence is one by construction.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from miocard.pipeline.state import PipelineStageResult

MAX_DECODED_PIXELS = 40_000_000
MIN_EDGE_PX = 200
MAX_ASPECT_RATIO = 5.0

# Several of the phone photographs in examples/ end in a single row of pure
# black - a decoder artefact of an odd image height, not something that was
# photographed. It survives the margin crop, because it is certainly not white,
# and then the path solver adopts it as the cheapest possible trace: measured on
# examples/, the bottom lead of two of the five sheets was reconstructed along
# that row instead of along its printed waveform. A line of real photographed
# dark - a table, a shadow, the edge of the paper - always carries texture, so
# uniformity is what separates the artefact from content.
DEAD_BORDER_MAX_MEAN = 24.0
DEAD_BORDER_MAX_STD = 12.0
DEAD_BORDER_MAX_LINES = 4


class ImageProcessingError(ValueError):
    """The upload cannot be turned into an ECG sheet worth analysing."""


@dataclass(frozen=True)
class LoadedImage:
    image: np.ndarray
    source_width: int
    source_height: int
    byte_size: int
    notes: tuple[str, ...] = field(default_factory=tuple)
    # Where ``image`` sits inside the decoded frame, once dead border lines have
    # been dropped. Everything downstream measures in ``image`` coordinates, so
    # this is what puts a result back on the file the user sent.
    origin: tuple[int, int] = (0, 0)

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]


def decode_image(raw: bytes) -> np.ndarray:
    """Decode to BGR, applying the EXIF rotation the camera recorded."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > MAX_DECODED_PIXELS:
                raise ImageProcessingError("Image has too many pixels to decode safely.")
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            return cv2.cvtColor(np.asarray(normalized), cv2.COLOR_RGB2BGR)
    except ImageProcessingError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageProcessingError("Could not read image as a valid file.") from exc


def trim_dead_border(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Drop border lines that are uniformly black, and say where the rest began.

    Only the outermost few lines are considered, and only while every one of
    them is both dark and featureless; anything with texture is treated as part
    of the photograph even if it is very dark.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    def dead(line: np.ndarray) -> bool:
        return float(line.mean()) < DEAD_BORDER_MAX_MEAN and float(line.std()) < DEAD_BORDER_MAX_STD

    top, bottom, left, right = 0, height, 0, width
    for _ in range(DEAD_BORDER_MAX_LINES):
        if bottom - top > MIN_EDGE_PX and dead(gray[top]):
            top += 1
        else:
            break
    for _ in range(DEAD_BORDER_MAX_LINES):
        if bottom - top > MIN_EDGE_PX and dead(gray[bottom - 1]):
            bottom -= 1
        else:
            break
    for _ in range(DEAD_BORDER_MAX_LINES):
        if right - left > MIN_EDGE_PX and dead(gray[top:bottom, left]):
            left += 1
        else:
            break
    for _ in range(DEAD_BORDER_MAX_LINES):
        if right - left > MIN_EDGE_PX and dead(gray[top:bottom, right - 1]):
            right -= 1
        else:
            break
    if (top, left) == (0, 0) and (bottom, right) == (height, width):
        return image, (0, 0)
    return image[top:bottom, left:right], (left, top)


def load_image(raw: bytes, max_bytes: int) -> LoadedImage:
    if not raw:
        raise ImageProcessingError("Empty input image payload.")
    if len(raw) > max_bytes:
        raise ImageProcessingError(f"File too large: maximum {max_bytes // 1024 // 1024} MB.")

    image = decode_image(raw)
    height, width = image.shape[:2]
    if min(width, height) < MIN_EDGE_PX:
        raise ImageProcessingError(
            f"Image too small; minimum {MIN_EDGE_PX} px per side required."
        )
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        raise ImageProcessingError(
            "Image aspect ratio is too extreme; please upload a standard ECG sheet."
        )

    notes: list[str] = []
    if image.std() < 1.0:
        notes.append("uniform_image")
    image, origin = trim_dead_border(image)
    if image.shape[:2] != (height, width):
        notes.append("trimmed_dead_border")
    return LoadedImage(
        image=image,
        source_width=width,
        source_height=height,
        byte_size=len(raw),
        notes=tuple(notes),
        origin=origin,
    )


def loader_stage(loaded: LoadedImage) -> PipelineStageResult:
    return PipelineStageResult(
        "Image loader",
        1.0,
        f"Decoded {loaded.source_width}x{loaded.source_height} px "
        f"from {loaded.byte_size / 1024:.0f} KiB.",
        evidence=(
            f"source={loaded.source_width}x{loaded.source_height}",
            f"megapixels={loaded.source_width * loaded.source_height / 1e6:.1f}",
        ),
    )
