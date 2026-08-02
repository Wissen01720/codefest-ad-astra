"""Image diagnostics for OCR quality and failure modes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat
import pytesseract

from .diagnostic_common import file_sha256, iter_corpus_files


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class OcrStatus(str, Enum):
    """Normalized OCR classification labels."""

    OCR_CORRECTO = "OCR_CORRECTO"
    SIN_TEXTO = "SIN_TEXTO"
    IMAGEN_BLANCA = "IMAGEN_BLANCA"
    IMAGEN_PEQUENA = "IMAGEN_PEQUENA"
    IMAGEN_BORROSA = "IMAGEN_BORROSA"
    IMAGEN_DANIADA = "IMAGEN_DANIADA"
    OCR_POBRE = "OCR_POBRE"


@dataclass(slots=True)
class OcrValidationResult:
    """Detailed OCR quality classification for a single image."""

    path: Path
    status: OcrStatus
    width: int = 0
    height: int = 0
    characters: int = 0
    mean_brightness: float = 0.0
    brightness_stddev: float = 0.0
    edge_stddev: float = 0.0
    reason: str = ""
    sha256: str | None = None
    errors: tuple[str, ...] = ()


@dataclass(slots=True)
class OcrValidationReport:
    """Corpus-level OCR summary."""

    corpus: Path
    total_files: int = 0
    results: list[OcrValidationResult] = field(default_factory=list)
    counts: dict[OcrStatus, int] = field(default_factory=lambda: defaultdict(int))


def _image_metrics(image: Image.Image) -> tuple[int, int, float, float, float]:
    grayscale = image.convert("L")
    brightness = ImageStat.Stat(grayscale)
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    edge_stats = ImageStat.Stat(edges)

    width, height = image.size
    mean_brightness = brightness.mean[0]
    brightness_stddev = brightness.stddev[0]
    edge_stddev = edge_stats.stddev[0]

    return width, height, mean_brightness, brightness_stddev, edge_stddev


def _classify_based_on_metrics(
    width: int,
    height: int,
    mean_brightness: float,
    brightness_stddev: float,
    edge_stddev: float,
    text: str,
) -> tuple[OcrStatus, str]:
    if width * height < 50_000 or min(width, height) < 160:
        return OcrStatus.IMAGEN_PEQUENA, "La imagen es demasiado pequeña para OCR confiable."

    if mean_brightness >= 245 and brightness_stddev < 5:
        return OcrStatus.IMAGEN_BLANCA, "La imagen parece estar vacía o casi blanca."

    if edge_stddev < 4 and not text.strip():
        return OcrStatus.IMAGEN_BORROSA, "La imagen es muy uniforme y no aporta bordes útiles."

    if not text.strip():
        return OcrStatus.SIN_TEXTO, "OCR no produjo texto utilizable."

    if len(text.strip()) < 20:
        return OcrStatus.OCR_POBRE, "OCR produjo muy poco texto."

    return OcrStatus.OCR_CORRECTO, "OCR produjo texto útil."


def validate_image(path: Path) -> OcrValidationResult:
    """Classify an image according to OCR quality signals."""

    sha = file_sha256(path)

    try:
        with Image.open(path) as image:
            width, height, mean_brightness, brightness_stddev, edge_stddev = _image_metrics(image)

            try:
                raw_text = pytesseract.image_to_string(image, lang="spa+eng+por")
            except Exception as exc:
                return OcrValidationResult(
                    path=path,
                    status=OcrStatus.IMAGEN_DANIADA,
                    width=width,
                    height=height,
                    mean_brightness=mean_brightness,
                    brightness_stddev=brightness_stddev,
                    edge_stddev=edge_stddev,
                    reason="No fue posible ejecutar OCR.",
                    sha256=sha,
                    errors=(str(exc),),
                )

            status, reason = _classify_based_on_metrics(
                width=width,
                height=height,
                mean_brightness=mean_brightness,
                brightness_stddev=brightness_stddev,
                edge_stddev=edge_stddev,
                text=raw_text,
            )

            return OcrValidationResult(
                path=path,
                status=status,
                width=width,
                height=height,
                characters=len(raw_text.strip()),
                mean_brightness=mean_brightness,
                brightness_stddev=brightness_stddev,
                edge_stddev=edge_stddev,
                reason=reason,
                sha256=sha,
            )

    except Exception as exc:
        return OcrValidationResult(
            path=path,
            status=OcrStatus.IMAGEN_DANIADA,
            reason=str(exc),
            sha256=sha,
            errors=(str(exc),),
        )


def validate_image_corpus(corpus: Path) -> OcrValidationReport:
    """Validate every image found in a corpus."""

    files = iter_corpus_files(corpus, IMAGE_SUFFIXES)
    report = OcrValidationReport(corpus=corpus, total_files=len(files))

    for path in files:
        result = validate_image(path)
        report.results.append(result)
        report.counts[result.status] = report.counts.get(result.status, 0) + 1

    return report