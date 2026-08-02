"""PDF classification helpers for corpus diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pdfplumber

from .diagnostic_common import file_sha256, iter_corpus_files, value_kind


PDF_SUFFIXES = {".pdf"}


class PdfStatus(str, Enum):
    """Normalized PDF health categories."""

    PDF_CORRUPTO = "PDF_CORRUPTO"
    PDF_ESCANEADO = "PDF_ESCANEADO"
    PDF_VACIO = "PDF_VACIO"
    PDF_CIFRADO = "PDF_CIFRADO"
    PDF_SIN_OCR = "PDF_SIN_OCR"
    PDF_INCOMPLETO = "PDF_INCOMPLETO"
    PDF_OK = "PDF_OK"
    PDF_ERROR = "PDF_ERROR"


@dataclass(slots=True)
class PdfValidationResult:
    """Detailed result of a PDF validation pass."""

    path: Path
    status: PdfStatus
    pages: int = 0
    pages_with_text: int = 0
    pages_with_images: int = 0
    characters: int = 0
    reason: str = ""
    sha256: str | None = None
    errors: tuple[str, ...] = ()


@dataclass(slots=True)
class PdfValidationReport:
    """Corpus-level PDF validation summary."""

    corpus: Path
    total_files: int = 0
    results: list[PdfValidationResult] = field(default_factory=list)
    counts: dict[PdfStatus, int] = field(default_factory=lambda: defaultdict(int))


def _classify_open_error(message: str) -> PdfStatus:
    lowered = message.lower()

    if "password" in lowered or "encrypt" in lowered:
        return PdfStatus.PDF_CIFRADO

    if any(token in lowered for token in ("root object", "xref", "eof", "malformed", "damaged", "syntax error")):
        return PdfStatus.PDF_CORRUPTO

    return PdfStatus.PDF_ERROR


def _page_has_images(page) -> bool:
    try:
        return bool(getattr(page, "images", []))
    except Exception:
        return False


def validate_pdf(path: Path) -> PdfValidationResult:
    """Classify a single PDF resource."""

    sha = file_sha256(path)

    try:
        with pdfplumber.open(path) as pdf:
            if getattr(pdf, "is_encrypted", False):
                return PdfValidationResult(
                    path=path,
                    status=PdfStatus.PDF_CIFRADO,
                    reason="El documento está cifrado.",
                    sha256=sha,
                )

            pages = len(pdf.pages)
            if pages == 0:
                return PdfValidationResult(
                    path=path,
                    status=PdfStatus.PDF_VACIO,
                    reason="El PDF no contiene páginas.",
                    sha256=sha,
                )

            page_text_lengths: list[int] = []
            pages_with_images = 0
            page_errors: list[str] = []

            for index, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text(x_tolerance=1) or ""
                except Exception as exc:
                    page_errors.append(f"Página {index}: {exc}")
                    text = ""

                page_text_lengths.append(len(text.strip()))

                if _page_has_images(page):
                    pages_with_images += 1

            characters = sum(page_text_lengths)
            pages_with_text = sum(1 for value in page_text_lengths if value > 0)

            if characters == 0:
                if pages_with_images > 0:
                    return PdfValidationResult(
                        path=path,
                        status=PdfStatus.PDF_ESCANEADO,
                        pages=pages,
                        pages_with_images=pages_with_images,
                        reason="No hay capa de texto; el documento parece escaneado.",
                        sha256=sha,
                        errors=tuple(page_errors),
                    )

                return PdfValidationResult(
                    path=path,
                    status=PdfStatus.PDF_VACIO,
                    pages=pages,
                    pages_with_images=pages_with_images,
                    reason="No se extrajo texto y tampoco hay indicios de imágenes.",
                    sha256=sha,
                    errors=tuple(page_errors),
                )

            if page_errors:
                return PdfValidationResult(
                    path=path,
                    status=PdfStatus.PDF_INCOMPLETO,
                    pages=pages,
                    pages_with_text=pages_with_text,
                    pages_with_images=pages_with_images,
                    characters=characters,
                    reason="Se detectaron errores al leer una o más páginas.",
                    sha256=sha,
                    errors=tuple(page_errors),
                )

            if pages_with_text < pages and pages_with_images > 0 and characters < 500:
                return PdfValidationResult(
                    path=path,
                    status=PdfStatus.PDF_SIN_OCR,
                    pages=pages,
                    pages_with_text=pages_with_text,
                    pages_with_images=pages_with_images,
                    characters=characters,
                    reason="Hay imágenes, pero la capa textual es insuficiente para un OCR usable.",
                    sha256=sha,
                )

            if pages_with_text < max(1, pages // 3) and characters < 1000:
                return PdfValidationResult(
                    path=path,
                    status=PdfStatus.PDF_INCOMPLETO,
                    pages=pages,
                    pages_with_text=pages_with_text,
                    pages_with_images=pages_with_images,
                    characters=characters,
                    reason="La extracción textual es parcial o muy escasa.",
                    sha256=sha,
                )

            return PdfValidationResult(
                path=path,
                status=PdfStatus.PDF_OK,
                pages=pages,
                pages_with_text=pages_with_text,
                pages_with_images=pages_with_images,
                characters=characters,
                reason="PDF legible con texto útil.",
                sha256=sha,
            )

    except Exception as exc:
        status = _classify_open_error(str(exc))
        return PdfValidationResult(
            path=path,
            status=status,
            reason=str(exc),
            sha256=sha,
        )


def validate_pdf_corpus(corpus: Path) -> PdfValidationReport:
    """Validate every PDF found in a corpus."""

    files = iter_corpus_files(corpus, PDF_SUFFIXES)
    report = PdfValidationReport(corpus=corpus, total_files=len(files))

    for path in files:
        result = validate_pdf(path)
        report.results.append(result)
        report.counts[result.status] = report.counts.get(result.status, 0) + 1

    return report