"""Text quality metrics for extracted corpus content."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TextQualityMetrics:
    """Compact summary of text quality signals."""

    characters: int
    words: int
    lines: int
    useful_lines: int
    garbage_lines: int
    symbol_ratio: float
    number_ratio: float
    repetition_ratio: float
    cleaning_ratio: float
    quality_score: float
    quality_label: str


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _line_is_useful(line: str) -> bool:
    stripped = line.strip()

    if len(stripped) < 20:
        return False

    letters = sum(1 for char in stripped if char.isalpha())
    return _safe_ratio(letters, len(stripped)) >= 0.35


def _score_quality(metrics: TextQualityMetrics) -> float:
    useful_ratio = _safe_ratio(metrics.useful_lines, metrics.lines)
    base = 0.45 * useful_ratio
    base += 0.20 * (1.0 - min(metrics.symbol_ratio, 1.0))
    base += 0.15 * (1.0 - min(metrics.number_ratio, 1.0))
    base += 0.20 * (1.0 - min(metrics.repetition_ratio, 1.0))
    base *= max(0.0, min(metrics.cleaning_ratio, 1.0))
    return max(0.0, min(1.0, base))


def _label_quality(score: float) -> str:
    if score >= 0.75:
        return "alta"
    if score >= 0.45:
        return "media"
    return "baja"


def assess_text_quality(text: str, cleaned_text: str | None = None) -> TextQualityMetrics:
    """Compute quality signals for a piece of text."""

    characters = len(text)
    words = len(WORD_PATTERN.findall(text))
    lines_list = text.splitlines()
    lines = len(lines_list) if lines_list else (1 if text else 0)

    useful_lines = sum(1 for line in lines_list if _line_is_useful(line))
    garbage_lines = max(0, lines - useful_lines)

    if characters == 0:
        return TextQualityMetrics(
            characters=0,
            words=0,
            lines=0,
            useful_lines=0,
            garbage_lines=0,
            symbol_ratio=0.0,
            number_ratio=0.0,
            repetition_ratio=0.0,
            cleaning_ratio=0.0,
            quality_score=0.0,
            quality_label="baja",
        )

    symbols = sum(1 for char in text if not char.isalnum() and not char.isspace())
    numbers = sum(1 for char in text if char.isdigit())

    unique_lines = {line.strip() for line in lines_list if line.strip()}
    repetition_ratio = 1.0 - _safe_ratio(len(unique_lines), len([line for line in lines_list if line.strip()]))

    if cleaned_text is None:
        cleaning_ratio = 1.0
    else:
        cleaning_ratio = _safe_ratio(len(cleaned_text), characters)

    metrics = TextQualityMetrics(
        characters=characters,
        words=words,
        lines=lines,
        useful_lines=useful_lines,
        garbage_lines=garbage_lines,
        symbol_ratio=_safe_ratio(symbols, characters),
        number_ratio=_safe_ratio(numbers, characters),
        repetition_ratio=max(0.0, min(1.0, repetition_ratio)),
        cleaning_ratio=max(0.0, min(1.0, cleaning_ratio)),
        quality_score=0.0,
        quality_label="baja",
    )

    score = _score_quality(metrics)
    return TextQualityMetrics(
        characters=metrics.characters,
        words=metrics.words,
        lines=metrics.lines,
        useful_lines=metrics.useful_lines,
        garbage_lines=metrics.garbage_lines,
        symbol_ratio=metrics.symbol_ratio,
        number_ratio=metrics.number_ratio,
        repetition_ratio=metrics.repetition_ratio,
        cleaning_ratio=metrics.cleaning_ratio,
        quality_score=score,
        quality_label=_label_quality(score),
    )