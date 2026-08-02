"""JSON field profiler for corpus-level statistics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .diagnostic_common import FieldStatistics, flatten_json, iter_corpus_files, load_json_document


JSON_SUFFIXES = {".json"}


@dataclass(slots=True)
class JsonProfileReport:
    """Per-field statistics collected from a JSON corpus."""

    corpus: Path
    total_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    root_kind_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    field_statistics: dict[str, FieldStatistics] = field(default_factory=dict)
    parse_errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def text_fields(self) -> list[FieldStatistics]:
        return sorted(
            (stats for stats in self.field_statistics.values() if stats.text_characters > 0),
            key=lambda stats: (-stats.documents, -stats.text_characters, stats.path),
        )

    @property
    def url_fields(self) -> list[FieldStatistics]:
        return sorted(
            (stats for stats in self.field_statistics.values() if stats.url_values > 0),
            key=lambda stats: (-stats.url_values, -stats.documents, stats.path),
        )

    @property
    def pdf_fields(self) -> list[FieldStatistics]:
        return sorted(
            (stats for stats in self.field_statistics.values() if stats.pdf_values > 0),
            key=lambda stats: (-stats.pdf_values, -stats.documents, stats.path),
        )


def _collect_field_statistics(report: JsonProfileReport, document) -> None:
    fields: dict[str, list[object]] = defaultdict(list)

    for path, value in flatten_json(document.data):
        if not path:
            continue
        fields[path].append(value)

    for path, values in fields.items():
        stats = report.field_statistics.setdefault(path, FieldStatistics(path=path))
        stats.register_document()

        for value in values:
            stats.register_value(value)


def profile_json_corpus(corpus: Path) -> JsonProfileReport:
    """Profile the occurrence and size of JSON fields in a corpus."""

    files = iter_corpus_files(corpus, JSON_SUFFIXES)
    report = JsonProfileReport(corpus=corpus, total_files=len(files))

    for path in files:
        try:
            document = load_json_document(path)
        except Exception as exc:
            report.failed_files += 1
            report.parse_errors.append((str(path), str(exc)))
            continue

        report.parsed_files += 1
        report.root_kind_counts[document.root_kind] = report.root_kind_counts.get(document.root_kind, 0) + 1
        _collect_field_statistics(report, document)

    return report


def top_text_fields(report: JsonProfileReport, limit: int = 20) -> list[FieldStatistics]:
    """Return the most informative text-bearing fields."""

    return report.text_fields[:limit]