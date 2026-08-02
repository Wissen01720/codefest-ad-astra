"""Automatic schema discovery for JSON corpora."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .diagnostic_common import (
    FieldStatistics,
    flatten_json,
    iter_corpus_files,
    is_http_url,
    load_json_document,
    looks_like_pdf_reference,
)


JSON_SUFFIXES = {".json"}


@dataclass(slots=True)
class SchemaCluster:
    """Documents that share the same structural fingerprint."""

    fingerprint: str
    root_kind: str
    documents: int = 0
    sample_paths: list[str] = field(default_factory=list)
    top_level_key_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def register(self, path: Path, top_level_keys: tuple[str, ...]) -> None:
        self.documents += 1

        if len(self.sample_paths) < 5:
            self.sample_paths.append(str(path))

        for key in top_level_keys:
            self.top_level_key_counts[key] = self.top_level_key_counts.get(key, 0) + 1


@dataclass(slots=True)
class SchemaDetectorReport:
    """Corpus-wide schema discovery results."""

    corpus: Path
    total_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    root_kind_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    top_level_key_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    field_statistics: dict[str, FieldStatistics] = field(default_factory=dict)
    schema_clusters: dict[str, SchemaCluster] = field(default_factory=dict)
    parse_errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def text_fields(self) -> list[FieldStatistics]:
        return sorted(
            (
                stats
                for stats in self.field_statistics.values()
                if stats.text_characters > 0 and stats.pdf_values == 0 and stats.url_values == 0
            ),
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
            (stats for stats in self.field_statistics.values() if stats.pdf_values > 0 or "pdf" in stats.path.lower()),
            key=lambda stats: (-stats.pdf_values, -stats.documents, stats.path),
        )

    @property
    def multi_kind_fields(self) -> list[FieldStatistics]:
        return sorted(
            (
                stats
                for stats in self.field_statistics.values()
                if len(stats.value_kinds) > 1
            ),
            key=lambda stats: (-stats.documents, -len(stats.value_kinds), stats.path),
        )


def _collect_field_statistics(report: SchemaDetectorReport, document) -> None:
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

        if document.root_kind == "dict" and path in document.top_level_keys:
            report.top_level_key_counts[path] = report.top_level_key_counts.get(path, 0) + 1


def discover_json_schemas(corpus: Path) -> SchemaDetectorReport:
    """Scan a corpus and group JSON documents by structural schema."""

    files = iter_corpus_files(corpus, JSON_SUFFIXES)
    report = SchemaDetectorReport(corpus=corpus, total_files=len(files))

    for path in files:
        try:
            document = load_json_document(path)
        except Exception as exc:
            report.failed_files += 1
            report.parse_errors.append((str(path), str(exc)))
            continue

        report.parsed_files += 1
        report.root_kind_counts[document.root_kind] = report.root_kind_counts.get(document.root_kind, 0) + 1

        cluster = report.schema_clusters.get(document.fingerprint)
        if cluster is None:
            cluster = SchemaCluster(
                fingerprint=document.fingerprint,
                root_kind=document.root_kind,
            )
            report.schema_clusters[document.fingerprint] = cluster

        cluster.register(path, document.top_level_keys)
        _collect_field_statistics(report, document)

    return report


def render_schema_summary(report: SchemaDetectorReport) -> str:
    """Render a compact human-readable summary of schema discovery."""

    lines = [
        "SCHEMA DETECTOR REPORT",
        f"Corpus: {report.corpus}",
        f"Files: {report.total_files}",
        f"Parsed: {report.parsed_files}",
        f"Failed: {report.failed_files}",
        "",
        "ROOT KINDS",
    ]

    for kind, count in sorted(report.root_kind_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {kind}: {count}")

    lines.extend(["", "SCHEMA CLUSTERS"])

    for cluster in sorted(report.schema_clusters.values(), key=lambda item: (-item.documents, item.root_kind, item.fingerprint))[:20]:
        lines.append(f"  {cluster.root_kind} {cluster.documents} {cluster.fingerprint}")

    lines.extend(["", "CANDIDATE TEXT FIELDS"])

    for stats in report.text_fields[:20]:
        lines.append(f"  {stats.path} | docs={stats.documents} | chars={stats.text_characters} | mean={stats.mean_characters:.1f}")

    return "\n".join(lines)