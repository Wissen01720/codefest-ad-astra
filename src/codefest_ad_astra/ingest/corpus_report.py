"""Corpus report generation and export helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, asdict
import csv
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class CorpusRecord:
    """Normalized record used to summarize a corpus."""

    path: str
    format: str = ""
    strategy: str = ""
    institution: str = ""
    language: str = ""
    status: str = ""
    quality_score: float | None = None
    characters: int = 0
    size_bytes: int = 0
    pages: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CorpusReport:
    """Aggregated corpus metrics and the source records that produced them."""

    records: list[CorpusRecord] = field(default_factory=list)
    total_records: int = 0
    by_format: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_strategy: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_institution: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_language: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_status: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_error: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    quality_buckets: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    characters_total: int = 0
    characters_min: int | None = None
    characters_max: int = 0
    size_total: int = 0
    size_min: int | None = None
    size_max: int = 0


def _bucket_quality(score: float | None) -> str:
    if score is None:
        return "sin_calcular"
    if score >= 0.75:
        return "alta"
    if score >= 0.45:
        return "media"
    return "baja"


def build_corpus_report(records: Iterable[CorpusRecord]) -> CorpusReport:
    """Build a corpus report from normalized records."""

    report = CorpusReport()
    report.records = list(records)
    report.total_records = len(report.records)

    for record in report.records:
        if record.format:
            report.by_format[record.format] = report.by_format.get(record.format, 0) + 1

        if record.strategy:
            report.by_strategy[record.strategy] = report.by_strategy.get(record.strategy, 0) + 1

        if record.institution:
            report.by_institution[record.institution] = report.by_institution.get(record.institution, 0) + 1

        if record.language:
            report.by_language[record.language] = report.by_language.get(record.language, 0) + 1

        if record.status:
            report.by_status[record.status] = report.by_status.get(record.status, 0) + 1

        if record.error:
            report.by_error[record.error] = report.by_error.get(record.error, 0) + 1

        report.quality_buckets[_bucket_quality(record.quality_score)] += 1

        report.characters_total += record.characters
        report.size_total += record.size_bytes

        if record.characters:
            if report.characters_min is None or record.characters < report.characters_min:
                report.characters_min = record.characters
            if record.characters > report.characters_max:
                report.characters_max = record.characters

        if record.size_bytes:
            if report.size_min is None or record.size_bytes < report.size_min:
                report.size_min = record.size_bytes
            if record.size_bytes > report.size_max:
                report.size_max = record.size_bytes

    return report


def report_to_dict(report: CorpusReport) -> dict[str, Any]:
    """Serialize a corpus report to a JSON-friendly dictionary."""

    return {
        "summary": {
            "total_records": report.total_records,
            "characters_total": report.characters_total,
            "characters_min": report.characters_min,
            "characters_max": report.characters_max,
            "size_total": report.size_total,
            "size_min": report.size_min,
            "size_max": report.size_max,
            "by_format": dict(sorted(report.by_format.items())),
            "by_strategy": dict(sorted(report.by_strategy.items())),
            "by_institution": dict(sorted(report.by_institution.items())),
            "by_language": dict(sorted(report.by_language.items())),
            "by_status": dict(sorted(report.by_status.items())),
            "by_error": dict(sorted(report.by_error.items())),
            "quality_buckets": dict(sorted(report.quality_buckets.items())),
        },
        "records": [asdict(record) for record in report.records],
    }


def write_corpus_report(report: CorpusReport, output_path: Path, format: str = "json") -> Path:
    """Write a corpus report to JSON, CSV, Markdown, or HTML."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_format = format.lower()

    if normalized_format == "json":
        output_path.write_text(
            json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    if normalized_format == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(report.records[0]).keys()) if report.records else ["path"])
            writer.writeheader()
            for record in report.records:
                writer.writerow(asdict(record))
        return output_path

    if normalized_format in {"md", "markdown"}:
        output_path.write_text(_render_markdown(report), encoding="utf-8")
        return output_path

    if normalized_format == "html":
        output_path.write_text(_render_html(report), encoding="utf-8")
        return output_path

    raise ValueError(f"Formato de salida no soportado: {format}")


def _render_markdown(report: CorpusReport) -> str:
    lines = [
        "# Corpus Report",
        "",
        f"- Total records: {report.total_records}",
        f"- Characters total: {report.characters_total}",
        f"- Size total: {report.size_total}",
        "",
        "## By format",
    ]

    for key, value in sorted(report.by_format.items()):
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## By strategy"])

    for key, value in sorted(report.by_strategy.items()):
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Records", ""])
    lines.append("| path | format | strategy | institution | language | status | quality | chars | size |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for record in report.records[:200]:
        lines.append(
            f"| {record.path} | {record.format} | {record.strategy} | {record.institution} | {record.language} | {record.status} | {record.quality_score if record.quality_score is not None else ''} | {record.characters} | {record.size_bytes} |"
        )

    return "\n".join(lines)


def _render_html(report: CorpusReport) -> str:
    rows = []

    for record in report.records[:500]:
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{escape(str(value))}</td>"
                for value in (
                    record.path,
                    record.format,
                    record.strategy,
                    record.institution,
                    record.language,
                    record.status,
                    "" if record.quality_score is None else f"{record.quality_score:.3f}",
                    record.characters,
                    record.size_bytes,
                )
            )
            + "</tr>"
        )

    summary_rows = []
    for label, value in (
        ("Total records", report.total_records),
        ("Characters total", report.characters_total),
        ("Size total", report.size_total),
    ):
        summary_rows.append(f"<tr><th>{escape(label)}</th><td>{escape(str(value))}</td></tr>")

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Corpus Report</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#111} table{border-collapse:collapse;width:100%;margin:1rem 0} td,th{border:1px solid #ddd;padding:.45rem;text-align:left} th{background:#f5f5f5}</style>"
        "</head><body>"
        "<h1>Corpus Report</h1>"
        f"<table>{''.join(summary_rows)}</table>"
        "<h2>Records</h2>"
        "<table><thead><tr><th>Path</th><th>Format</th><th>Strategy</th><th>Institution</th><th>Language</th><th>Status</th><th>Quality</th><th>Chars</th><th>Size</th></tr></thead><tbody>"
        f"{''.join(rows)}"
        "</tbody></table></body></html>"
    )