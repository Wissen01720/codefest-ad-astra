"""Genera el reporte final del corpus a partir del checkpoint de batch_runner.py.

Convierte los resultados de PdfValidationResult (guardados en el checkpoint
JSON Lines) a CorpusRecord, y usa corpus_report.py para exportar el reporte
final en JSON, CSV y Markdown -- el artefacto citable para el informe técnico.

Uso:
    uv run python -m codefest_ad_astra.ingest.generar_reporte_final \\
        --checkpoint data/diagnostics/pdf_batch_checkpoint.jsonl \\
        --output data/diagnostics/reporte_pdfs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .batch_runner import load_all_results
from .corpus_report import CorpusRecord, build_corpus_report, write_corpus_report

_STATUS_CON_ERROR = {"PDF_ERROR", "PDF_CORRUPTO", "BATCH_RUNNER_ERROR"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera el reporte final del corpus (JSON/CSV/MD) a partir del checkpoint de PDFs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Ruta base de salida, SIN extensión (ej. data/diagnostics/reporte_pdfs) "
        "-- se generan .json, .csv y .md a partir de esa base.",
    )
    args = parser.parse_args()

    registros: list[CorpusRecord] = []
    for r in load_all_results(args.checkpoint):
        status = r.get("status", "")
        registros.append(
            CorpusRecord(
                path=r.get("path", ""),
                format="pdf",
                status=status,
                characters=r.get("characters", 0) or 0,
                pages=r.get("pages", 0) or 0,
                error=r.get("reason") or r.get("error") if status in _STATUS_CON_ERROR else None,
            )
        )

    report = build_corpus_report(registros)

    for fmt in ("json", "csv", "md"):
        salida = args.output.with_suffix(f".{fmt}")
        write_corpus_report(report, salida, format=fmt)
        print(f"Escrito: {salida}")

    print()
    print(f"Total registros: {report.total_records}")
    print("Por status:")
    for status, cantidad in sorted(report.by_status.items()):
        print(f"  {status}: {cantidad}")


if __name__ == "__main__":
    main()