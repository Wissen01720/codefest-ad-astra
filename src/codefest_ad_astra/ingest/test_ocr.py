"""Corre validate_image_corpus() sobre las imágenes reales del corpus.

Solo hay 8 imágenes en todo data/raw_muestra, así que no hace falta
batching -- se valida todo en una sola corrida.

Uso:
    uv run python -m codefest_ad_astra.ingest.test_ocr --corpus data/raw_muestra
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .ocr_validator import validate_image_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida las imágenes del corpus con ocr_validator.py.")
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()

    report = validate_image_corpus(args.corpus)

    print(f"\nTotal de imágenes encontradas: {report.total_files}\n")
    for status, cantidad in report.counts.items():
        print(f"  {status.value:<20} {cantidad}")

    print("\nDetalle por imagen:")
    for resultado in report.results:
        print(f"  {resultado.path.name:<50} {resultado.status.value:<20} "
              f"{resultado.width}x{resultado.height}  chars={resultado.characters}")


if __name__ == "__main__":
    main()