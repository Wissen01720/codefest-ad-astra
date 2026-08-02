"""Resume el checkpoint de batch_runner.py: cuenta PDFs por categoría.

Uso:
    uv run python -m codefest_ad_astra.ingest.resumen_checkpoint --checkpoint data/diagnostics/pdf_batch_checkpoint.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def resumir(checkpoint_path: Path) -> None:
    conteo: Counter[str] = Counter()
    total = 0
    ejemplos_por_categoria: dict[str, str] = {}

    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            status = record.get("status", "SIN_STATUS")
            conteo[status] += 1
            total += 1
            if status not in ejemplos_por_categoria:
                ejemplos_por_categoria[status] = record.get("path", "")

    print(f"\nTotal de registros en checkpoint: {total}\n")
    print(f"{'Categoría':<20} {'Cantidad':>10} {'Porcentaje':>12}")
    print("-" * 44)
    for status, cantidad in conteo.most_common():
        porcentaje = (cantidad / total * 100) if total else 0
        print(f"{status:<20} {cantidad:>10} {porcentaje:>11.1f}%")

    print("\nUn ejemplo de cada categoría:")
    for status, ejemplo in ejemplos_por_categoria.items():
        print(f"  {status}: {ejemplo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume el checkpoint de batch_runner.py por categoría de PdfStatus.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    resumir(args.checkpoint)