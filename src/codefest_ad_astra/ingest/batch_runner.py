"""
batch_runner.py

Orquestador por lotes para validación masiva de PDFs (y, por extensión,
cualquier otro validador de diagnostic_common.py que sea costoso de correr
sobre el corpus completo en una sola pasada).

Diseño:
- No conoce la lógica interna de pdf_validator.py: solo lo invoca por archivo.
- Persiste resultados incrementales en JSON Lines (un archivo por línea),
  para poder reanudar si el proceso se interrumpe.
- Usa el hash de archivo (asumido en diagnostic_common.py) para saltar
  archivos ya procesados en corridas anteriores -> idempotente.
- Procesa en chunks configurables, con progreso visible.

Firmas reales usadas (confirmadas contra el código):
- pdf_validator.validate_pdf(path: Path) -> PdfValidationResult
  (dataclass slots=True con: path, status: PdfStatus, pages, pages_with_text,
  pages_with_images, characters, reason, sha256: str | None, errors: tuple)
- diagnostic_common.file_sha256(path: Path) -> str
- diagnostic_common.iter_corpus_files(corpus: Path, suffixes: set[str]) -> list[Path]
- pdf_validator.PDF_SUFFIXES = {".pdf"}

El checkpoint se identifica por el campo "sha256" del resultado (no por ruta),
así detecta archivos movidos/renombrados con el mismo contenido y evita
reprocesar duplicados. path y status (Path y Enum-str) se normalizan a texto
plano antes de escribir a JSON, porque json.dumps no serializa Path directamente.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .diagnostic_common import file_sha256, iter_corpus_files
from .pdf_validator import PDF_SUFFIXES, validate_pdf


@dataclass(frozen=True)
class BatchConfig:
    """Configuración de una corrida por lotes."""
    source_dir: Path
    checkpoint_path: Path
    chunk_size: int = 50
    log_every: int = 10


@dataclass(frozen=True)
class BatchProgress:
    """Resumen de una corrida (para reporte o log final)."""
    total_found: int
    already_done: int
    processed_this_run: int
    failed_this_run: int
    elapsed_seconds: float


def _load_checkpoint_hashes(checkpoint_path: Path) -> set[str]:
    """Lee el checkpoint existente y devuelve el set de hashes ya procesados.

    El checkpoint es JSON Lines: una línea por resultado ya persistido.
    Si el archivo no existe aún, devuelve un set vacío (primera corrida).
    """
    if not checkpoint_path.exists():
        return set()

    done: set[str] = set()
    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Línea corrupta (ej. corte a mitad de escritura por caída
                # previa del proceso) -> se ignora, no se cuenta como hecha,
                # el archivo correspondiente simplemente se reprocesará.
                continue
            file_hash = record.get("sha256")
            if file_hash:
                done.add(file_hash)
    return done


def _iter_chunks(items: list[Path], chunk_size: int) -> Iterator[list[Path]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _result_to_dict(result) -> dict:
    """Normaliza el resultado del validador a un dict serializable.

    Acepta dataclasses o dicts directamente, para no acoplar el runner
    a un tipo concreto de retorno. PdfValidationResult.path es un Path
    -- json.dumps no lo serializa, así que se convierte a str aquí.
    PdfStatus ya es str-Enum, así que serializa bien tal cual.
    """
    if is_dataclass(result):
        record = asdict(result)
    elif isinstance(result, dict):
        record = dict(result)
    else:
        raise TypeError(
            f"El validador debe devolver un dataclass o un dict, recibió {type(result)!r}"
        )

    if "path" in record and isinstance(record["path"], Path):
        record["path"] = str(record["path"])

    return record


def run_batch(
    config: BatchConfig,
    validate_fn: Callable[[Path], object] = validate_pdf,
    hash_fn: Callable[[Path], str] = file_sha256,
) -> BatchProgress:
    """Ejecuta la validación por lotes con checkpoint y reanudación.

    Por defecto usa pdf_validator.validate_pdf y diagnostic_common.file_sha256.
    Se dejan como parámetros (no hardcoded) para poder reusar el mismo
    orquestador con ocr_validator.py más adelante sin duplicar esta lógica.
    """
    started = time.monotonic()

    print(f"[batch_runner] escaneando {config.source_dir} ...", flush=True)
    all_files = iter_corpus_files(config.source_dir, PDF_SUFFIXES)
    total_found = len(all_files)
    print(f"[batch_runner] {total_found} PDFs encontrados", flush=True)

    done_hashes = _load_checkpoint_hashes(config.checkpoint_path)
    already_done = len(done_hashes)
    print(f"[batch_runner] {already_done} ya procesados en checkpoint previo", flush=True)

    # hash_fn se llama antes de abrir el PDF; si un archivo no es legible
    # a nivel de I/O (permisos, symlink roto), esto puede lanzar antes de
    # llegar a validate_pdf. Se protege aquí para que un archivo problemático
    # no tumbe el filtrado de pendientes de todo el lote.
    print("[batch_runner] calculando hashes para filtrar pendientes (puede tardar)...", flush=True)
    pending: list[Path] = []
    for i, path in enumerate(all_files, start=1):
        try:
            current_hash = hash_fn(path)
        except OSError as exc:
            print(f"[batch_runner] no se pudo leer {path} para hashear: {exc} -- se omite", flush=True)
            continue
        if current_hash in done_hashes:
            continue
        pending.append(path)
        if i % 100 == 0:
            print(f"[batch_runner] hasheados {i}/{total_found}...", flush=True)

    print(f"[batch_runner] {len(pending)} PDFs pendientes por validar", flush=True)

    processed = 0
    failed = 0

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Modo "append": nunca reescribe lo ya persistido, solo agrega lo nuevo.
    with config.checkpoint_path.open("a", encoding="utf-8") as checkpoint_file:
        for chunk_index, chunk in enumerate(_iter_chunks(pending, config.chunk_size), start=1):
            for path in chunk:
                try:
                    result = validate_fn(path)
                    record = _result_to_dict(result)
                except Exception as exc:  # noqa: BLE001 - se registra y se sigue
                    failed += 1
                    record = {
                        "path": str(path),
                        "status": "BATCH_RUNNER_ERROR",
                        "error": str(exc),
                        "file_hash": hash_fn(path),
                    }

                checkpoint_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                checkpoint_file.flush()  # garantiza que un corte a mitad no pierda el resto
                processed += 1

                if processed % config.log_every == 0:
                    print(
                        f"[batch_runner] procesados {processed}/{len(pending)} "
                        f"(chunk {chunk_index}) -- fallidos: {failed}",
                        flush=True,
                    )

    elapsed = time.monotonic() - started
    return BatchProgress(
        total_found=total_found,
        already_done=already_done,
        processed_this_run=processed,
        failed_this_run=failed,
        elapsed_seconds=elapsed,
    )


def load_all_results(checkpoint_path: Path) -> Iterable[dict]:
    """Lee todos los resultados acumulados del checkpoint, para pasarlos
    a corpus_report.py una vez que el batch está completo (o parcialmente
    completo -- el reporte puede correr sobre lo que haya hasta el momento).
    """
    if not checkpoint_path.exists():
        return []
    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validación por lotes de PDFs con checkpoint (no se cuelga en corridas completas)."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Carpeta raíz del corpus a validar (ej. data/raw_muestra o data/).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Ruta del archivo de checkpoint (JSON Lines). Por defecto: "
        "data/diagnostics/pdf_batch_checkpoint.jsonl",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Cantidad de PDFs a procesar por lote antes de loguear progreso (default: 50).",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or Path("data/diagnostics/pdf_batch_checkpoint.jsonl")

    config = BatchConfig(
        source_dir=args.corpus,
        checkpoint_path=checkpoint_path,
        chunk_size=args.chunk_size,
    )
    progress = run_batch(config)
    print()
    print(f"Total encontrados:      {progress.total_found}")
    print(f"Ya procesados (previo):  {progress.already_done}")
    print(f"Procesados esta corrida: {progress.processed_this_run}")
    print(f"Fallidos esta corrida:   {progress.failed_this_run}")
    print(f"Tiempo:                  {progress.elapsed_seconds:.1f}s")
    print(f"Checkpoint en:           {checkpoint_path}")