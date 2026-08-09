"""Orquestador de la Fase 3: lee documentos.jsonl (salida de
ingest.pipeline, Fase 1+2) y genera fragments.jsonl, un chunk por línea,
listo para la Fase 4 (embeddings + índice FAISS).

Uso:
    uv run python -m codefest_ad_astra.chunking.pipeline \
        --documentos data/processed/documentos.jsonl \
        --salida data/processed/fragments.jsonl
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

from ..ingest.validation import Document
from .chunker import chunk_document
from .fragment import Fragment
from .tokenizer import DEFAULT_ENCODER, count_tokens


def _leer_documentos(path: Path) -> Iterator[Document]:
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                yield Document(**json.loads(linea))


def procesar_documentos(
    documentos: Path,
    max_tokens: int = 400,
    overlap_sentences: int = 1,
    model_name: str = DEFAULT_ENCODER,
    contar_tokens: Callable[[str], int] | None = None,
) -> Iterator[Fragment]:
    contar = contar_tokens or partial(count_tokens, model_name=model_name)
    for doc in _leer_documentos(documentos):
        yield from chunk_document(
            doc,
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
            contar_tokens=contar,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 3: chunking de documentos.jsonl -> fragments.jsonl")
    parser.add_argument("--documentos", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--overlap-sentences", type=int, default=1)
    parser.add_argument("--modelo", type=str, default=DEFAULT_ENCODER)
    args = parser.parse_args()

    args.salida.parent.mkdir(parents=True, exist_ok=True)

    total_fragmentos = 0
    docs_vistos: set[str] = set()
    with open(args.salida, "w", encoding="utf-8") as f:
        for fragmento in procesar_documentos(
            args.documentos, args.max_tokens, args.overlap_sentences, args.modelo,
        ):
            f.write(fragmento.to_json_line() + "\n")
            total_fragmentos += 1
            docs_vistos.add(fragmento.doc_id)

    print(f"\nListo: {len(docs_vistos)} documentos -> {total_fragmentos} fragmentos -> {args.salida}")


if __name__ == "__main__":
    main()
