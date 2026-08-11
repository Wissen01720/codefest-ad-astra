"""Fase 6 — recuperación, sin depender de un Buscador (no existe en el código real)."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import faiss

from ..indexing.encoder import load_encoder, encode_texts

DEFAULT_TOP_DOCUMENTOS = 3
DEFAULT_K_CHUNKS = 20


@dataclass(slots=True)
class ResultadoChunk:
    rank: int
    score: float
    metadata: dict[str, Any]


def cargar_base(carpeta: Path):
    indice = faiss.read_index(str(carpeta / "index.faiss"))
    metadata = []
    with open(carpeta / "metadata.jsonl", encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                metadata.append(json.loads(linea))
    if indice.ntotal != len(metadata):
        raise RuntimeError(f"Desalineación: índice={indice.ntotal}, metadata={len(metadata)}")
    return indice, metadata


def buscar(indice, metadata, modelo, consulta: str, k: int) -> list[ResultadoChunk]:
    vector = encode_texts(modelo, [consulta], batch_size=1)
    scores, ids = indice.search(vector, k)
    return [
        ResultadoChunk(rank=r, score=float(s), metadata=metadata[i])
        for r, (i, s) in enumerate(zip(ids[0], scores[0]), start=1)
        if i != -1
    ]


@dataclass(slots=True)
class DocumentoRecuperado:
    doc_id: str
    score: float
    fuente: str
    formato: str
    fenomeno: int
    chunks: list[ResultadoChunk] = field(default_factory=list)


def agregar_a_documentos(resultados, top_documentos=DEFAULT_TOP_DOCUMENTOS):
    por_doc: dict[str, DocumentoRecuperado] = {}
    for r in resultados:
        doc_id = r.metadata["doc_id"]
        d = por_doc.setdefault(doc_id, DocumentoRecuperado(
            doc_id=doc_id, score=0.0,
            fuente=r.metadata.get("fuente", ""),
            formato=r.metadata.get("formato", ""),
            fenomeno=r.metadata.get("fenomeno", 0),
        ))
        d.score += r.score
        d.chunks.append(r)
    for d in por_doc.values():
        d.chunks.sort(key=lambda r: r.score, reverse=True)
    return sorted(por_doc.values(), key=lambda d: d.score, reverse=True)[:top_documentos]


def recuperar_documentos(carpeta_base: Path, consulta: str, *, modelo_nombre="BAAI/bge-m3",
                          k_chunks=DEFAULT_K_CHUNKS, top_documentos=DEFAULT_TOP_DOCUMENTOS):
    indice, metadata = cargar_base(carpeta_base)
    modelo = load_encoder(modelo_nombre)
    resultados = buscar(indice, metadata, modelo, consulta, k_chunks)
    return agregar_a_documentos(resultados, top_documentos)