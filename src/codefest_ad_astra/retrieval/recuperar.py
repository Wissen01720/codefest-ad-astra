"""Fase 6 — recuperación a nivel de documento sobre una base vectorial ya
construida en la Fase 4.

Reutiliza `Buscador` (`indexing.search`) para la parte de índice → chunks: es
la única implementación que lee el modelo, la dimensión y los prefijos de
consulta/pasaje desde `manifest.json` en vez de asumirlos, así que aquí no se
duplica esa lógica ni se corre el riesgo de que diverja (spec 8.1: la consulta
debe codificarse con el mismo encoder y los mismos prefijos con que se
construyó el índice). Este módulo solo añade lo propio de la Fase 6: agregar
los top-k chunks recuperados a nivel de documento.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..indexing.search import Buscador, Resultado

DEFAULT_TOP_DOCUMENTOS = 3
DEFAULT_K_CHUNKS = 20

# Alias para quien ya importaba ResultadoChunk desde este módulo.
ResultadoChunk = Resultado


@dataclass(slots=True)
class DocumentoRecuperado:
    doc_id: str
    score: float
    fuente: str
    formato: str
    fenomeno: int
    chunks: list[Resultado] = field(default_factory=list)


def agregar_a_documentos(
    resultados: list[Resultado], top_documentos: int = DEFAULT_TOP_DOCUMENTOS
) -> list[DocumentoRecuperado]:
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


def recuperar_documentos(
    carpeta_base: Path,
    consulta: str,
    *,
    buscador: Buscador | None = None,
    device: str | None = None,
    k_chunks: int = DEFAULT_K_CHUNKS,
    top_documentos: int = DEFAULT_TOP_DOCUMENTOS,
) -> list[DocumentoRecuperado]:
    """Busca `consulta`, recupera hasta `k_chunks` fragmentos y los agrega a
    los `top_documentos` documentos con mayor score acumulado.

    `buscador` se puede pasar ya construido (para no recargar el encoder en
    cada consulta, p. ej. en un bucle o servidor); si no se pasa, se construye
    uno nuevo a partir de `carpeta_base`, leyendo modelo/dimensión/prefijos
    de su `manifest.json` — nunca se asume un modelo por defecto aquí.
    """
    if buscador is None:
        buscador = Buscador(carpeta_base, device=device)
    resultados = buscador.buscar(consulta, k=k_chunks)
    return agregar_a_documentos(resultados, top_documentos)