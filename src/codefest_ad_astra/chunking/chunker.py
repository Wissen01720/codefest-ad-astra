"""Fase 3: empaqueta párrafos completos dentro del presupuesto de tokens del
encoder; solo baja a nivel oración cuando un párrafo entero no cabe.

Nunca corta una oración a mitad (spec Sección 3.3): el único lugar donde se
decide un corte es entre oraciones completas de `split_sentences()`.
"""
from __future__ import annotations

import re
from typing import Callable

from ..ingest.validation import Document
from .fragment import Fragment
from .sentence_splitter import split_sentences
from .tokenizer import count_tokens

_SEPARADOR_PARRAFOS = re.compile(r"\n{2,}")


def _dividir_en_parrafos(texto: str) -> list[str]:
    return [p.strip() for p in _SEPARADOR_PARRAFOS.split(texto) if p.strip()]


def chunk_document(
    doc: Document,
    max_tokens: int = 400,
    overlap_sentences: int = 1,
    contar_tokens: Callable[[str], int] = count_tokens,
) -> list[Fragment]:
    """Fragmenta `doc.texto` respetando completitud lingüística.

    Estrategia: por párrafo (separado por línea en blanco), empaquetando
    párrafos completos mientras quepan en `max_tokens`; si un párrafo no
    cabe junto al buffer actual, se cierra el chunk actual y se procesa
    ese párrafo oración por oración. `overlap_sentences` controla cuántas
    oraciones del final de un chunk se repiten al inicio del siguiente.
    """
    parrafos = _dividir_en_parrafos(doc.texto)
    if not parrafos:
        return []

    buffer: list[str] = []
    buffer_tokens = 0
    textos_de_chunks: list[str] = []

    def _cerrar_chunk_actual() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        textos_de_chunks.append(" ".join(buffer))
        cola = buffer[-overlap_sentences:] if overlap_sentences > 0 else []
        buffer = list(cola)
        buffer_tokens = sum(contar_tokens(o) for o in buffer)

    for parrafo in parrafos:
        oraciones_parrafo = split_sentences(parrafo)
        if not oraciones_parrafo:
            continue

        tokens_parrafo = sum(contar_tokens(o) for o in oraciones_parrafo)

        if buffer_tokens + tokens_parrafo <= max_tokens:
            buffer.extend(oraciones_parrafo)
            buffer_tokens += tokens_parrafo
            continue

        for oracion in oraciones_parrafo:
            tokens_oracion = contar_tokens(oracion)

            if tokens_oracion > max_tokens:
                _cerrar_chunk_actual()
                textos_de_chunks.append(oracion)
                continue

            if buffer_tokens + tokens_oracion > max_tokens:
                _cerrar_chunk_actual()

            buffer.append(oracion)
            buffer_tokens += tokens_oracion

    _cerrar_chunk_actual()

    return [
        Fragment(
            doc_id=doc.doc_id,
            chunk_id=f"{doc.doc_id}-chunk-{posicion:03d}",
            fuente=doc.fuente,
            formato=doc.formato,
            fenomeno=doc.fenomeno,
            posicion=posicion,
            num_tokens=contar_tokens(texto_chunk),
            texto=texto_chunk,
            idioma=doc.idioma,
        )
        for posicion, texto_chunk in enumerate(textos_de_chunks)
    ]
