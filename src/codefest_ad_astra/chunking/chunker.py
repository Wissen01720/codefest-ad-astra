"""Fase 3: empaqueta párrafos completos dentro del presupuesto de tokens del
encoder; solo baja a nivel oración cuando un párrafo entero no cabe.

Nunca corta una oración a mitad (spec Sección 3.3): el único lugar donde se
decide un corte es entre oraciones completas de `split_sentences()`.
"""
from __future__ import annotations

import re
import sys
from typing import Callable

from ..ingest.validation import Document
from .fragment import Fragment
from .sentence_splitter import split_sentences
from .tokenizer import count_tokens

_SEPARADOR_PARRAFOS = re.compile(r"\n{2,}")
_PATRON_PUNTUACION_ORACION = re.compile(r"[.!?]")


def _dividir_en_parrafos(texto: str) -> list[str]:
    return [p.strip() for p in _SEPARADOR_PARRAFOS.split(texto) if p.strip()]


def _partir_unidad_sobredimensionada(
    unidad: str, max_tokens: int, contar_tokens: Callable[[str], int]
) -> list[str]:
    """Divide una unidad de texto sin estructura de oraciones (p. ej. una fila
    tabular de CSV/XLSX/PBF convertida en una única "oración" sin puntuación
    de cierre) en piezas empaquetadas greedily por palabras hasta `max_tokens`.

    No es un corte por límite de oración (no hay oraciones que preservar en
    texto tabular) -- es un último recurso para no emitir un chunk de decenas
    de miles de caracteres que el encoder truncaría silenciosamente."""
    palabras = unidad.split()
    piezas: list[str] = []
    pieza_actual: list[str] = []
    tokens_pieza_actual = 0

    for palabra in palabras:
        tokens_palabra = contar_tokens(palabra)
        if pieza_actual and tokens_pieza_actual + tokens_palabra > max_tokens:
            piezas.append(" ".join(pieza_actual))
            pieza_actual = []
            tokens_pieza_actual = 0
        pieza_actual.append(palabra)
        tokens_pieza_actual += tokens_palabra

    if pieza_actual:
        piezas.append(" ".join(pieza_actual))

    return piezas


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
    # Rastrea si `buffer` recibió contenido nuevo desde el último cierre.
    # Sin esto, dos cierres consecutivos sin un `buffer.append()` entre
    # medio (p. ej. dos oraciones sobredimensionadas seguidas) reemiten el
    # mismo `buffer` (la cola de overlap) como un chunk duplicado -- ver
    # regresión en test_oraciones_sobredimensionadas_consecutivas_no_duplican_chunk.
    buffer_tiene_contenido_nuevo = False

    def _cerrar_chunk_actual() -> None:
        nonlocal buffer, buffer_tokens, buffer_tiene_contenido_nuevo
        if not buffer or not buffer_tiene_contenido_nuevo:
            return
        textos_de_chunks.append(" ".join(buffer))
        cola = buffer[-overlap_sentences:] if overlap_sentences > 0 else []
        buffer = list(cola)
        buffer_tokens = sum(contar_tokens(o) for o in buffer)
        buffer_tiene_contenido_nuevo = False

    for parrafo in parrafos:
        oraciones_parrafo = split_sentences(parrafo)
        if not oraciones_parrafo:
            continue

        tokens_parrafo = sum(contar_tokens(o) for o in oraciones_parrafo)

        if buffer_tokens + tokens_parrafo <= max_tokens:
            buffer.extend(oraciones_parrafo)
            buffer_tokens += tokens_parrafo
            buffer_tiene_contenido_nuevo = True
            continue

        for oracion in oraciones_parrafo:
            tokens_oracion = contar_tokens(oracion)

            if tokens_oracion > max_tokens:
                _cerrar_chunk_actual()
                # Solo se activa el fallback de división por palabras cuando
                # la unidad sobredimensionada no tiene NINGÚN punto de
                # corte de oración ([.!?]) en absoluto -- esa es la firma
                # real de texto tabular sin estructura de oraciones (filas
                # de CSV/XLSX/PBF, "columna: valor" sin puntuación de
                # cierre en ningún lado) -- ver hallazgo #4 de la revisión
                # de rama completa.
                #
                # OJO: `len(oraciones_parrafo) == 1` NO es un gate válido
                # para esto -- una oración real, larga, con puntuación de
                # cierre normal (p. ej. una sola oración de 300 palabras
                # que termina en '.') también produce `oraciones_parrafo`
                # de longitud 1 (split_sentences no tiene nada que cortar
                # dentro de una sola oración), pero SÍ debe dejarse sola
                # verbatim, no dividirse por palabras (spec 3.3, "nunca
                # corta una oración real a mitad"). El gate correcto es la
                # ausencia total de puntuación de cierre en el texto crudo
                # de la unidad, no el conteo de oraciones que devolvió
                # split_sentences().
                if not _PATRON_PUNTUACION_ORACION.search(oracion):
                    piezas = _partir_unidad_sobredimensionada(oracion, max_tokens, contar_tokens)
                    if len(piezas) > 1:
                        print(
                            f"[chunker] Aviso: unidad sobredimensionada en {doc.doc_id} "
                            f"({tokens_oracion} tokens > max_tokens={max_tokens}) sin estructura "
                            f"de oraciones -- dividida en {len(piezas)} chunks por palabras "
                            "(probable contenido tabular: CSV/XLSX/PBF).",
                            file=sys.stderr,
                        )
                    textos_de_chunks.extend(piezas)
                else:
                    textos_de_chunks.append(oracion)
                continue

            if buffer_tokens + tokens_oracion > max_tokens:
                _cerrar_chunk_actual()
                # El cierre anterior puede haber sido un no-op (buffer sin
                # contenido nuevo, p. ej. la cola de overlap tras un cierre
                # previo): en ese caso `buffer` sigue conteniendo esa cola y
                # sumarle `oracion` la dejaría por encima de max_tokens sin
                # ningún cierre real que la vuelva a bajar. La cola de
                # overlap ya fue emitida una vez en su chunk original y no
                # tiene dónde más ir, así que se descarta aquí -- esto no
                # reintroduce el bug de duplicados de la Task 5 porque una
                # cola descartada sin emitir no es una re-emisión.
                if buffer_tokens + tokens_oracion > max_tokens:
                    buffer, buffer_tokens = [], 0

            buffer.append(oracion)
            buffer_tokens += tokens_oracion
            buffer_tiene_contenido_nuevo = True

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
