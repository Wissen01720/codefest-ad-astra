"""Orquestador Fase 1 -> Fase 2: conecta extractors.py con cleaning.py.

Resuelve un problema de contrato entre ambos módulos: extraer_texto()
(el despachador de extractors.py) usa extract_pdf() para PDFs, que ya
devuelve un solo string concatenado -- no la lista por página que
necesita quitar_lineas_repetidas() en cleaning.py para detectar
headers/footers repetidos. Esta función es el único punto que el
equipo de chunking debería llamar: internamente decide, según el
formato, si usa extract_pdf_paginas() (PDF, con lista real de páginas)
o extraer_texto() envuelto en una lista de un elemento (todo lo demás).
"""

from pathlib import Path

from .extractors import extraer_texto, extract_pdf_paginas
from .cleaning import DocumentoLimpio, procesar_documento

_EXTENSIONES_PDF = {".pdf"}


def extraer_y_limpiar(path: Path) -> DocumentoLimpio:
    """Punto de entrada único para el equipo de chunking.

    Extrae el texto crudo (Fase 1) y lo limpia/normaliza (Fase 2) en
    un solo paso, aplicando el tratamiento por página correcto cuando
    el archivo es un PDF.
    """
    es_pdf = path.suffix.lower() in _EXTENSIONES_PDF

    if es_pdf:
        paginas = extract_pdf_paginas(path)
    else:
        paginas = [extraer_texto(path)]

    return procesar_documento(paginas=paginas, es_pdf=es_pdf)