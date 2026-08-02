"""Resolución de fuentes para documentos JSON.

Fase 1 - Ingesta.

Este módulo NO descarga ni extrae contenido todavía.

Su responsabilidad es inspeccionar un JSON y determinar de dónde debe
obtenerse el contenido principal:

    1. contenido local
    2. PDF externo
    3. página HTML externa
    4. abstract/metadata
    5. JSON sin estrategia conocida
    6. manifiesto/catálogo/registro

La separación es intencional: extractors.py extrae contenido, mientras
resolvers.py decide qué recurso debe utilizarse.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Campos conocidos
# ---------------------------------------------------------------------------

CAMPOS_TITULO = (
    "title",
    "headline",
    "titulo",
)

CAMPOS_CUERPO = (
    "body_text",
    "body_paragraphs",
    "text",
    "content",
    "article_text",
    "full_text",
    "description",
)

CAMPOS_ABSTRACT = (
    "abstract",
    "resumen",
    "excerpt",
    "summary",
)

CAMPOS_URL = (
    "url",
    "source_url",
    "web_url",
    "article_url",
)

CAMPOS_PDF_URL = (
    "pdf_url",
    "pdf",
    "pdf_link",
    "download_url",
)

# Campos que suelen indicar que el JSON es un índice/manifiesto
CAMPOS_MANIFIESTO = (
    "hashes",
    "urls",
    "articulos",
    "articles",
    "web_publications",
    "rss_articles",
    "pdfs_downloaded",
    "pdfs_descargados",
    "paginas_scrapeadas",
    "total_publicaciones",
)


# ---------------------------------------------------------------------------
# Estrategias
# ---------------------------------------------------------------------------

ESTRATEGIA_LOCAL = "local"
ESTRATEGIA_PDF = "pdf"
ESTRATEGIA_HTML = "html"
ESTRATEGIA_ABSTRACT = "abstract"
ESTRATEGIA_MANIFIESTO = "manifiesto"
ESTRATEGIA_DESCONOCIDA = "desconocida"


@dataclass
class ResolucionJSON:
    """Resultado de analizar un JSON."""

    estrategia: str

    # Campo que contiene el contenido local, si existe.
    campo_cuerpo: str | None = None

    # URL principal del documento.
    url: str | None = None

    # URL de PDF, si existe.
    pdf_url: str | None = None

    # Campo usado para el abstract.
    campo_abstract: str | None = None

    # Título, cuando está disponible.
    titulo: str | None = None

    # Motivo legible para diagnóstico.
    razon: str = ""

    # Algunas claves detectadas que pueden ser útiles para investigar
    # esquemas nuevos.
    claves: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _primer_campo_con_valor(
    data: dict[str, Any],
    campos: tuple[str, ...],
) -> tuple[str | None, Any]:
    """Devuelve el primer campo existente con un valor útil."""
    for campo in campos:
        valor = data.get(campo)

        if valor is None:
            continue

        if isinstance(valor, str) and not valor.strip():
            continue

        if isinstance(valor, list) and not valor:
            continue

        return campo, valor

    return None, None


def _es_url(valor: Any) -> bool:
    """Comprueba si un valor parece una URL HTTP/HTTPS."""
    if not isinstance(valor, str):
        return False

    try:
        parsed = urlparse(valor)
    except ValueError:
        return False

    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _buscar_url(
    data: dict[str, Any],
    campos: tuple[str, ...],
) -> str | None:
    """Busca una URL en una lista de campos conocidos."""
    for campo in campos:
        valor = data.get(campo)

        if _es_url(valor):
            return valor

    return None


def _es_lista_de_texto(valor: Any) -> bool:
    """Indica si el valor es una lista con elementos de texto."""
    return (
        isinstance(valor, list)
        and bool(valor)
        and all(isinstance(item, str) for item in valor)
    )


def _tiene_contenido_local(
    data: dict[str, Any],
) -> tuple[str | None, Any]:
    """Busca un campo que represente contenido textual local."""
    for campo in CAMPOS_CUERPO:
        valor = data.get(campo)

        if isinstance(valor, str) and valor.strip():
            return campo, valor

        if _es_lista_de_texto(valor):
            # Evitamos considerar listas de strings vacías.
            texto = "\n\n".join(
                item.strip()
                for item in valor
                if item.strip()
            )

            if texto:
                return campo, valor

    return None, None


def _tiene_indicios_de_manifiesto(data: dict[str, Any]) -> bool:
    """Detecta estructuras que parecen índices o manifiestos.

    No significa que debamos descartarlas automáticamente. Solo permite
    clasificarlas para que el pipeline pueda decidir posteriormente.
    """
    claves = set(data.keys())

    coincidencias = claves.intersection(CAMPOS_MANIFIESTO)

    return bool(coincidencias)


# ---------------------------------------------------------------------------
# Resolución principal
# ---------------------------------------------------------------------------

def resolver_json(data: dict[str, Any]) -> ResolucionJSON:
    """Determina cómo debe resolverse un JSON.

    Orden de prioridad:

        1. contenido local
        2. PDF externo
        3. HTML externo
        4. abstract
        5. manifiesto
        6. desconocido

    La prioridad del contenido local es importante. Si un JSON tiene
    body_text y también url/pdf_url, no debemos volver a descargar el
    contenido externo y duplicarlo.
    """

    if not isinstance(data, dict):
        raise TypeError(
            f"resolver_json esperaba un dict, recibió {type(data).__name__}"
        )

    claves = tuple(sorted(data.keys()))

    # ---------------------------------------------------------------
    # 1. Contenido local
    # ---------------------------------------------------------------

    campo_cuerpo, _ = _tiene_contenido_local(data)

    if campo_cuerpo:
        titulo, titulo_valor = _primer_campo_con_valor(
            data,
            CAMPOS_TITULO,
        )

        url = _buscar_url(data, CAMPOS_URL)
        pdf_url = _buscar_url(data, CAMPOS_PDF_URL)

        return ResolucionJSON(
            estrategia=ESTRATEGIA_LOCAL,
            campo_cuerpo=campo_cuerpo,
            url=url,
            pdf_url=pdf_url,
            titulo=str(titulo_valor) if titulo_valor else None,
            razon=(
                f"Se encontró contenido local en el campo "
                f"'{campo_cuerpo}'."
            ),
            claves=claves,
        )

    # ---------------------------------------------------------------
    # 2. PDF externo
    # ---------------------------------------------------------------

    pdf_url = _buscar_url(data, CAMPOS_PDF_URL)

    if pdf_url:
        titulo, titulo_valor = _primer_campo_con_valor(
            data,
            CAMPOS_TITULO,
        )

        url = _buscar_url(data, CAMPOS_URL)

        return ResolucionJSON(
            estrategia=ESTRATEGIA_PDF,
            pdf_url=pdf_url,
            url=url,
            titulo=str(titulo_valor) if titulo_valor else None,
            razon=(
                "No hay contenido local, pero el JSON proporciona "
                "una URL de PDF."
            ),
            claves=claves,
        )

    # ---------------------------------------------------------------
    # 3. Página HTML externa
    # ---------------------------------------------------------------

    url = _buscar_url(data, CAMPOS_URL)

    if url:
        titulo, titulo_valor = _primer_campo_con_valor(
            data,
            CAMPOS_TITULO,
        )

        return ResolucionJSON(
            estrategia=ESTRATEGIA_HTML,
            url=url,
            titulo=str(titulo_valor) if titulo_valor else None,
            razon=(
                "No hay contenido local ni PDF, pero el JSON "
                "proporciona una URL de recurso."
            ),
            claves=claves,
        )

    # ---------------------------------------------------------------
    # 4. Abstract
    # ---------------------------------------------------------------

    campo_abstract, abstract = _primer_campo_con_valor(
        data,
        CAMPOS_ABSTRACT,
    )

    if campo_abstract:
        titulo, titulo_valor = _primer_campo_con_valor(
            data,
            CAMPOS_TITULO,
        )

        return ResolucionJSON(
            estrategia=ESTRATEGIA_ABSTRACT,
            campo_abstract=campo_abstract,
            titulo=str(titulo_valor) if titulo_valor else None,
            razon=(
                f"No se encontró contenido local ni URL; se encontró "
                f"un abstract en '{campo_abstract}'."
            ),
            claves=claves,
        )

    # ---------------------------------------------------------------
    # 5. Manifiesto / catálogo / registro
    # ---------------------------------------------------------------

    if _tiene_indicios_de_manifiesto(data):
        return ResolucionJSON(
            estrategia=ESTRATEGIA_MANIFIESTO,
            razon=(
                "La estructura contiene campos característicos de "
                "índices, catálogos o registros."
            ),
            claves=claves,
        )

    # ---------------------------------------------------------------
    # 6. Desconocido
    # ---------------------------------------------------------------

    return ResolucionJSON(
        estrategia=ESTRATEGIA_DESCONOCIDA,
        razon=(
            "No se encontró un campo de contenido, PDF, URL, "
            "abstract ni estructura de manifiesto conocida."
        ),
        claves=claves,
    )


# ---------------------------------------------------------------------------
# Resolver directamente desde archivo
# ---------------------------------------------------------------------------

def resolver_archivo_json(path: Path) -> ResolucionJSON:
    """Carga un JSON del disco y devuelve su resolución."""

    data = json.loads(
        path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    )

    # -------------------------------------------------------
    # Algunos catálogos vienen como lista.
    # Intentamos resolver usando el primer elemento.
    # -------------------------------------------------------

    if isinstance(data, list):

        if not data:
            return ResolucionJSON(
                estrategia=ESTRATEGIA_MANIFIESTO,
                razon="Lista JSON vacía.",
            )

        primer = next(
            (
                item
                for item in data
                if isinstance(item, dict)
            ),
            None,
        )

        if primer is None:
            return ResolucionJSON(
                estrategia=ESTRATEGIA_MANIFIESTO,
                razon="Lista JSON sin elementos tipo dict.",
            )

        return resolver_json(primer)

    return resolver_json(data)