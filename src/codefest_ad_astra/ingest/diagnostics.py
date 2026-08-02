"""
Herramientas de diagnóstico para la Fase 1.

Este módulo NO modifica el corpus.

Su objetivo es identificar exactamente por qué un recurso
no puede convertirse en texto útil para el pipeline.

Produce información suficiente para:

- descubrir nuevos esquemas JSON
- detectar PDFs corruptos
- detectar PDFs escaneados
- detectar OCR deficiente
- detectar HTML vacío
- detectar imágenes sin OCR
- detectar manifiestos
- detectar catálogos
- detectar recursos duplicados
- producir estadísticas de calidad

Este archivo es únicamente de diagnóstico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import hashlib
import json
import traceback

from .cleaning import limpiar_texto
from .extractors import (
    extraer_texto,
    extract_pdf_paginas,
)

from .resolvers import (
    resolver_archivo_json,
)

# ---------------------------------------------------------
# Umbrales
# ---------------------------------------------------------

MIN_CARACTERES = 200

MIN_CARACTERES_PAGINA = 30

MIN_RATIO_LETRAS = 0.40

# ---------------------------------------------------------
# Categorías
# ---------------------------------------------------------

DIAG_OK = "ok"

DIAG_JSON = "json"

DIAG_PDF = "pdf"

DIAG_HTML = "html"

DIAG_IMAGEN = "imagen"

DIAG_EXCEL = "excel"

DIAG_TXT = "txt"

DIAG_ERROR = "error"

DIAG_DESCONOCIDO = "desconocido"


class EstadoDiagnostico(Enum):
    OK = "ok"
    warning = "warning"
    Error = "error"

@dataclass(slots=True)
class DiagnosticoResultado:
    archivo: Path
    estado: EstadoDiagnostico
    categoria: str
    mensaje: str
    detalles: dict | None = None


@dataclass(slots=True)
class DiagnosticoResumen:
    total: int = 0
    ok: int = 0
    warning: int = 0
    error: int = 0
    resultados: list[DiagnosticoResultado] = field(default_factory=list)


# ---------------------------------------------------------
# Resultado individual
# ---------------------------------------------------------


@dataclass
class Diagnostico:

    archivo: str

    categoria: str

    estrategia: str

    valido: bool

    titulo: str | None

    caracteres: int

    paginas: int = 0

    paginas_vacias: int = 0

    paginas_con_texto: int = 0

    motivo: str = ""

    detalles: list[str] = field(default_factory=list)

    excepcion: str | None = None

    hash_archivo: str | None = None

# ---------------------------------------------------------
# Estadísticas
# ---------------------------------------------------------


@dataclass
class Estadisticas:

    total: int = 0

    validos: int = 0

    invalidos: int = 0

    pdfs: int = 0

    jsons: int = 0

    html: int = 0

    imagenes: int = 0

    excels: int = 0

    txt: int = 0

    errores: int = 0

    manifiestos: int = 0

    catalogos: int = 0

    desconocidos: int = 0

    pdf_corruptos: int = 0

    pdf_vacios: int = 0

    pdf_ocr: int = 0

    json_desconocidos: int = 0

# ---------------------------------------------------------
# Utilidades
# ---------------------------------------------------------


def calcular_hash(path: Path) -> str:
    """
    Hash SHA256 del archivo.

    Sirve para detectar duplicados.
    """

    sha = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            bloque = f.read(1024 * 1024)

            if not bloque:
                break

            sha.update(bloque)

    return sha.hexdigest()


def contar_letras(texto: str) -> int:

    return sum(

        1

        for c in texto

        if c.isalpha()

    )


def ratio_letras(texto: str) -> float:

    if not texto:

        return 0.0

    return contar_letras(texto) / len(texto)


def registrar_error(exc: Exception) -> str:

    return "".join(

        traceback.format_exception_only(

            type(exc),

            exc,

        )

    ).strip()


def limpiar(texto: Any) -> str:

    if texto is None:

        return ""

    return limpiar_texto(str(texto))


def longitud(texto: str) -> int:

    return len(texto)


def extension(path: Path) -> str:

    return path.suffix.lower()


def es_pdf(path: Path) -> bool:

    return extension(path) == ".pdf"


def es_json(path: Path) -> bool:

    return extension(path) == ".json"


def es_excel(path: Path) -> bool:

    return extension(path) in {

        ".xlsx",

        ".xls",

    }


def es_html(path: Path) -> bool:

    return extension(path) in {

        ".html",

        ".htm",

    }


def es_imagen(path: Path) -> bool:

    return extension(path) in {

        ".jpg",

        ".jpeg",

        ".png",

        ".tif",

        ".tiff",

        ".bmp",

        ".webp",

    }


def es_txt(path: Path) -> bool:

    return extension(path) in {

        ".txt",

        ".csv",

        ".md",

    }


def cargar_json(path: Path) -> Any:

    return json.loads(

        path.read_text(

            encoding="utf-8",

            errors="ignore",

        )

    )
    
# ---------------------------------------------------------
# Diagnóstico JSON
# ---------------------------------------------------------

def _diagnosticar_json_lista(
    data: list[Any],
    path: Path,
) -> Diagnostico:
    """
    Diagnóstico para JSON cuya raíz es una lista.

    Muchos catálogos y manifiestos utilizan esta estructura.
    """

    detalles: list[str] = []

    total = len(data)

    detalles.append(f"La raíz del JSON es una lista con {total} elementos.")

    if total == 0:
        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_JSON,
            estrategia="lista",
            valido=False,
            titulo=None,
            caracteres=0,
            motivo="Lista JSON vacía.",
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    tipos: dict[str, int] = {}

    for item in data:
        nombre = type(item).__name__
        tipos[nombre] = tipos.get(nombre, 0) + 1

    detalles.append(f"Tipos encontrados: {tipos}")

    if all(isinstance(item, dict) for item in data):

        claves: dict[str, int] = {}

        for registro in data:
            for clave in registro.keys():
                claves[clave] = claves.get(clave, 0) + 1

        principales = sorted(
            claves.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:20]

        detalles.append("Campos más frecuentes:")

        for nombre, frecuencia in principales:
            detalles.append(
                f"  {nombre}: {frecuencia}"
            )

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_JSON,
            estrategia="lista_dict",
            valido=False,
            titulo=None,
            caracteres=0,
            motivo=(
                "El JSON raíz es una lista de registros. "
                "Debe existir un extractor específico."
            ),
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    return Diagnostico(
        archivo=str(path),
        categoria=DIAG_JSON,
        estrategia="lista",
        valido=False,
        titulo=None,
        caracteres=0,
        motivo="Lista JSON con estructura desconocida.",
        detalles=detalles,
        hash_archivo=calcular_hash(path),
    )


def _diagnosticar_json_dict(
    data: dict[str, Any],
    path: Path,
) -> Diagnostico:
    """
    Diagnóstico para JSON clásico.
    """

    resolucion = resolver_archivo_json(path)

    detalles: list[str] = []

    detalles.append(
        f"Estrategia detectada: {resolucion.estrategia}"
    )

    detalles.append(
        f"Razón: {resolucion.razon}"
    )

    detalles.append(
        f"Total claves: {len(data)}"
    )

    detalles.append(
        "Claves:"
    )

    for clave in sorted(data.keys()):
        detalles.append(f"  - {clave}")

    try:

        texto = extraer_texto(path)

    except Exception as exc:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_JSON,
            estrategia=resolucion.estrategia,
            valido=False,
            titulo=resolucion.titulo,
            caracteres=0,
            motivo="Error durante extracción.",
            detalles=detalles,
            excepcion=registrar_error(exc),
            hash_archivo=calcular_hash(path),
        )

    texto = limpiar(texto)

    caracteres = longitud(texto)

    detalles.append(
        f"Caracteres extraídos: {caracteres}"
    )

    detalles.append(
        f"Ratio letras: {ratio_letras(texto):.3f}"
    )

    if caracteres == 0:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_JSON,
            estrategia=resolucion.estrategia,
            valido=False,
            titulo=resolucion.titulo,
            caracteres=0,
            motivo="No fue posible extraer texto.",
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    if caracteres < MIN_CARACTERES:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_JSON,
            estrategia=resolucion.estrategia,
            valido=False,
            titulo=resolucion.titulo,
            caracteres=caracteres,
            motivo="Texto demasiado corto.",
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    if ratio_letras(texto) < MIN_RATIO_LETRAS:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_JSON,
            estrategia=resolucion.estrategia,
            valido=False,
            titulo=resolucion.titulo,
            caracteres=caracteres,
            motivo="El texto parece contener demasiado ruido.",
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    return Diagnostico(
        archivo=str(path),
        categoria=DIAG_JSON,
        estrategia=resolucion.estrategia,
        valido=True,
        titulo=resolucion.titulo,
        caracteres=caracteres,
        motivo="Extracción correcta.",
        detalles=detalles,
        hash_archivo=calcular_hash(path),
    )


def diagnosticar_json(
    path: Path,
) -> Diagnostico:
    """
    Punto de entrada para cualquier JSON.
    """

    try:

        data = cargar_json(path)

    except Exception as exc:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_JSON,
            estrategia="error",
            valido=False,
            titulo=None,
            caracteres=0,
            motivo="No fue posible leer el JSON.",
            excepcion=registrar_error(exc),
            hash_archivo=calcular_hash(path),
        )

    if isinstance(data, dict):
        return _diagnosticar_json_dict(
            data,
            path,
        )

    if isinstance(data, list):
        return _diagnosticar_json_lista(
            data,
            path,
        )

    return Diagnostico(
        archivo=str(path),
        categoria=DIAG_JSON,
        estrategia="desconocida",
        valido=False,
        titulo=None,
        caracteres=0,
        motivo=(
            "La raíz del JSON no es dict ni list."
        ),
        detalles=[
            f"Tipo encontrado: {type(data).__name__}"
        ],
        hash_archivo=calcular_hash(path),
    )
    
    # ---------------------------------------------------------
# Diagnóstico PDF
# ---------------------------------------------------------

def diagnosticar_pdf(
    path: Path,
) -> Diagnostico:
    """
    Diagnóstico completo de un PDF.

    Analiza:

    - número de páginas
    - páginas vacías
    - páginas con texto
    - cantidad total de texto
    - calidad del texto
    - posible necesidad de OCR
    """

    detalles: list[str] = []

    try:

        paginas = extract_pdf_paginas(path)

    except Exception as exc:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_PDF,
            estrategia="pdf",
            valido=False,
            titulo=None,
            caracteres=0,
            motivo="No fue posible abrir el PDF.",
            excepcion=registrar_error(exc),
            hash_archivo=calcular_hash(path),
        )

    total_paginas = len(paginas)

    texto_total: list[str] = []

    paginas_vacias = 0
    paginas_con_texto = 0
    paginas_ruido = 0

    detalles.append(f"Total páginas: {total_paginas}")

    for numero, pagina in enumerate(paginas, start=1):

        texto = limpiar(pagina)

        caracteres = len(texto)

        if caracteres == 0:

            paginas_vacias += 1

            detalles.append(
                f"Página {numero}: vacía"
            )

            continue

        paginas_con_texto += 1

        if caracteres < MIN_CARACTERES_PAGINA:

            paginas_ruido += 1

            detalles.append(
                f"Página {numero}: solamente {caracteres} caracteres"
            )

        else:

            detalles.append(
                f"Página {numero}: {caracteres} caracteres"
            )

        texto_total.append(texto)

    texto = "\n\n".join(texto_total)

    caracteres = len(texto)

    ratio = ratio_letras(texto)

    detalles.append(
        f"Caracteres totales: {caracteres}"
    )

    detalles.append(
        f"Ratio letras: {ratio:.3f}"
    )

    # --------------------------------------------------
    # PDF completamente vacío
    # --------------------------------------------------

    if paginas_con_texto == 0:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_PDF,
            estrategia="pdf",
            valido=False,
            titulo=None,
            caracteres=0,
            paginas=total_paginas,
            paginas_vacias=paginas_vacias,
            paginas_con_texto=0,
            motivo=(
                "No se pudo extraer texto. "
                "Probablemente es un PDF escaneado o corrupto."
            ),
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    # --------------------------------------------------
    # PDF muy corto
    # --------------------------------------------------

    if caracteres < MIN_CARACTERES:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_PDF,
            estrategia="pdf",
            valido=False,
            titulo=None,
            caracteres=caracteres,
            paginas=total_paginas,
            paginas_vacias=paginas_vacias,
            paginas_con_texto=paginas_con_texto,
            motivo="Texto insuficiente.",
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    # --------------------------------------------------
    # Mucho ruido
    # --------------------------------------------------

    if ratio < MIN_RATIO_LETRAS:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_PDF,
            estrategia="pdf",
            valido=False,
            titulo=None,
            caracteres=caracteres,
            paginas=total_paginas,
            paginas_vacias=paginas_vacias,
            paginas_con_texto=paginas_con_texto,
            motivo=(
                "El texto contiene demasiado ruido. "
                "Es probable que requiera OCR."
            ),
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    # --------------------------------------------------
    # Muchas páginas vacías
    # --------------------------------------------------

    if paginas_vacias > total_paginas * 0.50:

        detalles.append(
            "Más del 50% de las páginas no contienen texto."
        )

    # --------------------------------------------------
    # Muchas páginas con poco texto
    # --------------------------------------------------

    if paginas_ruido > total_paginas * 0.50:

        detalles.append(
            "Más del 50% de las páginas contienen muy poco texto."
        )

    # --------------------------------------------------
    # PDF correcto
    # --------------------------------------------------

    return Diagnostico(
        archivo=str(path),
        categoria=DIAG_PDF,
        estrategia="pdf",
        valido=True,
        titulo=None,
        caracteres=caracteres,
        paginas=total_paginas,
        paginas_vacias=paginas_vacias,
        paginas_con_texto=paginas_con_texto,
        motivo="Extracción correcta.",
        detalles=detalles,
        hash_archivo=calcular_hash(path),
    )
    
    # ---------------------------------------------------------
# Diagnóstico HTML
# ---------------------------------------------------------

def diagnosticar_html(path: Path) -> Diagnostico:
    """
    Diagnóstico para recursos HTML/HTM.
    """

    detalles: list[str] = []

    try:
        texto = extraer_texto(path)

    except Exception as exc:
        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_HTML,
            estrategia="html",
            valido=False,
            titulo=None,
            caracteres=0,
            motivo="Error extrayendo HTML.",
            excepcion=registrar_error(exc),
            hash_archivo=calcular_hash(path),
        )

    texto = limpiar(texto)
    caracteres = len(texto)

    detalles.append(f"Caracteres: {caracteres}")
    detalles.append(f"Ratio letras: {ratio_letras(texto):.3f}")

    if caracteres == 0:
        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_HTML,
            estrategia="html",
            valido=False,
            titulo=None,
            caracteres=0,
            motivo="HTML vacío.",
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    if caracteres < MIN_CARACTERES:
        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_HTML,
            estrategia="html",
            valido=False,
            titulo=None,
            caracteres=caracteres,
            motivo="Contenido demasiado corto.",
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    return Diagnostico(
        archivo=str(path),
        categoria=DIAG_HTML,
        estrategia="html",
        valido=True,
        titulo=None,
        caracteres=caracteres,
        motivo="Extracción correcta.",
        detalles=detalles,
        hash_archivo=calcular_hash(path),
    )


# ---------------------------------------------------------
# Diagnóstico Imágenes
# ---------------------------------------------------------

def diagnosticar_imagen(path: Path) -> Diagnostico:
    """
    Diagnóstico para imágenes (OCR).
    """

    detalles: list[str] = []

    try:
        texto = extraer_texto(path)

    except Exception as exc:
        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_IMAGEN,
            estrategia="ocr",
            valido=False,
            titulo=None,
            caracteres=0,
            motivo="Error ejecutando OCR.",
            excepcion=registrar_error(exc),
            hash_archivo=calcular_hash(path),
        )

    texto = limpiar(texto)

    caracteres = len(texto)

    detalles.append(f"Caracteres OCR: {caracteres}")

    if caracteres == 0:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_IMAGEN,
            estrategia="ocr",
            valido=False,
            titulo=None,
            caracteres=0,
            motivo="OCR no produjo texto.",
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    if caracteres < MIN_CARACTERES:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_IMAGEN,
            estrategia="ocr",
            valido=False,
            titulo=None,
            caracteres=caracteres,
            motivo="OCR produjo muy poco texto.",
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    return Diagnostico(
        archivo=str(path),
        categoria=DIAG_IMAGEN,
        estrategia="ocr",
        valido=True,
        titulo=None,
        caracteres=caracteres,
        motivo="OCR correcto.",
        detalles=detalles,
        hash_archivo=calcular_hash(path),
    )


# ---------------------------------------------------------
# Diagnóstico Excel
# ---------------------------------------------------------

def diagnosticar_excel(path: Path) -> Diagnostico:
    """
    Diagnóstico para hojas de cálculo.
    """

    detalles: list[str] = []

    try:
        texto = extraer_texto(path)

    except Exception as exc:
        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_EXCEL,
            estrategia="excel",
            valido=False,
            titulo=None,
            caracteres=0,
            motivo="Error leyendo Excel.",
            excepcion=registrar_error(exc),
            hash_archivo=calcular_hash(path),
        )

    texto = limpiar(texto)

    caracteres = len(texto)

    detalles.append(f"Caracteres: {caracteres}")

    if caracteres == 0:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_EXCEL,
            estrategia="excel",
            valido=False,
            titulo=None,
            caracteres=0,
            motivo="Excel sin contenido textual.",
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    if caracteres < MIN_CARACTERES:

        detalles.append(
            "Puede tratarse de un dataset numérico."
        )

    return Diagnostico(
        archivo=str(path),
        categoria=DIAG_EXCEL,
        estrategia="excel",
        valido=True,
        titulo=None,
        caracteres=caracteres,
        motivo="Lectura correcta.",
        detalles=detalles,
        hash_archivo=calcular_hash(path),
    )


# ---------------------------------------------------------
# Diagnóstico TXT / CSV / MD
# ---------------------------------------------------------

def diagnosticar_texto(path: Path) -> Diagnostico:

    detalles: list[str] = []

    try:
        texto = extraer_texto(path)

    except Exception as exc:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_TXT,
            estrategia="texto",
            valido=False,
            titulo=None,
            caracteres=0,
            motivo="Error leyendo archivo.",
            excepcion=registrar_error(exc),
            hash_archivo=calcular_hash(path),
        )

    texto = limpiar(texto)

    caracteres = len(texto)

    detalles.append(f"Caracteres: {caracteres}")

    if caracteres == 0:

        return Diagnostico(
            archivo=str(path),
            categoria=DIAG_TXT,
            estrategia="texto",
            valido=False,
            titulo=None,
            caracteres=0,
            motivo="Archivo vacío.",
            detalles=detalles,
            hash_archivo=calcular_hash(path),
        )

    return Diagnostico(
        archivo=str(path),
        categoria=DIAG_TXT,
        estrategia="texto",
        valido=True,
        titulo=None,
        caracteres=caracteres,
        motivo="Lectura correcta.",
        detalles=detalles,
        hash_archivo=calcular_hash(path),
    )


# ---------------------------------------------------------
# Dispatcher principal
# ---------------------------------------------------------

def diagnosticar_archivo(path: Path) -> Diagnostico:
    """
    Selecciona automáticamente el diagnóstico adecuado.
    """

    if es_json(path):
        return diagnosticar_json(path)

    if es_pdf(path):
        return diagnosticar_pdf(path)

    if es_html(path):
        return diagnosticar_html(path)

    if es_excel(path):
        return diagnosticar_excel(path)

    if es_imagen(path):
        return diagnosticar_imagen(path)

    if es_txt(path):
        return diagnosticar_texto(path)

    return Diagnostico(
        archivo=str(path),
        categoria=DIAG_DESCONOCIDO,
        estrategia="desconocida",
        valido=False,
        titulo=None,
        caracteres=0,
        motivo=f"Extensión no soportada: {path.suffix}",
        hash_archivo=calcular_hash(path),
    )
    
    # ---------------------------------------------------------
# Recorrido del corpus
# ---------------------------------------------------------

EXTENSIONES_SOPORTADAS = {
    ".json",
    ".pdf",
    ".html",
    ".htm",
    ".txt",
    ".csv",
    ".md",
    ".xlsx",
    ".xls",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def diagnosticar_corpus(
    corpus: Path,
) -> list[Diagnostico]:
    """
    Ejecuta el diagnóstico completo sobre un corpus.
    """

    archivos = sorted(
        p
        for p in corpus.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSIONES_SOPORTADAS
    )

    total = len(archivos)

    resultados: list[Diagnostico] = []

    for indice, archivo in enumerate(archivos, start=1):

        print(
            f"\rDiagnosticando {indice}/{total}...",
            end="",
            flush=True,
        )

        resultado = diagnosticar_archivo(
            archivo
        )

        resultados.append(resultado)

    print()

    return resultados


# ---------------------------------------------------------
# Estadísticas
# ---------------------------------------------------------

def generar_estadisticas(
    resultados: list[Diagnostico],
) -> Estadisticas:

    stats = Estadisticas()

    stats.total = len(resultados)

    for r in resultados:

        if r.valido:
            stats.validos += 1
        else:
            stats.invalidos += 1

        if r.categoria == DIAG_JSON:
            stats.jsons += 1

        elif r.categoria == DIAG_PDF:
            stats.pdfs += 1

        elif r.categoria == DIAG_HTML:
            stats.html += 1

        elif r.categoria == DIAG_IMAGEN:
            stats.imagenes += 1

        elif r.categoria == DIAG_EXCEL:
            stats.excels += 1

        elif r.categoria == DIAG_TXT:
            stats.txt += 1

        elif r.categoria == DIAG_ERROR:
            stats.errores += 1

        elif r.categoria == DIAG_DESCONOCIDO:
            stats.desconocidos += 1

        # --------------------------
        # Casos especiales
        # --------------------------

        motivo = r.motivo.lower()

        if "manifiesto" in motivo:
            stats.manifiestos += 1

        if "catálogo" in motivo:
            stats.catalogos += 1

        if "catalogo" in motivo:
            stats.catalogos += 1

        if "ocr" in motivo:
            stats.pdf_ocr += 1

        if "corrupto" in motivo:
            stats.pdf_corruptos += 1

        if (
            r.categoria == DIAG_PDF
            and r.caracteres == 0
        ):
            stats.pdf_vacios += 1

        if (
            r.categoria == DIAG_JSON
            and r.estrategia == "desconocida"
        ):
            stats.json_desconocidos += 1

    return stats


# ---------------------------------------------------------
# Estadísticas por estrategia
# ---------------------------------------------------------

def contar_por_estrategia(
    resultados: list[Diagnostico],
) -> dict[str, int]:

    conteo: dict[str, int] = {}

    for r in resultados:

        conteo[r.estrategia] = (
            conteo.get(
                r.estrategia,
                0,
            )
            + 1
        )

    return dict(
        sorted(
            conteo.items(),
            key=lambda x: x[0],
        )
    )


# ---------------------------------------------------------
# Estadísticas por categoría
# ---------------------------------------------------------

def contar_por_categoria(
    resultados: list[Diagnostico],
) -> dict[str, int]:

    conteo: dict[str, int] = {}

    for r in resultados:

        conteo[r.categoria] = (
            conteo.get(
                r.categoria,
                0,
            )
            + 1
        )

    return dict(
        sorted(
            conteo.items(),
            key=lambda x: x[0],
        )
    )


# ---------------------------------------------------------
# Archivos inválidos
# ---------------------------------------------------------

def obtener_invalidos(
    resultados: list[Diagnostico],
) -> list[Diagnostico]:

    return [
        r
        for r in resultados
        if not r.valido
    ]


# ---------------------------------------------------------
# Archivos válidos
# ---------------------------------------------------------

def obtener_validos(
    resultados: list[Diagnostico],
) -> list[Diagnostico]:

    return [
        r
        for r in resultados
        if r.valido
    ]


# ---------------------------------------------------------
# Ordenar por tamaño
# ---------------------------------------------------------

def ordenar_por_caracteres(
    resultados: list[Diagnostico],
) -> list[Diagnostico]:

    return sorted(
        resultados,
        key=lambda r: r.caracteres,
        reverse=True,
    )


# ---------------------------------------------------------
# Buscar duplicados
# ---------------------------------------------------------

def buscar_duplicados(
    resultados: list[Diagnostico],
) -> dict[str, list[Diagnostico]]:

    hashes: dict[
        str,
        list[Diagnostico],
    ] = {}

    for r in resultados:

        if r.hash_archivo is None:
            continue

        hashes.setdefault(
            r.hash_archivo,
            [],
        ).append(r)

    return {
        h: archivos
        for h, archivos in hashes.items()
        if len(archivos) > 1
    }
    
    # ---------------------------------------------------------------------
# Generación del resumen final
# ---------------------------------------------------------------------

def generar_resumen(
    resultados: list[DiagnosticoResultado],
) -> DiagnosticoResumen:
    """
    Consolida todos los diagnósticos obtenidos durante la validación.
    """

    resumen = DiagnosticoResumen()
    resumen.total = len(resultados)
    resumen.resultados = resultados

    for resultado in resultados:
        if resultado.estado == EstadoDiagnostico.OK:
            resumen.ok += 1
        elif resultado.estado == EstadoDiagnostico.warning:
            resumen.warning += 1
        else:
            resumen.error += 1

    return resumen


# ---------------------------------------------------------------------
# Impresión amigable
# ---------------------------------------------------------------------

def imprimir_resumen(
    resumen: DiagnosticoResumen,
) -> None:

    print()
    print("=" * 90)
    print("DIAGNÓSTICO GENERAL DEL CORPUS")
    print("=" * 90)

    print(f"Total archivos : {resumen.total}")
    print(f"OK             : {resumen.ok}")
    print(f"Warning        : {resumen.warning}")
    print(f"Error          : {resumen.error}")

    if not resumen.resultados:
        return

    por_categoria: dict[str, int] = {}
    mensajes: dict[str, int] = {}

    for r in resumen.resultados:
        por_categoria[r.categoria] = por_categoria.get(r.categoria, 0) + 1
        if r.estado != EstadoDiagnostico.OK:
            mensajes[r.mensaje] = mensajes.get(r.mensaje, 0) + 1

    print()
    print("Categorías")
    print("-" * 90)

    for categoria, cantidad in sorted(por_categoria.items()):
        print(f"{categoria:30} {cantidad}")

    print()
    print("Mensajes más frecuentes (no-OK)")
    print("-" * 90)

    for mensaje, cantidad in sorted(
        mensajes.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{cantidad:5}  {mensaje}")
    # ------------------------------------------------------------------
# Exploración profunda de JSON desconocidos
# ------------------------------------------------------------------

def analizar_json_desconocido(
    path: Path,
) -> dict[str, Any]:
    """
    Analiza un JSON que no pudo clasificarse correctamente.

    No intenta extraer texto.

    Sólo produce información para que podamos ampliar
    resolver_json() y extractors.py.
    """

    try:
        contenido = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )

    except Exception as exc:

        return {
            "archivo": str(path),
            "tipo": "json_invalido",
            "error": str(exc),
        }

    resultado = {
        "archivo": str(path),
        "tipo_python": type(contenido).__name__,
        "claves": [],
        "listas": [],
        "diccionarios": [],
        "strings_largos": [],
        "urls": [],
    }

    # --------------------------------------------------------------
    # Si el JSON raíz NO es un diccionario.
    # --------------------------------------------------------------

    if not isinstance(contenido, dict):

        resultado["tipo"] = "estructura_no_dict"

        if isinstance(contenido, list):
            resultado["longitud_lista"] = len(contenido)

            if contenido:

                primer = contenido[0]

                resultado["primer_elemento"] = (
                    type(primer).__name__
                )

                if isinstance(primer, dict):
                    resultado["claves_primer"] = sorted(
                        primer.keys()
                    )

        return resultado

    resultado["tipo"] = "dict"

    # --------------------------------------------------------------
    # Exploración de cada clave
    # --------------------------------------------------------------

    for clave, valor in contenido.items():

        resultado["claves"].append(clave)

        if isinstance(valor, dict):

            resultado["diccionarios"].append(
                {
                    "campo": clave,
                    "claves": sorted(valor.keys()),
                }
            )

        elif isinstance(valor, list):

            descripcion = {
                "campo": clave,
                "longitud": len(valor),
            }

            if valor:

                descripcion["primer_tipo"] = (
                    type(valor[0]).__name__
                )

            resultado["listas"].append(
                descripcion
            )

        elif isinstance(valor, str):

            texto = valor.strip()

            if len(texto) > 300:

                resultado["strings_largos"].append(
                    {
                        "campo": clave,
                        "longitud": len(texto),
                    }
                )

            if texto.startswith("http://") or texto.startswith("https://"):

                resultado["urls"].append(
                    {
                        "campo": clave,
                        "url": texto,
                    }
                )

    resultado["claves"] = sorted(resultado["claves"])

    return resultado