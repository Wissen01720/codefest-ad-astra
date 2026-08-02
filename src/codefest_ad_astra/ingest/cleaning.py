"""Limpieza y normalización de texto extraído (Fase 2)."""
import re
import unicodedata
from dataclasses import dataclass

from langdetect import detect, LangDetectException
from langdetect import DetectorFactory
DetectorFactory.seed = 0


_ESPACIOS_MULTIPLES = re.compile(r"[ \t]+")
_LINEAS_VACIAS_MULTIPLES = re.compile(r"\n{3,}")
_CARACTERES_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_GUION_CORTE = re.compile(r"(\w)-\n(\w)")
_SALTO_LINEA_SIMPLE = re.compile(r"(?<!\n)\n(?!\n)")
_PUNTOS_SUSPENSIVOS_INDICE = re.compile(r"(?:\.\s?){4,}")


def normalizar_saltos_pdf(texto: str) -> str:
    """Específico para texto extraído de PDF con `pdfplumber`, que preserva
    los saltos de línea visuales de la página (texto justificado).

    Hace dos cosas:
    1. Une palabras partidas por guión de fin de línea: 'innova-\\nción' -> 'innovación'
    2. Convierte saltos de línea simples (wrap visual dentro de un párrafo) en
       espacios, conservando los saltos dobles (línea en blanco) como separador
       real de párrafo.

    Aplícalo ANTES de limpiar_texto(), solo para documentos de formato PDF.
    """
    texto = _GUION_CORTE.sub(r"\1\2", texto)
    texto = _SALTO_LINEA_SIMPLE.sub(" ", texto)
    texto = _PUNTOS_SUSPENSIVOS_INDICE.sub(" ", texto)
    return texto


def limpiar_texto(texto: str) -> str:
    """Limpieza básica: normaliza codificación (UTF-8/NFC), quita caracteres
    de control y colapsa espacios/saltos de línea redundantes."""
    texto = unicodedata.normalize("NFC", texto)
    texto = _CARACTERES_CONTROL.sub("", texto)
    texto = _ESPACIOS_MULTIPLES.sub(" ", texto)
    texto = _LINEAS_VACIAS_MULTIPLES.sub("\n\n", texto)
    return texto.strip()


def detectar_idioma(texto: str) -> str:
    """Detecta el idioma predominante del documento (es/en/pt/...).

    Devuelve 'und' (indeterminado) si el texto es muy corto o falla la detección.
    """
    muestra = texto[:2000]  # suficiente para detectar, más rápido que el texto completo
    if len(muestra.strip()) < 20:
        return "und"
    try:
        return detect(muestra)
    except LangDetectException:
        return "und"


def quitar_lineas_repetidas(paginas: list[str], umbral: float = 0.6) -> list[str]:
    """Elimina líneas que se repiten en más del `umbral` de las páginas de un
    mismo documento (headers/footers/numeración de página tipo boilerplate).

    Úsalo opcionalmente en documentos con muchas páginas (PDFs largos) ANTES
    de unir todo en un solo texto, pasando la lista de textos por página.
    """
    if len(paginas) < 3:
        return paginas

    conteo: dict[str, int] = {}
    for pagina in paginas:
        for linea in set(l.strip() for l in pagina.split("\n") if l.strip()):
            conteo[linea] = conteo.get(linea, 0) + 1

    limite = umbral * len(paginas)
    lineas_boilerplate = {linea for linea, n in conteo.items() if n >= limite}

    resultado = []
    for pagina in paginas:
        lineas_filtradas = [
            l for l in pagina.split("\n") if l.strip() not in lineas_boilerplate
        ]
        resultado.append("\n".join(lineas_filtradas))
    return resultado


@dataclass(slots=True)
class DocumentoLimpio:
    """Salida única para el equipo de chunking: texto listo + metadata."""

    texto: str
    idioma: str
    paginas_originales: int
    lineas_boilerplate_removidas: bool


def procesar_documento(
    paginas: list[str],
    es_pdf: bool,
    quitar_boilerplate: bool | None = None,
) -> DocumentoLimpio:
    """Punto de entrada único para el equipo de chunking: recibe el texto
    crudo por página y devuelve el texto ya limpio, normalizado y con
    idioma detectado. Aplica internamente el orden correcto de las
    funciones de arriba, para que quien lo use no necesite conocerlo.

    Parameters
    ----------
    paginas: texto crudo de cada página, en el orden del documento original.
             Para fuentes no paginadas (HTML, TXT, JSON), pasar una lista
             de un solo elemento con el texto completo.
    es_pdf: si True, aplica normalizar_saltos_pdf antes de la limpieza básica
            (dehyphenation + colapso de saltos de línea simples).
    quitar_boilerplate: si None, se activa automáticamente cuando hay 3+
            páginas (el mínimo que ya exige quitar_lineas_repetidas).
            Pásalo explícitamente en False para desactivarlo siempre.
    """
    if quitar_boilerplate is None:
        quitar_boilerplate = len(paginas) >= 3

    paginas_trabajo = (
        quitar_lineas_repetidas(paginas) if quitar_boilerplate else paginas
    )

    texto_unido = "\n\n".join(paginas_trabajo)

    if es_pdf:
        texto_unido = normalizar_saltos_pdf(texto_unido)

    texto_final = limpiar_texto(texto_unido)
    idioma = detectar_idioma(texto_final)

    return DocumentoLimpio(
        texto=texto_final,
        idioma=idioma,
        paginas_originales=len(paginas),
        lineas_boilerplate_removidas=quitar_boilerplate,
    )