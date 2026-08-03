"""Divisor de oraciones para ES/EN/PT sobre texto ya limpio (Fase 2).

No usa un modelo de NLP: es una heurística basada en puntuación, con una
lista corta de abreviaturas comunes y protección de números decimales y
puntos suspensivos, para evitar el error más común de un split ingenuo
(cortar 'Dr. Pérez' o '3.14' en dos oraciones).
"""
import re

_MARCADOR = "\x00"

_ABREVIATURAS = (
    "sr", "sra", "srta", "dr", "dra", "ing", "lic", "prof",
    "av", "no", "art", "pp", "ed", "fig", "ref", "vs", "etc",
    "ee", "uu", "eu", "cap", "vol", "num", "núm",
)

_PATRON_ABREVIATURA = re.compile(
    r"\b(?:" + "|".join(_ABREVIATURAS) + r")\.",
    re.IGNORECASE,
)
_PATRON_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_PATRON_ELIPSIS = re.compile(r"\.\.\.")
_PATRON_CORTE = re.compile(
    r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑÀ-Ý0-9¿¡"“‘—(])'
)


def split_sentences(text: str) -> list[str]:
    """Divide `text` en oraciones completas. Nunca corta a mitad de una
    abreviatura conocida, un número decimal o unos puntos suspensivos."""
    text = text.strip()
    if not text:
        return []

    protegido = _PATRON_ABREVIATURA.sub(lambda m: m.group(0).replace(".", _MARCADOR), text)
    protegido = _PATRON_DECIMAL.sub(_MARCADOR, protegido)
    protegido = _PATRON_ELIPSIS.sub(_MARCADOR * 3, protegido)

    partes = _PATRON_CORTE.split(protegido)

    return [
        p.replace(_MARCADOR, ".").strip()
        for p in partes
        if p.strip()
    ]
