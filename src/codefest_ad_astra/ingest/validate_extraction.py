"""Validación de los recursos después de extracción y limpieza.

Este módulo NO modifica el pipeline.
Sirve para diagnosticar qué recursos del corpus producen texto
válido antes de pasar a la Fase 3 (chunking).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cleaning import limpiar_texto
from .extractors import extraer_texto, extract_pdf_paginas
from .resolvers import resolver_archivo_json


MIN_CARACTERES = 200


@dataclass
class ResultadoExtraccion:
    archivo: str
    estrategia: str
    titulo: str | None
    caracteres: int
    valido: bool
    motivo: str


def validar_json(path: Path) -> ResultadoExtraccion:
    """Resuelve, extrae y valida un JSON individual."""

    try:
        resolucion = resolver_archivo_json(path)
    except Exception as exc:
        return ResultadoExtraccion(
            archivo=str(path),
            estrategia="error",
            titulo=None,
            caracteres=0,
            valido=False,
            motivo=f"Error resolviendo JSON: {exc}",
        )

    estrategia = getattr(resolucion, "estrategia", "desconocida")
    titulo = getattr(resolucion, "titulo", None)

    # Los manifiestos/catálogos no son documentos para chunking.
    if estrategia == "manifiesto":
        return ResultadoExtraccion(
            archivo=str(path),
            estrategia=estrategia,
            titulo=titulo,
            caracteres=0,
            valido=False,
            motivo="Manifiesto o catálogo; no es contenido documental.",
        )

    try:
        texto = extraer_texto(path)
    except Exception as exc:
        return ResultadoExtraccion(
            archivo=str(path),
            estrategia=estrategia,
            titulo=titulo,
            caracteres=0,
            valido=False,
            motivo=f"Error extrayendo texto: {exc}",
        )

    if texto is None:
        return ResultadoExtraccion(
            archivo=str(path),
            estrategia=estrategia,
            titulo=titulo,
            caracteres=0,
            valido=False,
            motivo="El extractor no devolvió texto.",
        )

    texto_limpio = limpiar_texto(str(texto))
    caracteres = len(texto_limpio)

    if caracteres == 0:
        return ResultadoExtraccion(
            archivo=str(path),
            estrategia=estrategia,
            titulo=titulo,
            caracteres=0,
            valido=False,
            motivo="El texto quedó vacío después de limpieza.",
        )

    if caracteres < MIN_CARACTERES:
        return ResultadoExtraccion(
            archivo=str(path),
            estrategia=estrategia,
            titulo=titulo,
            caracteres=caracteres,
            valido=False,
            motivo=(
                f"Texto demasiado corto después de limpieza "
                f"(< {MIN_CARACTERES} caracteres)."
            ),
        )

    return ResultadoExtraccion(
        archivo=str(path),
        estrategia=estrategia,
        titulo=titulo,
        caracteres=caracteres,
        valido=True,
        motivo="Extracción y limpieza válidas.",
    )


def validar_archivo(path: Path) -> ResultadoExtraccion:
    """Valida un archivo documental según su extensión."""

    if path.suffix.lower() == ".json":
        return validar_json(path)

    try:
        if path.suffix.lower() == ".pdf":
            paginas = extract_pdf_paginas(path)
            texto = "\n\n".join(
                pagina for pagina in paginas if pagina.strip()
            )
        else:
            texto = extraer_texto(path)

        texto_limpio = limpiar_texto(str(texto))
        caracteres = len(texto_limpio)

        if caracteres == 0:
            return ResultadoExtraccion(
                archivo=str(path),
                estrategia="directo",
                titulo=None,
                caracteres=0,
                valido=False,
                motivo="El texto quedó vacío después de limpieza.",
            )

        if caracteres < MIN_CARACTERES:
            return ResultadoExtraccion(
                archivo=str(path),
                estrategia="directo",
                titulo=None,
                caracteres=caracteres,
                valido=False,
                motivo=(
                    f"Texto demasiado corto después de limpieza "
                    f"(< {MIN_CARACTERES} caracteres)."
                ),
            )

        return ResultadoExtraccion(
            archivo=str(path),
            estrategia="directo",
            titulo=None,
            caracteres=caracteres,
            valido=True,
            motivo="Extracción y limpieza válidas.",
        )

    except Exception as exc:
        return ResultadoExtraccion(
            archivo=str(path),
            estrategia="error",
            titulo=None,
            caracteres=0,
            valido=False,
            motivo=f"Error durante extracción: {exc}",
        )


def validar_corpus(corpus: Path) -> list[ResultadoExtraccion]:
    """Valida todos los recursos del corpus."""

    resultados: list[ResultadoExtraccion] = []

    extensiones = {
        ".json",
        ".pdf",
        ".html",
        ".htm",
        ".csv",
        ".xlsx",
        ".txt",
        ".png",
        ".jpg",
        ".jpeg",
        ".pbf",
    }

    archivos = sorted(
        path
        for path in corpus.rglob("*")
        if path.is_file() and path.suffix.lower() in extensiones
    )

    total = len(archivos)

    for numero, path in enumerate(archivos, start=1):
        print(
            f"\rValidando {numero}/{total}...",
            end="",
            flush=True,
        )

        resultado = validar_archivo(path)
        resultados.append(resultado)

    print()

    return resultados


def imprimir_resumen(resultados: list[ResultadoExtraccion]) -> None:
    """Imprime el resumen de la validación."""

    total = len(resultados)
    validos = sum(resultado.valido for resultado in resultados)
    invalidos = total - validos

    print()
    print("=" * 90)
    print("RESUMEN DE VALIDACIÓN DE EXTRACCIÓN")
    print("=" * 90)

    print(f"Total recursos analizados : {total}")
    print(f"Válidos                   : {validos}")
    print(f"No válidos                : {invalidos}")

    print()
    print("POR ESTRATEGIA")
    print("-" * 90)

    estrategias: dict[str, int] = {}

    for resultado in resultados:
        estrategias[resultado.estrategia] = (
            estrategias.get(resultado.estrategia, 0) + 1
        )

    for estrategia, cantidad in sorted(estrategias.items()):
        print(f"{estrategia:25} {cantidad:>6}")

    print()
    print("NO VÁLIDOS")
    print("-" * 90)

    for resultado in resultados:
        if not resultado.valido:
            print()
            print(resultado.archivo)
            print(f"  estrategia : {resultado.estrategia}")
            print(f"  caracteres : {resultado.caracteres}")
            print(f"  motivo     : {resultado.motivo}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Valida extracción y limpieza del corpus."
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Ruta al corpus, por ejemplo data/raw_muestra",
    )

    args = parser.parse_args()

    if not args.corpus.exists():
        raise SystemExit(
            f"No existe el corpus: {args.corpus}"
        )

    resultados = validar_corpus(args.corpus)
    imprimir_resumen(resultados)


if __name__ == "__main__":
    main()