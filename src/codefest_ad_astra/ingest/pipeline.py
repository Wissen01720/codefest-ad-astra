"""Orquestador de las Fases 1 y 2: recorre el corpus, extrae y limpia el texto,
y guarda un documento por línea en JSONL, listo para la Fase 3 (chunking).

Uso:
    uv run python -m codefest_ad_astra.ingest.pipeline --corpus data/raw --salida data/processed/documentos.jsonl

Estructura real del corpus entregado por ADL (carpetas que empiezan con
F1_/F2_/F3_ en cualquier nivel, ej.):
    data/raw/
        F1_IA_y_Capacidades_Estrategicas/Atlantic_Council/GeoTech_Cues/...
        F2_Seguridad_del_Entorno_Espacial/CSIS_Aerospace/...
        F3_Dinamicas_Territoriales/Alertas_Tempranas/...
"""
import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

from .extractors import extraer_texto, extract_pdf_paginas
from .cleaning import limpiar_texto, detectar_idioma, quitar_lineas_repetidas, normalizar_saltos_pdf
from .validation import Document

FORMATO_POR_EXTENSION = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".txt": "texto",
    ".png": "imagen",
    ".jpg": "imagen",
    ".jpeg": "imagen",
    ".pbf": "pbf",
}

# Archivos que parecen manifiestos/catálogos internos del scraping de ADL
# (listas de qué se descargó), no contenido real a indexar. CONFIRMA abriendo
# uno antes de confiar en esto — si resulta ser contenido real, comenta esta
# línea o ajusta el patrón.
_PATRON_MANIFIESTO = re.compile(r"(catalog|registro|tiles-index)", re.IGNORECASE)


def generar_doc_id(ruta_relativa: Path) -> str:
    """doc_id determinístico a partir de la ruta relativa del archivo, para que
    sea estable entre corridas (importante para la reproducibilidad de generador.py)."""
    hash_corto = hashlib.sha1(str(ruta_relativa).encode("utf-8")).hexdigest()[:10]
    return f"DOC-{hash_corto}"


_PATRON_FENOMENO = re.compile(r"^F([1-3])[_-]", re.IGNORECASE)


def inferir_fenomeno(path: Path) -> int:
    """Infiere el número de fenómeno buscando un segmento de carpeta que empiece
    con 'F1_', 'F2_' o 'F3_' (patrón real observado en la entrega de ADL, ej.
    'F1_IA_y_Capacidades_Estrategicas/Atlantic_Council/GeoTech_Cues/...')."""
    for parte in path.parts:
        m = _PATRON_FENOMENO.match(parte)
        if m:
            return int(m.group(1))
    raise ValueError(f"No se pudo inferir el fenómeno para {path}. Revisa la estructura de carpetas.")


def filtrar_pbf_zoom_maximo(archivos_pbf: list[Path]) -> list[Path]:
    """Los .pbf están organizados en pirámides de teselas (tiles/{zoom}/{x}/
    {archivo}.pbf). El mismo elemento geográfico se repite en cada nivel de
    zoom con distinta simplificación — el documento técnico (sección 2.1)
    recomienda quedarse con una sola versión para no duplicar la data. Aquí
    se implementa quedándose solo con el nivel de zoom más alto (más detalle)
    de cada árbol de teselas encontrado en el corpus."""
    raiz_y_zoom = {}
    zooms_por_raiz = defaultdict(set)

    for p in archivos_pbf:
        partes = p.parts
        raiz, zoom = None, None
        if "tiles" in partes:
            idx = partes.index("tiles")
            raiz = partes[: idx + 1]
            try:
                zoom = int(partes[idx + 1])
            except (IndexError, ValueError):
                zoom = None
        raiz_y_zoom[p] = (raiz, zoom)
        if raiz is not None and zoom is not None:
            zooms_por_raiz[raiz].add(zoom)

    zoom_maximo = {raiz: max(zooms) for raiz, zooms in zooms_por_raiz.items()}

    seleccionados = []
    for p in archivos_pbf:
        raiz, zoom = raiz_y_zoom[p]
        if raiz is None or zoom is None:
            seleccionados.append(p)  # no se pudo ubicar en una pirámide de zoom, se procesa igual
        elif zoom == zoom_maximo[raiz]:
            seleccionados.append(p)
    return seleccionados


def procesar_corpus(carpeta_corpus: Path):
    todos = [p for p in carpeta_corpus.rglob("*") if p.suffix.lower() in FORMATO_POR_EXTENSION]
    excluidos = [p for p in todos if _PATRON_MANIFIESTO.search(p.stem)]
    archivos = [p for p in todos if p not in excluidos]

    print(f"Encontrados {len(todos)} archivos procesables en {carpeta_corpus}")
    if excluidos:
        print(f"  [EXCLUIDOS] {len(excluidos)} archivos que parecen manifiestos/catálogos (no contenido):")
        for p in excluidos[:10]:
            print(f"    - {p.relative_to(carpeta_corpus)}")
        if len(excluidos) > 10:
            print(f"    ... y {len(excluidos) - 10} más")

    archivos_pbf = [p for p in archivos if p.suffix.lower() == ".pbf"]
    if archivos_pbf:
        pbf_filtrados = filtrar_pbf_zoom_maximo(archivos_pbf)
        print(f"  [PBF] {len(archivos_pbf)} teselas encontradas, procesando solo el zoom máximo: {len(pbf_filtrados)}")
        archivos = [p for p in archivos if p.suffix.lower() != ".pbf"] + pbf_filtrados

    for path in archivos:
        ruta_relativa = path.relative_to(carpeta_corpus)
        try:
            if path.suffix.lower() == ".pdf":
                # Extracción por página para poder quitar headers/footers repetidos
                paginas = extract_pdf_paginas(path)
                paginas_limpias = quitar_lineas_repetidas(paginas)
                texto_crudo = "\n\n".join(p for p in paginas_limpias if p.strip())
                texto_crudo = normalizar_saltos_pdf(texto_crudo)
            else:
                texto_crudo = extraer_texto(path)
            texto_limpio = limpiar_texto(texto_crudo)
        except Exception as e:
            print(f"  [ERROR] {ruta_relativa}: {e}")
            continue

        if not texto_limpio:
            print(f"  [AVISO] {ruta_relativa} quedó vacío tras la extracción, se omite")
            continue

        doc = Document(
            doc_id=generar_doc_id(ruta_relativa),
            fuente=str(ruta_relativa),
            formato=FORMATO_POR_EXTENSION[path.suffix.lower()],
            fenomeno=inferir_fenomeno(path),
            idioma=detectar_idioma(texto_limpio),
            texto=texto_limpio,
        )
        yield doc


def main():
    parser = argparse.ArgumentParser(description="Fases 1+2: extracción y limpieza del corpus")
    parser.add_argument("--corpus", type=Path, required=True, help="Carpeta raíz del corpus crudo")
    parser.add_argument("--salida", type=Path, required=True, help="Archivo JSONL de salida")
    args = parser.parse_args()

    args.salida.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(args.salida, "w", encoding="utf-8") as f:
        for doc in procesar_corpus(args.corpus):
            f.write(doc.to_json_line() + "\n")
            total += 1

    print(f"\nListo: {total} documentos procesados -> {args.salida}")


if __name__ == "__main__":
    main()