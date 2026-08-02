"""Extractores de texto por formato (Fase 1).

Cada función recibe la ruta de un archivo y devuelve el texto crudo extraído,
SIN limpiar todavía (eso lo hace cleaning.py, Fase 2).
"""
from pathlib import Path
import json

import pdfplumber
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
import pytesseract


def extract_pdf_paginas(path: Path) -> list[str]:
    """Extrae el texto de cada página del PDF por separado. Se deja así (en vez
    de un solo string) para poder detectar y quitar headers/footers repetidos
    entre páginas antes de unir todo en un solo texto (ver cleaning.py).

    x_tolerance bajo (en vez del default de pdfplumber) evita que se peguen
    palabras completas en párrafos justificados con espacios angostos
    (ej. 'Paraelvolumen' en vez de 'Para el volumen')."""
    paginas = []
    with pdfplumber.open(path) as pdf:
        for pagina in pdf.pages:
            texto = pagina.extract_text(x_tolerance=1)
            paginas.append(texto or "")
    return paginas


def extract_pdf(path: Path) -> str:
    """Extrae texto de un PDF preservando el orden de lectura de las páginas
    (versión simple, sin remover headers/footers repetidos)."""
    paginas = extract_pdf_paginas(path)
    return "\n\n".join(p for p in paginas if p)


def extract_html(path: Path) -> str:
    """Extrae solo el texto visible de un HTML, descartando scripts/estilos/markup."""
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")


_CAMPOS_TITULO = ("title", "headline", "titulo")
_CAMPOS_CUERPO = ("body_text", "body_paragraphs", "text", "content", "article_text", "full_text", "description")


def _extraer_valores_largos(obj, min_len: int = 40) -> list[str]:
    """Recorre recursivamente un JSON (dict/list) y recoge todos los valores
    de texto 'largos' (heurística: más de `min_len` caracteres), ignorando
    campos cortos que suelen ser metadata (ids, fechas, urls cortas, etc.).
    Se usa como último recurso cuando ningún campo conocido coincide."""
    encontrados = []
    if isinstance(obj, dict):
        for v in obj.values():
            encontrados.extend(_extraer_valores_largos(v, min_len))
    elif isinstance(obj, list):
        for v in obj:
            encontrados.extend(_extraer_valores_largos(v, min_len))
    elif isinstance(obj, str) and len(obj) >= min_len:
        encontrados.append(obj)
    return encontrados


def extract_json(path: Path) -> str:
    """Extrae texto de un JSON de artículo.

    Prioriza UN SOLO campo de cuerpo: se observó en el corpus real de ADL que
    algunos artículos traen el mismo contenido duplicado en 'body_text' (ya
    unido) y 'body_paragraphs' (como lista) — usar ambos duplicaría el texto.
    Campos descriptivos (url, date, authors, tags, excerpt) se dejan fuera del
    cuerpo a propósito, tal como recomienda el documento técnico (sección 2.1).

    Si ningún campo conocido coincide (observatorio con esquema distinto),
    usa un respaldo genérico: recoge todos los valores de texto largos del
    JSON en vez de dejar el documento vacío, y avisa en consola para que se
    pueda revisar y, si hace falta, agregar el nombre de campo real a
    _CAMPOS_CUERPO.
    """
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    
    if not isinstance(data, dict):
        print(f"  [FALLBACK JSON] {path.name}: raíz no es dict (es {type(data).__name__}), usando respaldo genérico")
        return "\n\n".join(_extraer_valores_largos(data)).strip()

    if not isinstance(data, dict):
        # Raíz de tipo list (u otro tipo no-dict): no hay campos con nombre
        # que buscar con .get(), así que se usa directamente el respaldo
        # genérico, que sí sabe recorrer listas recursivamente.
        print(f"  [FALLBACK JSON] {path.name}: raíz no es dict (es {type(data).__name__}), usando respaldo genérico")
        return "\n\n".join(_extraer_valores_largos(data)).strip()

    titulo = ""
    for campo in _CAMPOS_TITULO:
        if data.get(campo):
            titulo = str(data[campo])
            break

    cuerpo = ""
    for campo in _CAMPOS_CUERPO:
        valor = data.get(campo)
        if not valor:
            continue
        cuerpo = "\n\n".join(str(v) for v in valor) if isinstance(valor, list) else str(valor)
        break  # se detiene en el primer campo de cuerpo que aparezca, para no duplicar

    if not cuerpo:
        print(f"  [FALLBACK JSON] {path.name}: ningún campo conocido coincidió, usando respaldo genérico")
        cuerpo = "\n\n".join(_extraer_valores_largos(data))

    return f"{titulo}\n\n{cuerpo}".strip()


def extract_csv(path: Path) -> str:
    """Convierte cada fila en 'columna: valor', una fila por línea."""
    df = pd.read_csv(
        path,
        on_bad_lines="skip",
        engine="python"
    )
    return _dataframe_a_texto(df)


def extract_xlsx(path: Path) -> str:
    df = pd.read_excel(path)
    return _dataframe_a_texto(df)


def _dataframe_a_texto(df: pd.DataFrame) -> str:
    filas = []
    for _, fila in df.iterrows():
        pares = [f"{col}: {val}" for col, val in fila.items() if pd.notna(val)]
        filas.append(" | ".join(pares))
    return "\n".join(filas)


def extract_txt(path: Path) -> str:
    """Lee texto plano directamente."""
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_image(path: Path, idioma_ocr: str = "spa+eng+por") -> str:
    """OCR sobre imágenes con texto (infografías, gráficos con etiquetas).

    Requiere los paquetes de idioma de tesseract instalados:
        sudo apt install -y tesseract-ocr-spa tesseract-ocr-por
    """
    imagen = Image.open(path)
    return pytesseract.image_to_string(imagen, lang=idioma_ocr)


def extract_pbf(path: Path) -> str:
    """Decodifica un Mapbox Vector Tile (MVT) y extrae los atributos de sus
    features como pares 'atributo: valor', agrupados por capa — tal como
    recomienda el documento técnico (sección 2.1) para archivos PBF de mapas.

    Requiere: uv add mapbox-vector-tile
    (Nota: MVT es un formato de teselas para renderizar mapas web, distinto
    al PBF de OpenStreetMap — no confundir, usan librerías distintas.)
    """
    import mapbox_vector_tile

    datos = path.read_bytes()
    tile = mapbox_vector_tile.decode(datos)

    partes = []
    for nombre_capa, capa in tile.items():
        for feature in capa.get("features", []):
            props = feature.get("properties", {})
            if not props:
                continue
            pares = "; ".join(f"{k}: {v}" for k, v in props.items())
            partes.append(f"[{nombre_capa}] {pares}")
    return "\n".join(partes)


EXTRACTORES_POR_EXTENSION = {
    ".pdf": extract_pdf,
    ".html": extract_html,
    ".htm": extract_html,
    ".json": extract_json,
    ".csv": extract_csv,
    ".xlsx": extract_xlsx,
    ".txt": extract_txt,
    ".png": extract_image,
    ".jpg": extract_image,
    ".jpeg": extract_image,
    ".pbf": extract_pbf,
}


def extraer_texto(path: Path) -> str:
    """Despacha al extractor correcto según la extensión del archivo."""
    extension = path.suffix.lower()
    extractor = EXTRACTORES_POR_EXTENSION.get(extension)
    if extractor is None:
        raise ValueError(
            f"No hay extractor configurado para la extensión '{extension}' ({path.name})"
        )
    return extractor(path)