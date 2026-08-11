"""Construcción, persistencia y carga de la base vectorial (índice + metadata).

Invariante central de la entrega (Sección 1.4 y 5.3 de la especificación):

    la línea i de `metadata.jsonl` describe el vector con identificador interno
    i en `index.faiss`.

Si esa correspondencia se rompe, el sistema devuelve textos que no corresponden
a los vectores recuperados y la entrega queda inservible. Todo en este módulo
existe para hacer esa invariante explícita y verificable.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

NOMBRE_INDICE = "index.faiss"
NOMBRE_METADATA = "metadata.jsonl"
NOMBRE_MANIFIESTO = "manifest.json"

# Campos obligatorios por fragmento (Tabla 1 de la especificación). El resto de
# campos que traiga la Fase 3 (idioma, num_palabras, char_start/char_end...) se
# conservan: el reto permite añadir campos, no quitarlos.
CAMPOS_OBLIGATORIOS: tuple[str, ...] = (
    "doc_id",
    "chunk_id",
    "fuente",
    "formato",
    "fenomeno",
    "posicion",
    "num_tokens",
    "texto",
)


class IndexAlignmentError(RuntimeError):
    """El índice y la metadata dejaron de corresponderse 1 a 1."""


class MetadataInvalidaError(ValueError):
    """A un fragmento le faltan campos obligatorios de la Tabla 1."""


def validar_campos_obligatorios(registro: dict[str, Any], *, linea: int) -> None:
    faltantes = [c for c in CAMPOS_OBLIGATORIOS if c not in registro]
    if faltantes:
        raise MetadataInvalidaError(
            f"Línea {linea}: faltan campos obligatorios de la Tabla 1: {', '.join(faltantes)}"
        )


def construir_indice(vectores: np.ndarray):
    """Crea un `IndexFlatIP` y agrega los vectores en orden.

    Plano y exacto por decisión de diseño: el corpus del reto (decenas de miles
    de fragmentos) no justifica un índice aproximado, y un IVF/HNSW introduce
    variabilidad entre corridas que complica reproducir `resultados.jsonl`
    (requisito del entregable 4).

    Los vectores deben venir ya normalizados: con norma unitaria, el producto
    interno es la similitud coseno.
    """
    import faiss

    vectores = np.ascontiguousarray(vectores, dtype=np.float32)
    if vectores.ndim != 2:
        raise ValueError(f"Se esperaba una matriz 2-D, llegó shape={vectores.shape}")
    if vectores.shape[0] == 0:
        raise ValueError("No hay vectores que indexar: el conjunto de fragmentos quedó vacío")

    normas = np.linalg.norm(vectores, axis=1)
    if not np.allclose(normas, 1.0, atol=1e-3):
        peor = float(np.max(np.abs(normas - 1.0)))
        raise ValueError(
            f"Los vectores no están normalizados (desviación máxima {peor:.4f}). "
            "IndexFlatIP solo equivale a similitud coseno con norma unitaria."
        )

    indice = faiss.IndexFlatIP(vectores.shape[1])
    indice.add(vectores)
    return indice


def validar_alineacion(indice, metadata: Sequence[dict[str, Any]]) -> None:
    """Verifica la invariante índice ↔ metadata antes de escribir o consultar."""
    if indice.ntotal != len(metadata):
        raise IndexAlignmentError(
            f"El índice tiene {indice.ntotal} vectores y la metadata {len(metadata)} registros. "
            "La línea i de metadata.jsonl debe describir el vector i del índice."
        )


def _escribir_jsonl_a_temporal(path_final: Path, registros: Iterable[dict[str, Any]]) -> Path:
    """Escribe a un temporal en el mismo directorio del destino final, SIN
    renombrar todavía. Separar 'escribir' de 'confirmar' permite que
    `guardar_base_vectorial` prepare metadata.jsonl e index.faiss por completo
    antes de tocar cualquiera de los dos archivos finales — ver esa función
    para el porqué.
    """
    path_final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path_final.parent, prefix=path_final.name + ".", suffix=".tmp", delete=False
    ) as handle:
        tmp = Path(handle.name)
        for registro in registros:
            handle.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return tmp


def _escribir_indice_a_temporal(path_final: Path, indice) -> Path:
    """Análogo a `_escribir_jsonl_a_temporal` pero para el índice FAISS.

    `faiss.write_index` no es atómico: escribe directamente sobre la ruta que
    se le da. Escribiéndolo primero a un `.tmp` en el mismo directorio, un
    fallo o un corte a mitad de escritura deja el `.tmp` corrupto/incompleto
    pero NO toca `index.faiss` — igual que ya se hacía con metadata.jsonl.
    """
    import faiss

    path_final.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path_final.parent, prefix=path_final.name + ".", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        faiss.write_index(indice, str(tmp))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def guardar_base_vectorial(
    directorio: Path,
    indice,
    metadata: Sequence[dict[str, Any]],
    manifiesto: dict[str, Any] | None = None,
) -> None:
    """Persiste `index.faiss` + `metadata.jsonl` (+ `manifest.json`) alineados.

    Escribe primero AMBOS archivos a temporales (sin tocar los finales) y solo
    si los dos se completaron sin error los "confirma" con `os.replace`, uno
    justo después del otro. Con el orden anterior (metadata primero vía
    `os.replace`, índice después vía `faiss.write_index` directo sobre el
    archivo final) un fallo a mitad de la escritura del índice podía dejar
    `metadata.jsonl` ya reemplazado por la versión nueva mientras
    `index.faiss` quedaba corrupto o con la versión vieja — exactamente la
    desalineación silenciosa que este módulo existe para imposibilitar, y que
    `validar_alineacion` no detecta si el conteo de vectores coincide por
    casualidad entre la versión vieja y la nueva. Con el orden nuevo, la
    ventana de riesgo real se reduce a los dos `os.replace` consecutivos
    (rápidos, sin I/O de por medio) en vez de a todo el tiempo que toma
    escribir metadata.jsonl + index.faiss completos.
    """
    validar_alineacion(indice, metadata)
    directorio.mkdir(parents=True, exist_ok=True)

    path_metadata = directorio / NOMBRE_METADATA
    path_indice = directorio / NOMBRE_INDICE

    tmp_metadata = _escribir_jsonl_a_temporal(path_metadata, metadata)
    try:
        tmp_indice = _escribir_indice_a_temporal(path_indice, indice)
    except Exception:
        tmp_metadata.unlink(missing_ok=True)
        raise

    try:
        os.replace(tmp_metadata, path_metadata)
        os.replace(tmp_indice, path_indice)
    finally:
        # Si algo sobrevivió sin renombrarse (falla entre los dos replace,
        # extremadamente improbable pero no imposible), no dejar basura .tmp.
        for leftover in (tmp_metadata, tmp_indice):
            if leftover.exists():
                leftover.unlink(missing_ok=True)

    if manifiesto is not None:
        (directorio / NOMBRE_MANIFIESTO).write_text(
            json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def leer_metadata(path: Path) -> list[dict[str, Any]]:
    registros: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for numero, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                registro = json.loads(linea)
            except json.JSONDecodeError as exc:
                raise MetadataInvalidaError(f"Línea {numero} de {path.name} no es JSON válido: {exc}") from exc
            validar_campos_obligatorios(registro, linea=numero)
            registros.append(registro)
    return registros


def cargar_base_vectorial(directorio: Path):
    """Carga (índice, metadata, manifiesto) y valida la alineación.

    El índice se lee con `faiss.read_index()` estándar, sin dependencias
    adicionales, tal como exige el entregable.
    """
    import faiss

    directorio = Path(directorio)
    path_indice = directorio / NOMBRE_INDICE
    path_metadata = directorio / NOMBRE_METADATA
    if not path_indice.exists():
        raise FileNotFoundError(f"No se encontró {path_indice}")
    if not path_metadata.exists():
        raise FileNotFoundError(f"No se encontró {path_metadata}")

    indice = faiss.read_index(str(path_indice))
    metadata = leer_metadata(path_metadata)
    validar_alineacion(indice, metadata)

    path_manifiesto = directorio / NOMBRE_MANIFIESTO
    manifiesto = json.loads(path_manifiesto.read_text(encoding="utf-8")) if path_manifiesto.exists() else {}
    return indice, metadata, manifiesto