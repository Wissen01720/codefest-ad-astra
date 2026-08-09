"""Fase 4 — CLI: `chunks.jsonl` (Fase 3) → `index.faiss` + `metadata.jsonl`.

    uv run python -m codefest_ad_astra.indexing.build_index \
        --entrada data/processed/chunks.jsonl \
        --salida entrega/base_vectorial \
        --modelo BAAI/bge-m3

Escribe en `<salida>/encoder_<slug>/`. La codificación se hace por bloques y
cada bloque se persiste en `_parciales/`, de modo que una corrida de horas que
se interrumpa pueda reanudarse con `--reanudar` sin recodificar todo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .encoders import (
    MAX_SEQ_LENGTH_POR_DEFECTO,
    MODELO_POR_DEFECTO,
    crear_encoder,
    slug_modelo,
)
from .faiss_store import (
    CAMPOS_OBLIGATORIOS,
    MetadataInvalidaError,
    construir_indice,
    guardar_base_vectorial,
    validar_campos_obligatorios,
)

TAMANO_BLOQUE_POR_DEFECTO = 2048
NOMBRE_PARCIALES = "_parciales"
NOMBRE_ESTADO = "estado.json"
NOMBRE_DESCARTADOS = "descartados.jsonl"


def sha256_archivo(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_chunks(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Itera (numero_de_linea, registro) sobre el chunks.jsonl de la Fase 3."""
    with open(path, encoding="utf-8") as f:
        for numero, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                registro = json.loads(linea)
            except json.JSONDecodeError as exc:
                raise MetadataInvalidaError(f"Línea {numero} de {path.name} no es JSON válido: {exc}") from exc
            yield numero, registro


def preparar_fragmentos(
    path_entrada: Path, limite: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Carga y filtra los fragmentos a indexar.

    Devuelve (fragmentos, descartados). Se descartan los fragmentos con `texto`
    vacío o solo espacios: no aportan nada al índice y su vector sería ruido.
    El filtro es determinístico sobre el archivo de entrada, así que reanudar
    una corrida reproduce exactamente el mismo orden.
    """
    fragmentos: list[dict[str, Any]] = []
    descartados: list[dict[str, Any]] = []

    for numero, registro in leer_chunks(path_entrada):
        validar_campos_obligatorios(registro, linea=numero)
        texto = registro.get("texto")
        if not isinstance(texto, str) or not texto.strip():
            descartados.append(
                {
                    "linea": numero,
                    "doc_id": registro.get("doc_id"),
                    "chunk_id": registro.get("chunk_id"),
                    "motivo": "texto_vacio",
                }
            )
            continue
        fragmentos.append(registro)
        if limite is not None and len(fragmentos) >= limite:
            break

    return fragmentos, descartados


def detectar_truncamiento(
    fragmentos: list[dict[str, Any]], encoder, *, muestra: int = 200
) -> list[dict[str, Any]]:
    """Detecta fragmentos que el encoder truncaría por exceder `max_seq_length`.

    La Fase 3 acota los chunks con el tokenizer que se le pase por CLI. Si ese
    no es el del encoder de indexación, el conteo no coincide y el modelo corta
    el texto en silencio: el vector representa solo el principio del fragmento,
    pero `metadata.jsonl` sigue guardando el texto completo.

    Revisa solo los `muestra` fragmentos más largos por caracteres — si alguno
    sobra, sobra ahí.
    """
    limite = getattr(encoder, "max_seq_length", None)
    contar = getattr(encoder, "contar_tokens", None)
    if not limite or contar is None or not fragmentos:
        return []

    candidatos = sorted(fragmentos, key=lambda f: len(f["texto"]), reverse=True)[:muestra]
    conteos = contar([f["texto"] for f in candidatos])
    return [
        {"chunk_id": f["chunk_id"], "doc_id": f["doc_id"], "tokens_reales": n, "limite": limite}
        for f, n in zip(candidatos, conteos)
        if n > limite
    ]


def _path_shard(dir_parciales: Path, indice_bloque: int) -> Path:
    return dir_parciales / f"emb_{indice_bloque:06d}.npy"


def _cargar_estado(dir_parciales: Path) -> dict[str, Any]:
    path = dir_parciales / NOMBRE_ESTADO
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _guardar_estado(dir_parciales: Path, estado: dict[str, Any]) -> None:
    (dir_parciales / NOMBRE_ESTADO).write_text(
        json.dumps(estado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _estado_compatible(estado: dict[str, Any], firma: dict[str, Any]) -> bool:
    """Un checkpoint solo sirve si viene del mismo input, modelo y dimensión."""
    return all(estado.get(clave) == valor for clave, valor in firma.items())


def codificar_por_bloques(
    fragmentos: list[dict[str, Any]],
    encoder,
    dir_parciales: Path,
    *,
    tamano_bloque: int,
    firma: dict[str, Any],
    reanudar: bool,
) -> np.ndarray:
    """Codifica los textos por bloques, persistiendo cada bloque en disco."""
    dir_parciales.mkdir(parents=True, exist_ok=True)
    total_bloques = (len(fragmentos) + tamano_bloque - 1) // tamano_bloque

    estado = _cargar_estado(dir_parciales) if reanudar else {}
    if reanudar and estado and not _estado_compatible(estado, firma):
        raise SystemExit(
            "El checkpoint en "
            f"{dir_parciales} no corresponde a esta corrida "
            f"(entrada/modelo/dimensión distintos). Bórralo o corre sin --reanudar."
        )
    if not reanudar:
        for viejo in dir_parciales.glob("emb_*.npy"):
            viejo.unlink()

    bloques_listos = int(estado.get("bloques_completados", 0)) if reanudar else 0
    if bloques_listos:
        print(f"  [REANUDAR] {bloques_listos}/{total_bloques} bloques ya codificados, continuando")

    inicio = time.time()
    for indice_bloque in range(bloques_listos, total_bloques):
        desde = indice_bloque * tamano_bloque
        hasta = min(desde + tamano_bloque, len(fragmentos))
        textos = [f["texto"] for f in fragmentos[desde:hasta]]

        vectores = encoder.codificar_pasajes(textos)
        if vectores.shape != (len(textos), encoder.dimension):
            raise RuntimeError(
                f"El encoder devolvió shape {vectores.shape}, se esperaba "
                f"{(len(textos), encoder.dimension)}"
            )
        np.save(_path_shard(dir_parciales, indice_bloque), vectores)

        _guardar_estado(
            dir_parciales,
            {**firma, "bloques_completados": indice_bloque + 1, "total_bloques": total_bloques},
        )

        procesados = hasta
        transcurrido = time.time() - inicio
        ritmo = procesados / transcurrido if transcurrido > 0 else 0.0
        print(
            f"  [{indice_bloque + 1}/{total_bloques}] {procesados}/{len(fragmentos)} fragmentos "
            f"({ritmo:.1f} frag/s)",
            flush=True,
        )

    partes = [np.load(_path_shard(dir_parciales, i)) for i in range(total_bloques)]
    vectores = np.ascontiguousarray(np.vstack(partes), dtype=np.float32)
    if vectores.shape[0] != len(fragmentos):
        raise RuntimeError(
            f"Se recuperaron {vectores.shape[0]} vectores de los parciales pero hay "
            f"{len(fragmentos)} fragmentos. Borra {dir_parciales} y vuelve a correr."
        )
    return vectores


def construir_metadata(fragmentos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ordena cada registro con los campos de la Tabla 1 primero.

    Conserva íntegros los campos extra de la Fase 3 (idioma, num_palabras,
    char_start, char_end): la especificación los permite y dan trazabilidad.
    """
    metadata = []
    for fragmento in fragmentos:
        registro = {campo: fragmento[campo] for campo in CAMPOS_OBLIGATORIOS}
        for clave, valor in fragmento.items():
            if clave not in registro:
                registro[clave] = valor
        metadata.append(registro)
    return metadata


def _versiones() -> dict[str, str]:
    versiones: dict[str, str] = {"python": sys.version.split()[0]}
    for nombre, modulo in (("faiss", "faiss"), ("numpy", "numpy"), ("torch", "torch"),
                           ("sentence_transformers", "sentence_transformers")):
        try:
            versiones[nombre] = __import__(modulo).__version__
        except Exception:
            versiones[nombre] = "no disponible"
    return versiones


def construir_base_vectorial(
    path_entrada: Path,
    dir_salida_base: Path,
    *,
    nombre_modelo: str = MODELO_POR_DEFECTO,
    device: str | None = None,
    batch_size: int = 32,
    max_seq_length: int | None = MAX_SEQ_LENGTH_POR_DEFECTO,
    tamano_bloque: int = TAMANO_BLOQUE_POR_DEFECTO,
    limite: int | None = None,
    reanudar: bool = False,
    conservar_parciales: bool = False,
    encoder=None,
) -> Path:
    """Ejecuta la Fase 4 completa y devuelve el directorio del encoder escrito."""
    path_entrada = Path(path_entrada)
    if not path_entrada.exists():
        raise FileNotFoundError(f"No se encontró el archivo de fragmentos: {path_entrada}")

    print(f"Leyendo fragmentos de {path_entrada}")
    fragmentos, descartados = preparar_fragmentos(path_entrada, limite=limite)
    if not fragmentos:
        raise SystemExit("No quedó ningún fragmento con texto que indexar.")
    print(f"  {len(fragmentos)} fragmentos a indexar, {len(descartados)} descartados por texto vacío")
    if limite is not None:
        print(f"  [AVISO] --limite {limite}: índice PARCIAL, no apto para la entrega final")

    if encoder is None:
        print(f"Cargando encoder {nombre_modelo} (device={device or 'auto'})")
        encoder = crear_encoder(
            nombre_modelo,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        )
    print(f"  dimensión={encoder.dimension} device={getattr(encoder, 'device', '?')} "
          f"max_seq_length={getattr(encoder, 'max_seq_length', '?')}")

    truncados = detectar_truncamiento(fragmentos, encoder)
    if truncados:
        peor = max(t["tokens_reales"] for t in truncados)
        print(
            f"  [AVISO] al menos {len(truncados)} fragmentos superan el límite de "
            f"{truncados[0]['limite']} tokens del encoder (el mayor tiene {peor}) y se truncarán. "
            "Su vector solo representará el inicio del texto. Vuelve a correr la Fase 3 con "
            f"--tokenizer-model {encoder.nombre} y un --max-tokens por debajo del límite."
        )

    dir_encoder = Path(dir_salida_base) / f"encoder_{slug_modelo(encoder.nombre)}"
    dir_parciales = dir_encoder / NOMBRE_PARCIALES

    hash_entrada = sha256_archivo(path_entrada)
    firma = {
        "sha256_entrada": hash_entrada,
        "modelo": encoder.nombre,
        "dimension": encoder.dimension,
        "num_fragmentos": len(fragmentos),
    }

    print(f"Codificando {len(fragmentos)} fragmentos en bloques de {tamano_bloque}")
    inicio = time.time()
    vectores = codificar_por_bloques(
        fragmentos,
        encoder,
        dir_parciales,
        tamano_bloque=tamano_bloque,
        firma=firma,
        reanudar=reanudar,
    )
    segundos = time.time() - inicio

    print("Construyendo IndexFlatIP")
    indice = construir_indice(vectores)
    metadata = construir_metadata(fragmentos)

    manifiesto = {
        "fase": "4 - embeddings + indice FAISS",
        "creado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": encoder.nombre,
        "dimension": encoder.dimension,
        "device": getattr(encoder, "device", None),
        "max_seq_length": getattr(encoder, "max_seq_length", None),
        "batch_size": batch_size,
        "prefijo_consulta": getattr(encoder, "prefijo_consulta", ""),
        "prefijo_pasaje": getattr(encoder, "prefijo_pasaje", ""),
        "tipo_indice": "IndexFlatIP",
        "metrica": "similitud coseno (producto interno sobre vectores normalizados)",
        "vectores_normalizados": True,
        "num_vectores": int(indice.ntotal),
        "num_fragmentos_descartados": len(descartados),
        "fragmentos_truncados_detectados": len(truncados),
        "entrada": str(path_entrada),
        "sha256_entrada": hash_entrada,
        "indice_parcial": limite is not None,
        "limite_aplicado": limite,
        "segundos_codificacion": round(segundos, 1),
        "versiones": _versiones(),
    }

    guardar_base_vectorial(dir_encoder, indice, metadata, manifiesto)
    if descartados:
        with open(dir_encoder / NOMBRE_DESCARTADOS, "w", encoding="utf-8") as f:
            for registro in descartados:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    if not conservar_parciales and dir_parciales.exists():
        shutil.rmtree(dir_parciales)

    print(f"\nListo: {indice.ntotal} vectores de dimensión {encoder.dimension} -> {dir_encoder}")
    print(f"  {dir_encoder / 'index.faiss'}")
    print(f"  {dir_encoder / 'metadata.jsonl'}")
    return dir_encoder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fase 4: genera embeddings de los fragmentos y construye el índice FAISS."
    )
    parser.add_argument("--entrada", type=Path, required=True, help="chunks.jsonl de la Fase 3")
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("entrega/base_vectorial"),
        help="Directorio base; se escribe en <salida>/encoder_<slug>/",
    )
    parser.add_argument("--modelo", type=str, default=MODELO_POR_DEFECTO, help="Encoder de HuggingFace")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda... (por defecto: automático)")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de lote del encoder")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH_POR_DEFECTO,
        help="Tokens máximos por fragmento en el encoder",
    )
    parser.add_argument(
        "--tamano-bloque",
        type=int,
        default=TAMANO_BLOQUE_POR_DEFECTO,
        help="Fragmentos por checkpoint en disco",
    )
    parser.add_argument("--limite", type=int, default=None, help="Solo los primeros N fragmentos (pruebas)")
    parser.add_argument("--reanudar", action="store_true", help="Continúa desde los bloques ya codificados")
    parser.add_argument(
        "--conservar-parciales", action="store_true", help="No borra _parciales/ al terminar"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        construir_base_vectorial(
            args.entrada,
            args.salida,
            nombre_modelo=args.modelo,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            tamano_bloque=args.tamano_bloque,
            limite=args.limite,
            reanudar=args.reanudar,
            conservar_parciales=args.conservar_parciales,
        )
    except SystemExit as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
