"""Fase 4 — re-codifica un subconjunto de fragmentos YA indexados y reemplaza
sus vectores en el índice existente, sin tocar el resto.

Caso de uso: se corrigió cómo se arma el texto que ve el encoder (p. ej.
quitar URLs antes de vectorizar, ver `_texto_para_encoder` en build_index.py)
y se quiere que esa corrección se refleje solo en los fragmentos afectados,
sin re-encodear los ~200k restantes que ya están bien.

    uv run python -m codefest_ad_astra.indexing.reencode_subset \
        --base entrega/base_vectorial/encoder_bge-m3 \
        --chunks-subset /tmp/chunks_con_url.jsonl

`chunks-subset` debe ser un chunks.jsonl con SOLO los fragmentos a
re-codificar (incluye doc_id/chunk_id/texto etc. — el `texto` es el original,
con URL; `_texto_para_encoder` se aplica igual que en una corrida normal).
Cada chunk_id en el subset DEBE existir ya en la base — este script nunca
agrega fragmentos nuevos, solo reemplaza vectores existentes.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .build_index import (
    TAMANO_BLOQUE_POR_DEFECTO,
    codificar_por_bloques,
    detectar_truncamiento,
    preparar_fragmentos,
    sha256_archivo,
)
from .encoders import MAX_SEQ_LENGTH_POR_DEFECTO, crear_encoder
from .faiss_store import (
    NOMBRE_MANIFIESTO,
    cargar_base_vectorial,
    construir_indice,
    guardar_base_vectorial,
    validar_alineacion,
)

NOMBRE_PARCIALES_REENCODE = "_parciales_reencode"


def reencodar_subconjunto(
    dir_base: Path,
    path_chunks_subset: Path,
    *,
    device: str | None = None,
    batch_size: int = 32,
    tamano_bloque: int = TAMANO_BLOQUE_POR_DEFECTO,
    reanudar: bool = False,
    conservar_parciales: bool = False,
) -> Path:
    dir_base = Path(dir_base)
    print(f"Cargando base vectorial existente de {dir_base}")
    indice_viejo, metadata_vieja, manifiesto_viejo = cargar_base_vectorial(dir_base)
    print(f"  {indice_viejo.ntotal} vectores ya indexados, modelo {manifiesto_viejo.get('modelo')}")

    posicion_por_chunk_id = {r["chunk_id"]: i for i, r in enumerate(metadata_vieja)}

    fragmentos, descartados = preparar_fragmentos(path_chunks_subset)
    if not fragmentos:
        raise SystemExit("No quedó ningún fragmento con texto en chunks-subset.")

    faltantes = [f["chunk_id"] for f in fragmentos if f["chunk_id"] not in posicion_por_chunk_id]
    if faltantes:
        raise SystemExit(
            f"{len(faltantes)} chunk_id de chunks-subset NO existen en la base actual "
            f"(ej. {faltantes[0]}). Este script solo reemplaza vectores existentes, "
            "usa extend_index.py para agregar fragmentos nuevos."
        )

    print(f"  {len(fragmentos)} fragmentos a re-codificar, {len(descartados)} descartados por texto vacío")

    nombre_modelo = manifiesto_viejo.get("modelo")
    if not nombre_modelo:
        raise SystemExit(f"{dir_base / NOMBRE_MANIFIESTO} no indica el modelo original; no se puede reencodar.")

    print(f"Cargando encoder {nombre_modelo} (debe ser EXACTAMENTE el mismo que construyó la base)")
    encoder = crear_encoder(
        nombre_modelo,
        device=device,
        batch_size=batch_size,
        max_seq_length=manifiesto_viejo.get("max_seq_length", MAX_SEQ_LENGTH_POR_DEFECTO),
        dimension=manifiesto_viejo.get("dimension"),
    )
    if encoder.dimension != indice_viejo.d:
        raise SystemExit(
            f"El encoder da dimensión {encoder.dimension} pero el índice existente es de "
            f"dimensión {indice_viejo.d}. ¿Es realmente el mismo modelo?"
        )

    truncados = detectar_truncamiento(fragmentos, encoder)
    if truncados:
        peor = max(t["tokens_reales"] for t in truncados)
        print(f"  [AVISO] al menos {len(truncados)} fragmentos del subset se truncarán (el mayor tiene {peor} tokens)")

    dir_parciales = dir_base / NOMBRE_PARCIALES_REENCODE
    hash_entrada = sha256_archivo(path_chunks_subset)
    firma = {
        "sha256_entrada": hash_entrada,
        "modelo": encoder.nombre,
        "dimension": encoder.dimension,
        "num_fragmentos": len(fragmentos),
    }

    print(f"Re-codificando {len(fragmentos)} fragmentos en bloques de {tamano_bloque}")
    inicio = time.time()
    vectores_nuevos = codificar_por_bloques(
        fragmentos, encoder, dir_parciales,
        tamano_bloque=tamano_bloque, firma=firma, reanudar=reanudar,
    )
    segundos = time.time() - inicio

    print("Reemplazando vectores en el índice existente")
    vectores_completos = indice_viejo.reconstruct_n(0, indice_viejo.ntotal)
    vectores_completos = np.ascontiguousarray(vectores_completos, dtype=np.float32)
    for fragmento, vector_nuevo in zip(fragmentos, vectores_nuevos):
        pos = posicion_por_chunk_id[fragmento["chunk_id"]]
        vectores_completos[pos] = vector_nuevo

    # La metadata (incluido el 'texto' original, con URL) no cambia — solo
    # el vector. construir_indice ya valida que los vectores queden
    # normalizados tras el reemplazo.
    indice_nuevo = construir_indice(vectores_completos)
    validar_alineacion(indice_nuevo, metadata_vieja)

    manifiesto_nuevo = dict(manifiesto_viejo)
    manifiesto_nuevo.update({
        "actualizado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_vectores": int(indice_nuevo.ntotal),
        "num_vectores_reencodados_en_esta_actualizacion": len(fragmentos),
        "num_fragmentos_descartados_en_esta_actualizacion": len(descartados),
        "fragmentos_truncados_detectados_en_esta_actualizacion": len(truncados),
        "entrada_reencodada": str(path_chunks_subset),
        "sha256_entrada_reencodada": hash_entrada,
        "segundos_codificacion_reencode": round(segundos, 1),
    })

    guardar_base_vectorial(dir_base, indice_nuevo, metadata_vieja, manifiesto_nuevo)

    if not conservar_parciales and dir_parciales.exists():
        shutil.rmtree(dir_parciales)

    print(f"\nListo: {len(fragmentos)} vectores reemplazados de {indice_nuevo.ntotal} totales -> {dir_base}")
    return dir_base


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-codifica un subconjunto de fragmentos ya indexados y reemplaza sus vectores."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--chunks-subset", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tamano-bloque", type=int, default=TAMANO_BLOQUE_POR_DEFECTO)
    parser.add_argument("--reanudar", action="store_true")
    parser.add_argument("--conservar-parciales", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        reencodar_subconjunto(
            args.base, args.chunks_subset,
            device=args.device, batch_size=args.batch_size,
            tamano_bloque=args.tamano_bloque, reanudar=args.reanudar,
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