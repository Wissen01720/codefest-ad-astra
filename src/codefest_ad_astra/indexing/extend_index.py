"""Fase 4 — agrega chunks nuevos a una base vectorial ya construida, SIN
re-encodear lo que ya está indexado.

Uso pensado para recuperar documentos que antes cayeron en skip-document y se
re-procesaron aparte (chunks_nuevos.jsonl): en vez de correr build_index.py
sobre el chunks.jsonl completo (re-encoda TODO, incluido lo que ya estaba
bien), esto solo codifica los fragmentos nuevos y los agrega al index.faiss +
metadata.jsonl existentes.

    uv run python -m codefest_ad_astra.indexing.extend_index \
        --base entrega/base_vectorial/encoder_bge-m3 \
        --chunks-nuevos /tmp/reintentos_chunks.jsonl

Reusa el mismo encoder/checkpointing por bloques de build_index.py, así que
si se interrumpe a mitad de camino, --reanudar retoma sin recodificar los
bloques nuevos ya hechos (los viejos, ya indexados, nunca se tocan).
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
    NOMBRE_PARCIALES,
    TAMANO_BLOQUE_POR_DEFECTO,
    codificar_por_bloques,
    construir_metadata,
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

NOMBRE_PARCIALES_INCREMENTO = "_parciales_incremento"


def extender_base_vectorial(
    dir_base: Path,
    path_chunks_nuevos: Path,
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

    doc_ids_existentes = {r["doc_id"] for r in metadata_vieja}

    fragmentos, descartados = preparar_fragmentos(path_chunks_nuevos)
    if not fragmentos:
        raise SystemExit("No quedó ningún fragmento con texto que indexar en chunks-nuevos.")

    # Nunca dupliques un doc_id que ya está en la base: si chunks_nuevos trae
    # por error algo ya indexado, mejor abortar que corromper la alineación.
    ya_indexados = {f["doc_id"] for f in fragmentos} & doc_ids_existentes
    if ya_indexados:
        raise SystemExit(
            f"{len(ya_indexados)} doc_id de chunks-nuevos ya están en la base existente "
            f"(ej. {sorted(ya_indexados)[0]}). Revisa que no estés re-agregando lo mismo dos veces."
        )

    print(f"  {len(fragmentos)} fragmentos nuevos a indexar, {len(descartados)} descartados por texto vacío")

    nombre_modelo = manifiesto_viejo.get("modelo")
    if not nombre_modelo:
        raise SystemExit(f"{dir_base / NOMBRE_MANIFIESTO} no indica el modelo original; no se puede extender.")

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
        print(f"  [AVISO] al menos {len(truncados)} fragmentos nuevos se truncarán (el mayor tiene {peor} tokens)")

    dir_parciales = dir_base / NOMBRE_PARCIALES_INCREMENTO
    hash_entrada = sha256_archivo(path_chunks_nuevos)
    firma = {
        "sha256_entrada": hash_entrada,
        "modelo": encoder.nombre,
        "dimension": encoder.dimension,
        "num_fragmentos": len(fragmentos),
    }

    print(f"Codificando {len(fragmentos)} fragmentos nuevos en bloques de {tamano_bloque}")
    inicio = time.time()
    vectores_nuevos = codificar_por_bloques(
        fragmentos, encoder, dir_parciales,
        tamano_bloque=tamano_bloque, firma=firma, reanudar=reanudar,
    )
    segundos = time.time() - inicio

    print("Combinando con el índice existente")
    metadata_nueva = construir_metadata(fragmentos)
    vectores_combinados = np.ascontiguousarray(
        np.vstack([indice_viejo.reconstruct_n(0, indice_viejo.ntotal), vectores_nuevos]),
        dtype=np.float32,
    )
    metadata_combinada = list(metadata_vieja) + metadata_nueva

    indice_combinado = construir_indice(vectores_combinados)
    validar_alineacion(indice_combinado, metadata_combinada)

    manifiesto_nuevo = dict(manifiesto_viejo)
    manifiesto_nuevo.update({
        "actualizado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_vectores": int(indice_combinado.ntotal),
        "num_vectores_agregados_en_esta_extension": int(vectores_nuevos.shape[0]),
        "num_fragmentos_descartados_en_esta_extension": len(descartados),
        "fragmentos_truncados_detectados_en_esta_extension": len(truncados),
        "entrada_agregada": str(path_chunks_nuevos),
        "sha256_entrada_agregada": hash_entrada,
        "segundos_codificacion_extension": round(segundos, 1),
    })

    guardar_base_vectorial(dir_base, indice_combinado, metadata_combinada, manifiesto_nuevo)

    if not conservar_parciales and dir_parciales.exists():
        shutil.rmtree(dir_parciales)

    print(f"\nListo: {indice_viejo.ntotal} + {vectores_nuevos.shape[0]} = {indice_combinado.ntotal} vectores -> {dir_base}")
    return dir_base


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agrega chunks nuevos a una base vectorial existente sin re-encodear lo ya indexado."
    )
    parser.add_argument("--base", type=Path, required=True, help="Directorio encoder_<slug>/ ya existente")
    parser.add_argument("--chunks-nuevos", type=Path, required=True, help="chunks.jsonl solo con los fragmentos nuevos")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tamano-bloque", type=int, default=TAMANO_BLOQUE_POR_DEFECTO)
    parser.add_argument("--reanudar", action="store_true")
    parser.add_argument("--conservar-parciales", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        extender_base_vectorial(
            args.base, args.chunks_nuevos,
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