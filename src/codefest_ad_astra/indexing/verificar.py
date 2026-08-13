"""Verificación de una base vectorial ya construida.

Comprueba lo que la evaluación automática puede castigar (Sección 4 del plan de
proyecto y Secciones 1.4 / 5.3 de la especificación) y describe el contenido
indexado para el informe técnico.

    uv run python -m codefest_ad_astra.indexing.verificar --base entrega/base_vectorial/encoder_bge-m3

Devuelve 0 si todo está bien, 1 si hay algún error bloqueante.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .faiss_store import CAMPOS_OBLIGATORIOS, cargar_base_vectorial

# Límite de la Sección 9.2: cada fragmento entregado debe tener ≤250 palabras.
# Aquí solo se informa: recortar o concatenar es responsabilidad de la Fase 7.
LIMITE_PALABRAS_ENTREGA = 250


def verificar(directorio: Path, *, muestra_normas: int = 1000) -> tuple[list[str], list[str], dict]:
    """Devuelve (errores, advertencias, estadisticas)."""
    errores: list[str] = []
    advertencias: list[str] = []

    indice, metadata, manifiesto = cargar_base_vectorial(directorio)

    # 1. Alineación índice ↔ metadata (ya validada al cargar, se reafirma aquí).
    if indice.ntotal != len(metadata):
        errores.append(f"index.faiss tiene {indice.ntotal} vectores y metadata.jsonl {len(metadata)} líneas")

    # 2. Tipo de índice: plano = exacto y reproducible.
    tipo = type(indice).__name__
    if tipo != "IndexFlatIP":
        advertencias.append(
            f"El índice es {tipo}, no IndexFlatIP. Con vectores normalizados, IndexFlatIP es el que "
            "equivale a similitud coseno con resultados exactos."
        )

    # 3. Campos obligatorios de la Tabla 1 en todas las líneas.
    for numero, registro in enumerate(metadata, start=1):
        faltantes = [c for c in CAMPOS_OBLIGATORIOS if c not in registro]
        if faltantes:
            errores.append(f"Línea {numero}: faltan {', '.join(faltantes)}")
            break  # con un caso basta para saber que hay que rehacer la corrida

    # 4. Normalización: sin ella el producto interno no es coseno.
    total = indice.ntotal
    posiciones = np.linspace(0, total - 1, num=min(muestra_normas, total), dtype=int)
    vectores = np.vstack([indice.reconstruct(int(i)) for i in posiciones])
    normas = np.linalg.norm(vectores, axis=1)
    if not np.allclose(normas, 1.0, atol=1e-3):
        errores.append(
            f"Hay vectores sin normalizar (norma mínima {normas.min():.4f}, máxima {normas.max():.4f})"
        )

    # 5. Identificadores duplicados: rompen la trazabilidad chunk → documento.
    chunk_ids = Counter(r["chunk_id"] for r in metadata)
    duplicados = [cid for cid, n in chunk_ids.items() if n > 1]
    if duplicados:
        advertencias.append(
            f"{len(duplicados)} chunk_id repetidos (ej. {duplicados[0]}). No rompe el índice, "
            "pero complica la trazabilidad al reportar resultados."
        )

    # 6. Fragmentos que exceden el límite de palabras del formato de salida.
    largos = sum(1 for r in metadata if len(r["texto"].split()) > LIMITE_PALABRAS_ENTREGA)
    if largos:
        advertencias.append(
            f"{largos} fragmentos superan {LIMITE_PALABRAS_ENTREGA} palabras. La Fase 7 debe "
            "subdividirlos respetando oraciones completas antes de escribir resultados.jsonl."
        )

    if manifiesto.get("indice_parcial"):
        errores.append(
            f"Índice PARCIAL: se construyó con --limite {manifiesto.get('limite_aplicado')}. "
            "No sirve para la entrega final."
        )

    estadisticas = {
        "directorio": str(directorio),
        "num_vectores": total,
        "dimension": indice.d,
        "tipo_indice": tipo,
        "modelo": manifiesto.get("modelo", "desconocido"),
        "num_documentos": len({r["doc_id"] for r in metadata}),
        "num_fuentes": len({r["fuente"] for r in metadata}),
        "por_fenomeno": dict(sorted(Counter(r["fenomeno"] for r in metadata).items(), key=lambda x: str(x[0]))),
        "por_idioma": dict(Counter(r.get("idioma", "?") for r in metadata).most_common()),
        "por_formato": dict(Counter(r["formato"] for r in metadata).most_common()),
        "tokens_promedio": round(
            sum(r["num_tokens"] for r in metadata) / len(metadata), 1
        ) if metadata else 0,
        "tokens_maximo": max((r["num_tokens"] for r in metadata), default=0),
        "fragmentos_sobre_250_palabras": largos,
    }
    return errores, advertencias, estadisticas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verifica una base vectorial de la Fase 4")
    parser.add_argument("--base", type=Path, required=True, help="Directorio encoder_<slug>/")
    args = parser.parse_args(argv)

    try:
        errores, advertencias, estadisticas = verificar(args.base)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Base vectorial: {estadisticas['directorio']}\n")
    print(f"  modelo            {estadisticas['modelo']}")
    print(f"  índice            {estadisticas['tipo_indice']} (dim {estadisticas['dimension']})")
    print(f"  fragmentos        {estadisticas['num_vectores']}")
    print(f"  documentos        {estadisticas['num_documentos']} ({estadisticas['num_fuentes']} fuentes)")
    print(f"  por fenómeno      {estadisticas['por_fenomeno']}")
    print(f"  por idioma        {estadisticas['por_idioma']}")
    print(f"  por formato       {estadisticas['por_formato']}")
    print(f"  tokens            promedio {estadisticas['tokens_promedio']}, máximo {estadisticas['tokens_maximo']}")

    if advertencias:
        print("\nAdvertencias:")
        for a in advertencias:
            print(f"  - {a}")

    if errores:
        print("\nERRORES BLOQUEANTES:")
        for e in errores:
            print(f"  - {e}")
        return 1

    print("\nOK: índice y metadata alineados, vectores normalizados, campos de la Tabla 1 completos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
