"""Fase 4: genera embeddings de fragments.jsonl y construye/persiste el
índice FAISS + su almacén de metadata, en la estructura de entrega exigida
(spec Sección 1.4): base_vectorial/encoder_<nombre>/{index.faiss,metadata.jsonl}

Uso:
    uv run python -m codefest_ad_astra.indexing.build_index \
        --fragmentos data/processed/fragments.jsonl \
        --salida base_vectorial \
        --encoder-nombre bge-m3 \
        --modelo BAAI/bge-m3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np

from .encoder import DEFAULT_ENCODER, encode_texts, load_encoder


def _leer_fragmentos(path: Path) -> list[dict]:
    fragmentos = []
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                fragmentos.append(json.loads(linea))
    return fragmentos


def construir_indice(vectores: np.ndarray) -> faiss.Index:
    """IndexFlatIP: búsqueda exacta por producto interno, equivalente a
    similitud coseno porque los vectores ya vienen normalizados (spec 5.2, 6, 8.2)."""
    dimension = vectores.shape[1]
    indice = faiss.IndexFlatIP(dimension)
    indice.add(vectores)
    return indice


def guardar_base_vectorial(
    fragmentos: list[dict],
    vectores: np.ndarray,
    carpeta_salida: Path,
    nombre_encoder: str,
) -> None:
    """Escribe index.faiss y metadata.jsonl en carpeta_salida/encoder_<nombre>/.

    La línea i de metadata.jsonl corresponde SIEMPRE al fragmento insertado
    en la posición i de `vectores`, que a su vez es el ID interno i que le
    asigna FAISS (los vectores se insertan con index.add() en ese mismo
    orden, sin IDs explícitos) -- así se cumple el requisito de la spec
    Sección 1.4/5.3 de que el orden de metadata.jsonl coincida con los IDs
    internos de FAISS.

    Ambos archivos se escriben primero a rutas temporales dentro de la misma
    carpeta y solo se mueven a su ubicación final (vía os.replace, atómico
    en el mismo filesystem tanto en POSIX como en Windows) una vez que
    AMBAS escrituras terminaron con éxito. Así, si el proceso muere o falla
    la I/O a mitad de camino (disco lleno, fragmento no serializable, etc.),
    nunca queda un index.faiss completo junto a un metadata.jsonl ausente o
    truncado -- el escenario de "índice a medio escribir sin metadata
    correspondiente" que la spec señala como peor que fallar directamente.
    """
    if len(fragmentos) != vectores.shape[0]:
        raise ValueError(
            f"Descuadre entre fragmentos ({len(fragmentos)}) y vectores ({vectores.shape[0]}): "
            "el orden de metadata.jsonl debe coincidir exactamente con los IDs internos de FAISS."
        )

    carpeta_encoder = carpeta_salida / f"encoder_{nombre_encoder}"
    carpeta_encoder.mkdir(parents=True, exist_ok=True)

    ruta_index = carpeta_encoder / "index.faiss"
    ruta_metadata = carpeta_encoder / "metadata.jsonl"
    ruta_index_tmp = carpeta_encoder / "index.faiss.tmp"
    ruta_metadata_tmp = carpeta_encoder / "metadata.jsonl.tmp"

    try:
        indice = construir_indice(vectores)
        faiss.write_index(indice, str(ruta_index_tmp))

        with open(ruta_metadata_tmp, "w", encoding="utf-8") as f:
            for fragmento in fragmentos:
                f.write(json.dumps(fragmento, ensure_ascii=False) + "\n")

        # Ambas escrituras temporales completas: mover a destino final.
        os.replace(ruta_index_tmp, ruta_index)
        os.replace(ruta_metadata_tmp, ruta_metadata)
    finally:
        # Si algo falló antes de completar ambos replace, no debe quedar
        # ningún archivo temporal huérfano en la carpeta de salida.
        for tmp in (ruta_index_tmp, ruta_metadata_tmp):
            if tmp.exists():
                tmp.unlink()


def advertir_si_fragmentos_exceden_limite_modelo(fragmentos: list[dict], modelo) -> None:
    """Compara `num_tokens` de cada fragmento contra `max_seq_length` del
    modelo de embeddings cargado y avisa por stderr si alguno lo excede.

    No es infalible como guardia (spec 4.3): el chunking (Fase 3) cuenta
    tokens con el tokenizer de un modelo específico (`chunking.tokenizer.
    DEFAULT_ENCODER` salvo que se haya corrido con otro), y nada impide
    correr `build_index.py --modelo <otro-modelo>` sobre un fragments.jsonl
    dimensionado para un modelo distinto -- los límites de tokens de dos
    tokenizers no son directamente comparables en general, pero un fragmento
    cuyo `num_tokens` ya excede `max_seq_length` del modelo de embeddings es
    una señal inequívoca de descuadre, así que al menos ese caso se detecta
    y se reporta. No se aborta la corrida (dado el tiempo del hackathon, un
    aviso visible es suficiente); `sentence-transformers` trunca en
    silencio por su cuenta al codificar.
    """
    max_seq_length = getattr(modelo, "max_seq_length", None)
    if max_seq_length is None:
        return

    excedidos = [f for f in fragmentos if f.get("num_tokens", 0) > max_seq_length]
    if excedidos:
        print(
            f"[build_index] Aviso: {len(excedidos)} fragmento(s) tienen num_tokens > "
            f"max_seq_length={max_seq_length} del modelo cargado -- probablemente el "
            "chunking (Fase 3) se corrió con un tokenizer/modelo distinto al de "
            "--modelo aquí. Esos fragmentos serán truncados en silencio al codificar.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 4: embeddings + índice FAISS")
    parser.add_argument("--fragmentos", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True, help="Carpeta base_vectorial/")
    parser.add_argument("--encoder-nombre", type=str, required=True, help="Subcarpeta encoder_<nombre>")
    parser.add_argument("--modelo", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    fragmentos = _leer_fragmentos(args.fragmentos)
    if not fragmentos:
        raise SystemExit(f"No se encontraron fragmentos en {args.fragmentos}")

    textos = [f["texto"] for f in fragmentos]

    modelo = load_encoder(args.modelo)
    advertir_si_fragmentos_exceden_limite_modelo(fragmentos, modelo)
    vectores = encode_texts(modelo, textos, batch_size=args.batch_size)

    guardar_base_vectorial(fragmentos, vectores, args.salida, args.encoder_nombre)

    print(f"\nListo: {len(fragmentos)} fragmentos indexados -> {args.salida}/encoder_{args.encoder_nombre}/")


if __name__ == "__main__":
    main()
