#!/usr/bin/env python3
"""Genera ``resultados.jsonl`` usando solo la base incluida en la entrega.

No depende de ``src/`` ni emplea modelos generativos. La consulta se codifica
con el mismo encoder y prefijo registrados en ``manifest.json``; los
documentos se agregan por suma de scores y los fragmentos conservan el ranking
global de FAISS.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MAX_DOCUMENTOS = 3
MAX_FRAGMENTOS = 10
MAX_PALABRAS = 250
_FIN_ORACION_RE = re.compile(r"[.!?…](?=[\]\)\}\"'»”’]*(?:\s|$))")
_PRIMERA_LETRA_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")


def _normalizar(vectores: np.ndarray) -> np.ndarray:
    vectores = np.ascontiguousarray(vectores, dtype=np.float32)
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    return np.ascontiguousarray(vectores / np.where(normas == 0, 1, normas), dtype=np.float32)


def _texto_completo(texto: str, max_palabras: int = MAX_PALABRAS) -> str | None:
    """Devuelve el prefijo más largo que acaba en una oración completa.

    Además del límite de palabras, esto elimina restos de la siguiente
    oración que hayan quedado al final de un chunk del índice. La sección
    9.2.1 permite reportar un subfragmento conservando el ``chunk_id``.
    """
    ultimo_fin: int | None = None
    for coincidencia in _FIN_ORACION_RE.finditer(texto):
        candidato = texto[:coincidencia.end()].strip()
        if len(candidato.split()) <= max_palabras:
            ultimo_fin = coincidencia.end()
        else:
            break
    if ultimo_fin is None:
        return None
    limpio = texto[:ultimo_fin].strip()
    while True:
        primera = _PRIMERA_LETRA_RE.search(limpio)
        if primera is None or not primera.group().islower():
            return limpio
        fin_parcial = _FIN_ORACION_RE.search(limpio, primera.end())
        if fin_parcial is None:
            return limpio
        restante = limpio[fin_parcial.end():].lstrip(" \t\r\n]})\"'»”’")
        if not restante:
            return limpio
        limpio = restante


def _leer_jsonl(path: Path) -> list[dict[str, Any]]:
    registros: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as archivo:
        for numero, linea in enumerate(archivo, start=1):
            if not linea.strip():
                continue
            try:
                registro = json.loads(linea)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: línea {numero} no es JSON válido: {exc}") from exc
            if not isinstance(registro, dict):
                raise ValueError(f"{path}: línea {numero} no contiene un objeto")
            registros.append(registro)
    return registros


def _cargar_base(base: Path) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    indice = faiss.read_index(str(base / "index.faiss"))
    metadata = _leer_jsonl(base / "metadata.jsonl")
    if indice.ntotal != len(metadata):
        raise ValueError(
            f"índice/metadata desalineados: {indice.ntotal} vectores y {len(metadata)} registros"
        )
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return indice, metadata, manifest


def _crear_encoder(modelo: str, device: str | None, max_seq_length: int) -> SentenceTransformer:
    encoder = SentenceTransformer(modelo, device=device)
    encoder.max_seq_length = max_seq_length
    return encoder


def _codificar_consulta(
    encoder: SentenceTransformer, pregunta: str, prefijo: str
) -> np.ndarray:
    entrada = prefijo + pregunta if prefijo else pregunta
    vector = encoder.encode(
        [entrada], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return _normalizar(vector)


def _generar_consulta(
    indice: Any,
    metadata: list[dict[str, Any]],
    encoder: SentenceTransformer,
    consulta: dict[str, Any],
    prefijo: str,
    k_inicial: int,
) -> dict[str, Any]:
    if indice.ntotal < 1:
        raise ValueError("el índice está vacío")
    if k_inicial < 1:
        raise ValueError("k-inicial debe ser mayor que cero")
    vector = _codificar_consulta(encoder, consulta["pregunta"], prefijo)
    k = min(k_inicial, indice.ntotal)
    documentos: list[tuple[str, float]] = []
    fragmentos: list[tuple[dict[str, Any], str]] = []

    while True:
        scores, ids = indice.search(vector, k)
        acumulados: dict[str, float] = defaultdict(float)
        fragmentos = []
        for score, id_interno in zip(scores[0], ids[0]):
            if id_interno < 0:
                continue
            registro = metadata[int(id_interno)]
            acumulados[str(registro["doc_id"])] += float(score)
            texto = _texto_completo(str(registro["texto"]))
            if texto is not None and len(fragmentos) < MAX_FRAGMENTOS:
                fragmentos.append((registro, texto))
        documentos = sorted(acumulados.items(), key=lambda item: (-item[1], item[0]))[:MAX_DOCUMENTOS]
        if len(documentos) == MAX_DOCUMENTOS and len(fragmentos) == MAX_FRAGMENTOS:
            break
        if k >= indice.ntotal:
            break
        k = min(k * 4, indice.ntotal)

    return {
        "query_id": consulta["query_id"],
        "documents": [
            {"rank": rank, "doc_id": doc_id}
            for rank, (doc_id, _score) in enumerate(documentos, start=1)
        ],
        "fragments": [
            {
                "rank": rank,
                "chunk_id": registro["chunk_id"],
                "doc_id": registro["doc_id"],
                "text": texto,
            }
            for rank, (registro, texto) in enumerate(fragmentos, start=1)
        ],
    }


def generar(
    base: Path,
    consultas_path: Path,
    salida: Path,
    *,
    modelo: str | None,
    device: str | None,
    max_seq_length: int | None,
    k_inicial: int,
) -> None:
    indice, metadata, manifest = _cargar_base(base)
    nombre_modelo = modelo or manifest.get("modelo")
    if not nombre_modelo:
        raise ValueError("falta --modelo y manifest.json no registra uno")
    longitud = max_seq_length or int(manifest.get("max_seq_length", 512))
    prefijo = str(manifest.get("prefijo_consulta", ""))
    encoder = _crear_encoder(nombre_modelo, device, longitud)
    consultas = _leer_jsonl(consultas_path)

    resultados = [
        _generar_consulta(indice, metadata, encoder, consulta, prefijo, k_inicial)
        for consulta in consultas
    ]
    salida.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(prefix=salida.name + ".", dir=salida.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as archivo:
            for resultado in resultados:
                archivo.write(json.dumps(resultado, ensure_ascii=False) + "\n")
        os.replace(temporal, salida)
    finally:
        if os.path.exists(temporal):
            os.unlink(temporal)


def parse_args() -> argparse.Namespace:
    raiz = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Genera resultados.jsonl desde la base FAISS")
    parser.add_argument("--base", type=Path, default=Path(__file__).parent / "base_vectorial/encoder_bge-m3")
    parser.add_argument("--consultas", type=Path, default=raiz / "data/consultas.jsonl")
    parser.add_argument("--salida", type=Path, default=Path(__file__).parent / "resultados.jsonl")
    parser.add_argument("--modelo")
    parser.add_argument("--device")
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--k-inicial", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        generar(
            args.base,
            args.consultas,
            args.salida,
            modelo=args.modelo,
            device=args.device,
            max_seq_length=args.max_seq_length,
            k_inicial=args.k_inicial,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
