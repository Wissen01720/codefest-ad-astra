"""Fase 7 — genera resultados.jsonl a partir de las 50 consultas (q001-q050).

Reusa `Buscador` (Fase 6) y `agregar_a_documentos` (misma agregación por
documento que ya usa retrieval/recuperar.py — no se reimplementa nada de esa
lógica aquí, solo se formatea la salida según el esquema de la Sección 9).

    uv run python -m codefest_ad_astra.retrieval.generador \
        --base entrega/base_vectorial/encoder_bge-m3 \
        --consultas data/consultas.jsonl \
        --salida entrega/resultados.jsonl

Reproducible: mismas consultas + mismo índice -> mismo resultados.jsonl
byte a byte (FAISS con IndexFlatIP es determinístico, sin aleatoriedad).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ..indexing.search import Buscador
from .recuperar import DEFAULT_TOP_DOCUMENTOS, agregar_a_documentos

MAX_PALABRAS_FRAGMENTO = 250
MAX_DOCUMENTOS = 3
MAX_FRAGMENTOS = 10
K_CHUNKS_POR_DEFECTO = 60  # suficientes candidatos para que los 3 docs top sumen >=10 chunks entre todos

_FIN_ORACION_RE = re.compile(r"(?<=[.!?])\s+")


def _truncar_a_oraciones_completas(texto: str, max_palabras: int) -> str:
    """Recorta `texto` a lo sumo `max_palabras`, cortando siempre en un
    límite de oración completa — nunca a mitad de una oración.

    La mayoría de los chunks de Fase 3 ya respetan max_words=250 por
    construcción; esto es un resguardo defensivo para cualquier caso borde
    que se haya colado (p. ej. por la aproximación de conteo de tokens vs
    palabras en el fallback de chunking).
    """
    palabras = texto.split()
    if len(palabras) <= max_palabras:
        return texto

    oraciones = _FIN_ORACION_RE.split(texto)
    acumuladas: list[str] = []
    conteo = 0
    for oracion in oraciones:
        n = len(oracion.split())
        if conteo + n > max_palabras:
            break
        acumuladas.append(oracion)
        conteo += n

    if not acumuladas:
        # Ni la primera oración cabe sola (oración anómalamente larga):
        # último recurso, corte duro por palabras.
        return " ".join(palabras[:max_palabras])
    return " ".join(acumuladas)


def _cargar_consultas(path: Path) -> list[dict[str, str]]:
    consultas = []
    with open(path, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                consultas.append(json.loads(linea))
    return consultas


def generar_resultado(buscador: Buscador, query_id: str, pregunta: str) -> dict[str, Any]:
    # Búsqueda adaptativa: si los MAX_DOCUMENTOS documentos top no traen entre
    # todos MAX_FRAGMENTOS chunks dentro de los primeros k candidatos, se
    # amplía k y se repite — algunas consultas concentran sus mejores
    # coincidencias en documentos con pocos chunks entre los primeros k.
    k = K_CHUNKS_POR_DEFECTO
    documentos: list[Any] = []
    candidatos: list[Any] = []
    while True:
        resultados = buscador.buscar(pregunta, k=k)
        documentos = agregar_a_documentos(resultados, top_documentos=MAX_DOCUMENTOS)
        candidatos = [c for doc in documentos for c in doc.chunks]
        if len(candidatos) >= MAX_FRAGMENTOS or k >= buscador.indice.ntotal:
            break
        k = min(k * 4, buscador.indice.ntotal)

    documents_out = [
        {"rank": i + 1, "doc_id": doc.doc_id} for i, doc in enumerate(documentos)
    ]

    candidatos.sort(key=lambda c: c.score, reverse=True)
    top_fragmentos = candidatos[:MAX_FRAGMENTOS]

    fragments_out = [
        {
            "rank": i + 1,
            "chunk_id": c.chunk_id,
            "doc_id": c.metadata["doc_id"],
            "text": _truncar_a_oraciones_completas(c.metadata["texto"], MAX_PALABRAS_FRAGMENTO),
        }
        for i, c in enumerate(top_fragmentos)
    ]

    return {"query_id": query_id, "documents": documents_out, "fragments": fragments_out}


def validar_resultados(path: Path, num_consultas_esperado: int) -> list[str]:
    """Validación programática de la Fase 7. Devuelve lista de problemas
    encontrados (vacía si todo está bien)."""
    problemas = []
    lineas = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lineas) != num_consultas_esperado:
        problemas.append(f"esperaba {num_consultas_esperado} líneas, encontré {len(lineas)}")

    for i, linea in enumerate(lineas, start=1):
        r = json.loads(linea)
        qid = r.get("query_id", f"línea {i}")
        docs = r.get("documents", [])
        frags = r.get("fragments", [])
        if len(docs) != MAX_DOCUMENTOS:
            problemas.append(f"{qid}: {len(docs)} documentos, esperaba {MAX_DOCUMENTOS}")
        if len(frags) != MAX_FRAGMENTOS:
            problemas.append(f"{qid}: {len(frags)} fragmentos, esperaba {MAX_FRAGMENTOS}")
        for frag in frags:
            n_palabras = len(frag.get("text", "").split())
            if n_palabras > MAX_PALABRAS_FRAGMENTO:
                problemas.append(
                    f"{qid}: fragmento {frag.get('chunk_id')} tiene {n_palabras} palabras "
                    f"(> {MAX_PALABRAS_FRAGMENTO})"
                )
    return problemas


def generar_resultados_jsonl(
    dir_base: Path,
    path_consultas: Path,
    path_salida: Path,
    *,
    device: str | None = None,
) -> None:
    consultas = _cargar_consultas(path_consultas)
    print(f"{len(consultas)} consultas cargadas de {path_consultas}")

    buscador = Buscador(dir_base, device=device)

    path_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(path_salida, "w", encoding="utf-8") as f:
        for i, c in enumerate(consultas, start=1):
            resultado = generar_resultado(buscador, c["query_id"], c["pregunta"])
            f.write(json.dumps(resultado, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(consultas)}] {c['query_id']} -> "
                  f"{[d['doc_id'] for d in resultado['documents']]}")

    print(f"\nEscrito {path_salida}")
    problemas = validar_resultados(path_salida, len(consultas))
    if problemas:
        print(f"\n[VALIDACIÓN] {len(problemas)} problema(s) encontrado(s):")
        for p in problemas:
            print(f"  - {p}")
        raise SystemExit(1)
    print("[VALIDACIÓN] OK: todas las líneas cumplen el esquema de la Sección 9.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera resultados.jsonl (Fase 7).")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--consultas", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        generar_resultados_jsonl(args.base, args.consultas, args.salida, device=args.device)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())