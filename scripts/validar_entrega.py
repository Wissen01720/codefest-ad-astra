#!/usr/bin/env python3
"""Valida la estructura y los contratos técnicos del paquete de entrega."""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pdfplumber

RAIZ = Path(__file__).resolve().parent.parent
ENTREGA = RAIZ / "entrega"
CAMPOS_METADATA = {
    "doc_id", "chunk_id", "fuente", "formato", "fenomeno", "posicion", "num_tokens", "texto"
}
IMPORTS_GENERATIVOS = {
    "openai", "anthropic", "google.generativeai", "transformers.pipeline", "vllm", "llama_cpp"
}
_FIN_ORACION_RE = re.compile(r"[.!?…][\]\)\}\"'»”’]*$")


def _leer_jsonl(path: Path) -> list[dict[str, Any]]:
    registros: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as archivo:
        for numero, linea in enumerate(archivo, start=1):
            if not linea.strip():
                continue
            try:
                registro = json.loads(linea)
            except json.JSONDecodeError as exc:
                raise ValueError(f"línea {numero}: JSON inválido: {exc}") from exc
            if not isinstance(registro, dict):
                raise ValueError(f"línea {numero}: se esperaba un objeto")
            registros.append(registro)
    return registros


def validar_estructura() -> str:
    requeridos = {"resultados.jsonl", "generador.py", "informe_tecnico.pdf", "base_vectorial"}
    presentes = {ruta.name for ruta in ENTREGA.iterdir()}
    faltan = sorted(requeridos - presentes)
    if faltan:
        raise ValueError("faltan: " + ", ".join(faltan))
    encoder_dirs = [ruta for ruta in (ENTREGA / "base_vectorial").iterdir() if ruta.is_dir() and ruta.name.startswith("encoder_")]
    if len(encoder_dirs) != 1:
        raise ValueError(f"se esperaba un directorio encoder_<nombre>, hay {len(encoder_dirs)}")
    return str(encoder_dirs[0])


def validar_resultados() -> str:
    registros = _leer_jsonl(ENTREGA / "resultados.jsonl")
    if len(registros) != 50:
        raise ValueError(f"hay {len(registros)} consultas, se esperaban 50")
    chunks_por_consulta: list[set[str]] = []
    for numero, registro in enumerate(registros, start=1):
        qid = f"q{numero:03d}"
        if set(registro) != {"query_id", "documents", "fragments"}:
            raise ValueError(f"{qid}: esquema raíz inválido")
        if registro["query_id"] != qid:
            raise ValueError(f"línea {numero}: query_id fuera de orden")
        docs, frags = registro["documents"], registro["fragments"]
        if not isinstance(docs, list) or len(docs) != 3:
            raise ValueError(f"{qid}: deben existir exactamente 3 documentos")
        if not isinstance(frags, list) or len(frags) != 10:
            raise ValueError(f"{qid}: deben existir exactamente 10 fragmentos")
        if len({doc.get("doc_id") for doc in docs if isinstance(doc, dict)}) != 3:
            raise ValueError(f"{qid}: los 3 documentos deben ser distintos")
        for rank, doc in enumerate(docs, start=1):
            if not isinstance(doc, dict) or set(doc) != {"rank", "doc_id"}:
                raise ValueError(f"{qid}: documento {rank} tiene esquema inválido")
            if doc["rank"] != rank or not isinstance(doc["doc_id"], str) or not doc["doc_id"]:
                raise ValueError(f"{qid}: documento {rank} tiene rank/doc_id inválido")
        ids: set[str] = set()
        for rank, frag in enumerate(frags, start=1):
            if not isinstance(frag, dict) or set(frag) != {"rank", "chunk_id", "doc_id", "text"}:
                raise ValueError(f"{qid}: fragmento {rank} tiene esquema inválido")
            if frag["rank"] != rank:
                raise ValueError(f"{qid}: fragmento {rank} tiene rank inválido")
            for campo in ("chunk_id", "doc_id", "text"):
                if not isinstance(frag[campo], str) or not frag[campo]:
                    raise ValueError(f"{qid}: fragmento {rank}, {campo} vacío o inválido")
            if len(frag["text"].split()) > 250:
                raise ValueError(f"{qid}: fragmento {rank} supera 250 palabras")
            if not _FIN_ORACION_RE.search(frag["text"].rstrip()):
                raise ValueError(f"{qid}: fragmento {rank} no termina en una oración completa")
            if frag["chunk_id"] in ids:
                raise ValueError(f"{qid}: chunk_id duplicado {frag['chunk_id']}")
            ids.add(frag["chunk_id"])
        chunks_por_consulta.append(ids)
    return "50 consultas; 3 documentos y 10 fragmentos por consulta; límite de 250 palabras"


def validar_base(dir_encoder: Path) -> str:
    indice_path = dir_encoder / "index.faiss"
    metadata_path = dir_encoder / "metadata.jsonl"
    if not indice_path.is_file() or not metadata_path.is_file():
        raise ValueError("faltan index.faiss o metadata.jsonl")
    manifest_path = dir_encoder / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("falta manifest.json, necesario para reproducir el encoder")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indice = faiss.read_index(str(indice_path))
    if indice.ntotal < 1:
        raise ValueError("el índice está vacío")
    if type(indice).__name__ != "IndexFlatIP":
        raise ValueError(f"tipo de índice inesperado: {type(indice).__name__}")
    if manifest.get("indice_parcial"):
        raise ValueError("manifest.json declara un índice parcial")
    if manifest.get("dimension") != indice.d:
        raise ValueError("dimensión de manifest.json distinta de index.faiss")

    resultados = _leer_jsonl(ENTREGA / "resultados.jsonl")
    chunks_salida = {
        frag["chunk_id"]: (frag["doc_id"], frag["text"])
        for resultado in resultados for frag in resultado["fragments"]
    }
    docs_salida = {
        doc["doc_id"] for resultado in resultados for doc in resultado["documents"]
    }
    chunks_encontrados: set[str] = set()
    docs_encontrados: set[str] = set()
    cantidad = 0
    chunks: set[str] = set()
    with metadata_path.open(encoding="utf-8") as archivo:
        for numero, linea in enumerate(archivo, start=1):
            registro = json.loads(linea)
            faltan = CAMPOS_METADATA - set(registro)
            if faltan:
                raise ValueError(f"metadata línea {numero}: faltan {sorted(faltan)}")
            chunk_id = registro["chunk_id"]
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"metadata línea {numero}: chunk_id inválido")
            if chunk_id in chunks:
                raise ValueError(f"metadata línea {numero}: chunk_id duplicado {chunk_id}")
            chunks.add(chunk_id)
            docs_encontrados.add(registro["doc_id"])
            if chunk_id in chunks_salida:
                doc_salida, texto_salida = chunks_salida[chunk_id]
                if doc_salida != registro["doc_id"]:
                    raise ValueError(f"{chunk_id}: doc_id de salida no coincide con metadata")
                if texto_salida not in registro["texto"]:
                    raise ValueError(f"{chunk_id}: text no es un subfragmento trazable de metadata")
                chunks_encontrados.add(chunk_id)
            cantidad += 1
    if indice.ntotal != cantidad:
        raise ValueError(f"index.ntotal={indice.ntotal}, metadata={cantidad}")
    faltan_chunks = set(chunks_salida) - chunks_encontrados
    faltan_docs = docs_salida - docs_encontrados
    if faltan_chunks or faltan_docs:
        raise ValueError(
            f"salida no trazable: {len(faltan_chunks)} chunks y {len(faltan_docs)} documentos faltantes"
        )
    posiciones = np.linspace(0, indice.ntotal - 1, num=min(1000, indice.ntotal), dtype=int)
    normas = np.linalg.norm(np.vstack([indice.reconstruct(int(i)) for i in posiciones]), axis=1)
    if not np.allclose(normas, 1.0, atol=1e-3):
        raise ValueError(f"vectores sin normalizar: normas {normas.min():.4f}–{normas.max():.4f}")
    return (
        f"{indice.ntotal} vectores, dimensión {indice.d}; metadata alineada, "
        f"normas unitarias y {len(chunks_salida)} fragmentos de salida trazables"
    )


def validar_generador() -> str:
    path = ENTREGA / "generador.py"
    codigo = path.read_text(encoding="utf-8")
    arbol = ast.parse(codigo)
    imports: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            imports.update(alias.name for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            imports.add(nodo.module)
    prohibidos = sorted(
        nombre for nombre in imports
        if any(nombre == prefijo or nombre.startswith(prefijo + ".") for prefijo in IMPORTS_GENERATIVOS)
    )
    if prohibidos:
        raise ValueError("imports generativos: " + ", ".join(prohibidos))
    if "codefest_ad_astra" in codigo:
        raise ValueError("depende de src/codefest_ad_astra")
    return "sintaxis válida, autónomo respecto de src/ y sin imports generativos"


def validar_pdf() -> str:
    path = ENTREGA / "informe_tecnico.pdf"
    with pdfplumber.open(path) as documento:
        paginas = len(documento.pages)
        texto = "\n".join(pagina.extract_text() or "" for pagina in documento.pages).lower()
    if paginas > 8:
        raise ValueError(f"tiene {paginas} páginas; máximo permitido: 8")
    for termino in ("chunking", "encoder", "faiss", "grafo"):
        if termino not in texto:
            raise ValueError(f"no aparece la sección requerida: {termino}")
    return f"{paginas} páginas y secciones obligatorias presentes"


def reproducir(dir_encoder: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="validar-entrega-") as temporal:
        salida = Path(temporal) / "resultados.jsonl"
        comando = [
            sys.executable, str(ENTREGA / "generador.py"),
            "--base", str(dir_encoder),
            "--consultas", str(RAIZ / "data/consultas.jsonl"),
            "--salida", str(salida),
        ]
        subprocess.run(comando, cwd=RAIZ, check=True)
        if salida.read_bytes() != (ENTREGA / "resultados.jsonl").read_bytes():
            raise ValueError("la salida regenerada no coincide byte a byte")
    return "salida reproducida y coincidente byte a byte"


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida el paquete final de CodeFest")
    parser.add_argument("--ejecutar-generador", action="store_true")
    args = parser.parse_args()
    fallos = 0
    dir_encoder: Path | None = None

    comprobaciones = [
        ("Estructura de entrega", lambda: validar_estructura()),
        ("resultados.jsonl", validar_resultados),
        ("generador.py", validar_generador),
        ("informe_tecnico.pdf", validar_pdf),
    ]
    for nombre, funcion in comprobaciones:
        try:
            detalle = funcion()
            print(f"[OK] {nombre}: {detalle}")
            if nombre == "Estructura de entrega":
                dir_encoder = Path(detalle)
        except Exception as exc:
            fallos += 1
            print(f"[FALLO] {nombre}: {exc}")

    if dir_encoder is not None:
        try:
            print(f"[OK] Base vectorial: {validar_base(dir_encoder)}")
        except Exception as exc:
            fallos += 1
            print(f"[FALLO] Base vectorial: {exc}")
        if args.ejecutar_generador and fallos == 0:
            try:
                print(f"[OK] Reproducción: {reproducir(dir_encoder)}")
            except Exception as exc:
                fallos += 1
                print(f"[FALLO] Reproducción: {exc}")

    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
