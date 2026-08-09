from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from codefest_ad_astra.indexing.faiss_store import (
    CAMPOS_OBLIGATORIOS,
    IndexAlignmentError,
    MetadataInvalidaError,
    cargar_base_vectorial,
    construir_indice,
    guardar_base_vectorial,
    leer_metadata,
    validar_alineacion,
    validar_campos_obligatorios,
)
from codefest_ad_astra.indexing.encoders import normalizar


def metadata_valida(n: int) -> list[dict]:
    return [
        {
            "doc_id": f"DOC-{i // 2:03d}",
            "chunk_id": f"DOC-{i // 2:03d}-chunk-{i:03d}",
            "fuente": f"F1_Fenomeno/archivo_{i // 2}.pdf",
            "formato": "pdf",
            "fenomeno": 1,
            "posicion": i % 2,
            "num_tokens": 100 + i,
            "texto": f"Texto del fragmento {i}.",
        }
        for i in range(n)
    ]


def vectores_normalizados(n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    return normalizar(rng.standard_normal((n, dim)))


def test_construir_indice_agrega_todos_los_vectores():
    vectores = vectores_normalizados(5)
    indice = construir_indice(vectores)
    assert indice.ntotal == 5
    assert indice.d == 8


def test_construir_indice_rechaza_vectores_sin_normalizar():
    vectores = np.full((3, 4), 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match="normalizados"):
        construir_indice(vectores)


def test_construir_indice_rechaza_conjunto_vacio():
    with pytest.raises(ValueError, match="vacío"):
        construir_indice(np.zeros((0, 8), dtype=np.float32))


def test_indice_plano_recupera_el_vector_identico_primero():
    vectores = vectores_normalizados(20)
    indice = construir_indice(vectores)
    consulta = vectores[7:8]
    scores, ids = indice.search(consulta, 3)
    assert ids[0][0] == 7
    assert scores[0][0] == pytest.approx(1.0, abs=1e-5)


def test_validar_alineacion_detecta_desfase():
    indice = construir_indice(vectores_normalizados(5))
    with pytest.raises(IndexAlignmentError):
        validar_alineacion(indice, metadata_valida(4))


def test_validar_campos_obligatorios_detecta_faltantes():
    registro = {c: "x" for c in CAMPOS_OBLIGATORIOS if c != "num_tokens"}
    with pytest.raises(MetadataInvalidaError, match="num_tokens"):
        validar_campos_obligatorios(registro, linea=1)


def test_guardar_y_cargar_preserva_el_orden(tmp_path: Path):
    vectores = vectores_normalizados(6)
    indice = construir_indice(vectores)
    metadata = metadata_valida(6)

    guardar_base_vectorial(tmp_path, indice, metadata, {"modelo": "fake/encoder-test"})
    indice_leido, metadata_leida, manifiesto = cargar_base_vectorial(tmp_path)

    assert indice_leido.ntotal == 6
    assert manifiesto["modelo"] == "fake/encoder-test"
    assert [m["chunk_id"] for m in metadata_leida] == [m["chunk_id"] for m in metadata]


def test_id_interno_faiss_corresponde_a_la_linea_de_metadata(tmp_path: Path):
    """La invariante que hace utilizable la entrega: id i del índice ↔ línea i."""
    vectores = vectores_normalizados(12)
    indice = construir_indice(vectores)
    metadata = metadata_valida(12)
    guardar_base_vectorial(tmp_path, indice, metadata)

    indice_leido, metadata_leida, _ = cargar_base_vectorial(tmp_path)
    for esperado in range(12):
        _, ids = indice_leido.search(vectores[esperado : esperado + 1], 1)
        id_interno = int(ids[0][0])
        assert id_interno == esperado
        assert metadata_leida[id_interno]["chunk_id"] == metadata[esperado]["chunk_id"]


def test_guardar_rechaza_metadata_desalineada(tmp_path: Path):
    indice = construir_indice(vectores_normalizados(5))
    with pytest.raises(IndexAlignmentError):
        guardar_base_vectorial(tmp_path, indice, metadata_valida(3))
    assert not (tmp_path / "index.faiss").exists()
    assert not (tmp_path / "metadata.jsonl").exists()


def test_metadata_jsonl_tiene_un_objeto_por_linea(tmp_path: Path):
    indice = construir_indice(vectores_normalizados(4))
    guardar_base_vectorial(tmp_path, indice, metadata_valida(4))

    lineas = (tmp_path / "metadata.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lineas) == 4
    for linea in lineas:
        registro = json.loads(linea)
        assert all(campo in registro for campo in CAMPOS_OBLIGATORIOS)


def test_metadata_se_escribe_en_utf8_sin_escapes(tmp_path: Path):
    indice = construir_indice(vectores_normalizados(1))
    metadata = metadata_valida(1)
    metadata[0]["texto"] = "Congestión orbital y órbita baja terrestre"
    guardar_base_vectorial(tmp_path, indice, metadata)

    crudo = (tmp_path / "metadata.jsonl").read_text(encoding="utf-8")
    assert "Congestión orbital y órbita baja terrestre" in crudo


def test_leer_metadata_rechaza_json_invalido(tmp_path: Path):
    path = tmp_path / "metadata.jsonl"
    path.write_text("{ esto no es json }\n", encoding="utf-8")
    with pytest.raises(MetadataInvalidaError):
        leer_metadata(path)


def test_leer_metadata_rechaza_registro_incompleto(tmp_path: Path):
    path = tmp_path / "metadata.jsonl"
    path.write_text(json.dumps({"doc_id": "DOC-1"}) + "\n", encoding="utf-8")
    with pytest.raises(MetadataInvalidaError):
        leer_metadata(path)


def test_cargar_falla_si_falta_el_indice(tmp_path: Path):
    (tmp_path / "metadata.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        cargar_base_vectorial(tmp_path)


def test_indice_es_legible_con_faiss_read_index_estandar(tmp_path: Path):
    """El entregable exige que index.faiss abra con faiss.read_index() a secas."""
    import faiss

    indice = construir_indice(vectores_normalizados(7))
    guardar_base_vectorial(tmp_path, indice, metadata_valida(7))

    recargado = faiss.read_index(str(tmp_path / "index.faiss"))
    assert recargado.ntotal == 7
    assert isinstance(recargado, faiss.IndexFlatIP)
