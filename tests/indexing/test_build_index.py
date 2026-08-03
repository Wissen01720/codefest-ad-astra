import json

import faiss
import numpy as np
import pytest

from codefest_ad_astra.indexing.build_index import construir_indice, guardar_base_vectorial


def _vectores_normalizados(n: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    v = rng.random((n, dim)).astype("float32")
    normas = np.linalg.norm(v, axis=1, keepdims=True)
    return v / normas


def _fragmentos_de_prueba(n: int) -> list[dict]:
    return [
        {
            "doc_id": f"DOC-{i}",
            "chunk_id": f"DOC-{i}-chunk-000",
            "fuente": f"archivo-{i}.pdf",
            "formato": "pdf",
            "fenomeno": 1,
            "posicion": 0,
            "num_tokens": 10,
            "texto": f"Texto del fragmento {i}.",
        }
        for i in range(n)
    ]


def test_construir_indice_es_flatip_y_contiene_todos_los_vectores():
    vectores = _vectores_normalizados(5)
    indice = construir_indice(vectores)

    assert isinstance(indice, faiss.IndexFlatIP)
    assert indice.ntotal == 5
    assert indice.d == 4


def test_guardar_base_vectorial_escribe_index_y_metadata_en_orden(tmp_path):
    vectores = _vectores_normalizados(3)
    fragmentos = _fragmentos_de_prueba(3)

    guardar_base_vectorial(fragmentos, vectores, tmp_path, nombre_encoder="bge-m3")

    carpeta = tmp_path / "encoder_bge-m3"
    assert (carpeta / "index.faiss").exists()
    assert (carpeta / "metadata.jsonl").exists()

    indice_leido = faiss.read_index(str(carpeta / "index.faiss"))
    assert indice_leido.ntotal == 3

    lineas = (carpeta / "metadata.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 3
    for i, linea in enumerate(lineas):
        datos = json.loads(linea)
        assert datos["doc_id"] == f"DOC-{i}"  # orden de línea == orden de inserción == id interno FAISS


def test_guardar_base_vectorial_falla_si_hay_descuadre_fragmentos_vectores(tmp_path):
    vectores = _vectores_normalizados(2)
    fragmentos = _fragmentos_de_prueba(3)

    with pytest.raises(ValueError, match="Descuadre"):
        guardar_base_vectorial(fragmentos, vectores, tmp_path, nombre_encoder="bge-m3")


def test_indice_recuperado_devuelve_el_vecino_mas_cercano_esperado():
    vectores = _vectores_normalizados(4)
    indice = construir_indice(vectores)

    puntuaciones, ids = indice.search(vectores[[0]], k=1)

    assert ids[0][0] == 0  # el vecino más cercano de un vector es él mismo
    assert puntuaciones[0][0] == pytest.approx(1.0, abs=1e-4)  # coseno consigo mismo == 1
