from __future__ import annotations

import numpy as np
import pytest

from codefest_ad_astra.indexing.encoders import (
    FakeEncoder,
    crear_encoder,
    normalizar,
    prefijos_para_modelo,
    slug_modelo,
)


def test_slug_modelo_usa_el_ultimo_segmento():
    assert slug_modelo("BAAI/bge-m3") == "bge-m3"
    assert slug_modelo("intfloat/multilingual-e5-large") == "multilingual-e5-large"
    assert slug_modelo("sentence-transformers/paraphrase-MiniLM-L3-v2") == "paraphrase-MiniLM-L3-v2"


def test_slug_modelo_sanea_caracteres_invalidos():
    assert slug_modelo("org/modelo raro:v2") == "modelo-raro-v2"


def test_slug_modelo_rechaza_nombre_vacio():
    with pytest.raises(ValueError):
        slug_modelo("///")


def test_prefijos_e5_requieren_query_y_passage():
    assert prefijos_para_modelo("intfloat/multilingual-e5-large") == ("query: ", "passage: ")


def test_prefijos_bge_m3_vacios():
    assert prefijos_para_modelo("BAAI/bge-m3") == ("", "")


def test_prefijos_modelo_desconocido_vacios():
    assert prefijos_para_modelo("algun/modelo-nuevo") == ("", "")


def test_normalizar_produce_norma_unitaria():
    vectores = np.array([[3.0, 4.0], [1.0, 0.0], [-2.0, 0.0]])
    normalizados = normalizar(vectores)
    assert np.allclose(np.linalg.norm(normalizados, axis=1), 1.0)
    assert normalizados.dtype == np.float32


def test_normalizar_no_divide_por_cero():
    normalizados = normalizar(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert np.all(np.isfinite(normalizados))
    assert np.allclose(normalizados[0], 0.0)


def test_normalizar_rechaza_matriz_no_2d():
    with pytest.raises(ValueError):
        normalizar(np.array([1.0, 2.0, 3.0]))


def test_fake_encoder_es_determinista_y_normalizado():
    encoder = FakeEncoder(dimension=8)
    a = encoder.codificar_pasajes(["hola mundo", "otro texto"])
    b = encoder.codificar_pasajes(["hola mundo", "otro texto"])
    assert a.shape == (2, 8)
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_fake_encoder_textos_distintos_dan_vectores_distintos():
    encoder = FakeEncoder(dimension=8)
    vectores = encoder.codificar_pasajes(["uno", "dos"])
    assert not np.allclose(vectores[0], vectores[1])


def test_fake_encoder_lista_vacia_devuelve_matriz_vacia():
    encoder = FakeEncoder(dimension=8)
    vectores = encoder.codificar_pasajes([])
    assert vectores.shape == (0, 8)


def test_crear_encoder_fake_no_descarga_pesos():
    encoder = crear_encoder("fake")
    assert isinstance(encoder, FakeEncoder)
