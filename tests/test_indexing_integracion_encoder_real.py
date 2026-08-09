"""Integración con un encoder real de sentence-transformers.

Estos tests validan lo que el `FakeEncoder` no puede: que el wrapper carga un
modelo de verdad, que los vectores salen normalizados y que la recuperación
semántica funciona de punta a punta contra un `IndexFlatIP`.

Usan un modelo pequeño (`paraphrase-MiniLM-L3-v2`, ~68 MB) a propósito: el
objetivo es verificar el *camino de código*, no la calidad del encoder de
producción. Se saltan si el modelo no está en cache y no hay red.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from codefest_ad_astra.indexing.build_index import construir_base_vectorial
from codefest_ad_astra.indexing.encoders import SentenceTransformerEncoder
from codefest_ad_astra.indexing.faiss_store import cargar_base_vectorial
from codefest_ad_astra.indexing.search import Buscador

MODELO_PEQUENO = "sentence-transformers/paraphrase-MiniLM-L3-v2"

pytestmark = pytest.mark.integracion


@pytest.fixture(scope="module")
def encoder():
    try:
        return SentenceTransformerEncoder(MODELO_PEQUENO, device="cpu", mostrar_progreso=False)
    except Exception as exc:  # sin cache local y sin red
        pytest.skip(
            f"No se pudo cargar {MODELO_PEQUENO} ({type(exc).__name__}). "
            "Este test necesita el modelo en cache de HuggingFace o conexión a internet. "
            "Antes de cerrar la Fase 4, córrelo con el modelo disponible."
        )


def test_encoder_real_expone_dimension_y_device(encoder):
    assert encoder.dimension > 0
    assert encoder.device == "cpu"
    assert encoder.max_seq_length == 512


def test_encoder_real_devuelve_vectores_normalizados(encoder):
    vectores = encoder.codificar_pasajes(["primer texto", "segundo texto", "tercer texto"])
    assert vectores.shape == (3, encoder.dimension)
    assert vectores.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectores, axis=1), 1.0, atol=1e-5)


def test_encoder_real_es_reproducible(encoder):
    a = encoder.codificar_pasajes(["texto de prueba"])
    b = encoder.codificar_pasajes(["texto de prueba"])
    assert np.allclose(a, b, atol=1e-6)


def test_similitud_coseno_en_rango_valido(encoder):
    vectores = encoder.codificar_pasajes(["gato", "perro", "satélite en órbita"])
    similitudes = vectores @ vectores.T
    assert np.all(similitudes <= 1.0 + 1e-5)
    assert np.all(similitudes >= -1.0 - 1e-5)


def test_pipeline_completo_recupera_el_fragmento_semanticamente_cercano(tmp_path: Path, encoder):
    textos = [
        "La basura espacial en órbita baja aumenta el riesgo de colisiones entre satélites.",
        "El presupuesto de defensa incorpora sistemas de inteligencia artificial.",
        "Las lluvias afectaron los cultivos de café en la región andina.",
    ]
    registros = [
        {
            "doc_id": f"DOC-{i:03d}",
            "chunk_id": f"DOC-{i:03d}-chunk-000",
            "fuente": f"F2_Espacio/doc_{i}.pdf",
            "formato": "pdf",
            "fenomeno": 2,
            "idioma": "es",
            "posicion": 0,
            "num_tokens": 30,
            "texto": texto,
        }
        for i, texto in enumerate(textos)
    ]
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in registros) + "\n", encoding="utf-8"
    )

    dir_encoder = construir_base_vectorial(
        chunks, tmp_path / "bv", encoder=encoder, tamano_bloque=2
    )
    indice, metadata, manifiesto = cargar_base_vectorial(dir_encoder)

    assert indice.ntotal == 3
    assert manifiesto["dimension"] == encoder.dimension
    assert manifiesto["modelo"] == MODELO_PEQUENO

    buscador = Buscador(dir_encoder, encoder=encoder)
    mejor = buscador.buscar("colisiones de satélites y desechos orbitales", k=1)[0]
    assert mejor.doc_id == "DOC-000", f"recuperó {mejor.doc_id}: {mejor.texto}"


def test_consulta_multilingue_encuentra_el_documento(tmp_path: Path, encoder):
    """El corpus del reto es ES/EN/PT: una consulta debe cruzar idiomas."""
    registros = [
        {
            "doc_id": "DOC-EN",
            "chunk_id": "DOC-EN-chunk-000",
            "fuente": "F2_Espacio/report.html",
            "formato": "html",
            "fenomeno": 2,
            "idioma": "en",
            "posicion": 0,
            "num_tokens": 30,
            "texto": "Space debris in low Earth orbit increases the probability of satellite collisions.",
        },
        {
            "doc_id": "DOC-ES",
            "chunk_id": "DOC-ES-chunk-000",
            "fuente": "F3_Territorio/informe.pdf",
            "formato": "pdf",
            "fenomeno": 3,
            "idioma": "es",
            "posicion": 0,
            "num_tokens": 30,
            "texto": "El desempleo juvenil creció en las zonas rurales del país.",
        },
    ]
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in registros) + "\n", encoding="utf-8"
    )
    dir_encoder = construir_base_vectorial(chunks, tmp_path / "bv", encoder=encoder)

    buscador = Buscador(dir_encoder, encoder=encoder)
    mejor = buscador.buscar("basura espacial en órbita baja terrestre", k=1)[0]
    assert mejor.doc_id == "DOC-EN", (
        f"recuperó {mejor.doc_id}. Con este modelo pequeño el cruce de idiomas es débil; "
        "el encoder de producción (bge-m3) sí es multilingüe nativo."
    )


@pytest.mark.skipif(
    os.environ.get("CODEFEST_TEST_BGE_M3") != "1",
    reason="Pesa ~2.2 GB. Actívalo con CODEFEST_TEST_BGE_M3=1 cuando el modelo esté en cache.",
)
def test_encoder_de_produccion_bge_m3_carga_y_da_1024_dimensiones():
    encoder = SentenceTransformerEncoder("BAAI/bge-m3", device="cpu", mostrar_progreso=False)
    assert encoder.dimension == 1024
    assert encoder.prefijo_consulta == ""
    assert encoder.prefijo_pasaje == ""

    vectores = encoder.codificar_pasajes(
        [
            "La órbita baja terrestre está congestionada.",
            "Low Earth orbit is congested.",
            "A órbita baixa da Terra está congestionada.",
        ]
    )
    assert np.allclose(np.linalg.norm(vectores, axis=1), 1.0, atol=1e-5)

    # Las tres frases dicen lo mismo en ES/EN/PT: deben quedar cerca entre sí.
    similitudes = vectores @ vectores.T
    assert similitudes[0, 1] > 0.7
    assert similitudes[0, 2] > 0.7
