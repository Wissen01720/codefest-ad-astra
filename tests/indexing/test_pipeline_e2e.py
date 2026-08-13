"""Prueba de extremo a extremo de la Fase 4 + Fase 6 con `FakeEncoder`.

No descarga ningún modelo real (usa `codefest_ad_astra.indexing.encoders.FakeEncoder`,
determinístico vía hash), así que corre rápido y offline. Cubre la cadena
completa que corre en producción y que hasta ahora no tenía ningún test:

    chunks.jsonl -> build_index.construir_base_vectorial
                 -> faiss_store.{construir_indice,guardar_base_vectorial,cargar_base_vectorial}
                 -> search.Buscador
                 -> retrieval.recuperar.recuperar_documentos

Cada paso reusa el módulo real de producción, no una reimplementación de prueba,
así que un cambio que rompa la integración entre módulos (p. ej. un nombre de
campo distinto entre lo que build_index.py escribe y lo que search.py espera)
se detecta aquí.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from codefest_ad_astra.indexing.build_index import construir_base_vectorial
from codefest_ad_astra.indexing.encoders import FakeEncoder
from codefest_ad_astra.indexing.faiss_store import (
    IndexAlignmentError,
    MetadataInvalidaError,
    cargar_base_vectorial,
)
from codefest_ad_astra.indexing.search import Buscador
from codefest_ad_astra.retrieval.recuperar import recuperar_documentos


CHUNKS_EJEMPLO = [
    dict(doc_id="D1", chunk_id="D1-0", fuente="f1.csv", formato="csv", fenomeno=1,
         idioma="es", posicion=0, num_tokens=5, texto="seguridad espacial en orbita baja"),
    dict(doc_id="D1", chunk_id="D1-1", fuente="f1.csv", formato="csv", fenomeno=1,
         idioma="es", posicion=1, num_tokens=5, texto="riesgos de colision satelital"),
    dict(doc_id="D2", chunk_id="D2-0", fuente="f2.pdf", formato="pdf", fenomeno=2,
         idioma="en", posicion=0, num_tokens=5, texto="inteligencia artificial en defensa"),
    dict(doc_id="D3", chunk_id="D3-0", fuente="f3.pdf", formato="pdf", fenomeno=3,
         idioma="es", posicion=0, num_tokens=3, texto="   "),  # texto vacío -> se descarta
]


def _escribir_chunks(tmp_path: Path, registros: list[dict]) -> Path:
    ruta = tmp_path / "chunks.jsonl"
    with open(ruta, "w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return ruta


def test_pipeline_completo_construye_busca_y_agrega_por_documento(tmp_path: Path) -> None:
    ruta_chunks = _escribir_chunks(tmp_path, CHUNKS_EJEMPLO)
    fake = FakeEncoder(dimension=8)

    dir_encoder = construir_base_vectorial(ruta_chunks, tmp_path / "base_vectorial", encoder=fake)

    # El fragmento de solo-espacios se descarta, no se indexa ni rompe nada.
    assert (dir_encoder / "descartados.jsonl").exists()
    descartados = [json.loads(l) for l in open(dir_encoder / "descartados.jsonl", encoding="utf-8")]
    assert len(descartados) == 1
    assert descartados[0]["doc_id"] == "D3"

    indice, metadata, manifiesto = cargar_base_vectorial(dir_encoder)
    assert indice.ntotal == 3
    assert len(metadata) == 3
    assert manifiesto["modelo"] == fake.nombre
    assert manifiesto["num_vectores"] == 3
    assert manifiesto["num_fragmentos_descartados"] == 1

    # Buscador: debe leer el manifiesto y reconstruir un encoder compatible,
    # o aceptar uno ya construido (evita cargar pesos reales en el test).
    buscador = Buscador(dir_encoder, encoder=fake)
    resultados = buscador.buscar("satelites y colisiones en orbita", k=5)
    assert len(resultados) == 3
    assert resultados[0].rank == 1
    # Los scores deben venir ordenados de mayor a menor (coseno descendente).
    scores = [r.score for r in resultados]
    assert scores == sorted(scores, reverse=True)

    # Fase 6: agregación a nivel de documento reutilizando el mismo Buscador.
    documentos = recuperar_documentos(
        dir_encoder, "satelites y colisiones en orbita",
        buscador=buscador, k_chunks=5, top_documentos=2,
    )
    assert len(documentos) <= 2
    ids_vistos = {d.doc_id for d in documentos}
    assert ids_vistos <= {"D1", "D2"}
    for doc in documentos:
        # cada chunk agregado debe pertenecer al documento que lo agrupa
        assert all(c.metadata["doc_id"] == doc.doc_id for c in doc.chunks)
        # los chunks dentro de un documento vienen ordenados por score
        chunk_scores = [c.score for c in doc.chunks]
        assert chunk_scores == sorted(chunk_scores, reverse=True)


def test_descuadre_metadata_vectores_no_pasa_silencioso(tmp_path: Path) -> None:
    """Si algo corrompe la alineación índice<->metadata, cargar_base_vectorial
    debe fallar ruidosamente, nunca devolver resultados desalineados."""
    ruta_chunks = _escribir_chunks(tmp_path, CHUNKS_EJEMPLO[:2])
    fake = FakeEncoder(dimension=8)
    dir_encoder = construir_base_vectorial(ruta_chunks, tmp_path / "base_vectorial", encoder=fake)

    # Truncar metadata.jsonl a mano para simular una corrupción externa.
    ruta_metadata = dir_encoder / "metadata.jsonl"
    lineas = ruta_metadata.read_text(encoding="utf-8").splitlines()
    ruta_metadata.write_text(lineas[0] + "\n", encoding="utf-8")

    with pytest.raises(IndexAlignmentError):
        cargar_base_vectorial(dir_encoder)


def test_chunk_sin_campo_obligatorio_se_rechaza_antes_de_indexar(tmp_path: Path) -> None:
    malos = [dict(CHUNKS_EJEMPLO[0])]
    del malos[0]["fenomeno"]
    ruta_chunks = _escribir_chunks(tmp_path, malos)
    fake = FakeEncoder(dimension=8)

    with pytest.raises(MetadataInvalidaError):
        construir_base_vectorial(ruta_chunks, tmp_path / "base_vectorial", encoder=fake)


def test_buscador_rechaza_dimension_incompatible(tmp_path: Path) -> None:
    ruta_chunks = _escribir_chunks(tmp_path, CHUNKS_EJEMPLO[:2])
    dir_encoder = construir_base_vectorial(
        ruta_chunks, tmp_path / "base_vectorial", encoder=FakeEncoder(dimension=8)
    )

    otro_encoder = FakeEncoder(dimension=16)
    with pytest.raises(ValueError):
        Buscador(dir_encoder, encoder=otro_encoder)


def test_reanudar_desde_checkpoint_produce_los_mismos_vectores(tmp_path: Path) -> None:
    """Simula una corrida interrumpida: codifica solo el primer bloque, deja
    el checkpoint en disco y confirma que --reanudar retoma en vez de repetir
    desde cero, con el mismo resultado final que una corrida sin interrupción."""
    from codefest_ad_astra.indexing.build_index import (
        NOMBRE_PARCIALES,
        codificar_por_bloques,
        preparar_fragmentos,
    )

    ruta_chunks = _escribir_chunks(tmp_path, CHUNKS_EJEMPLO)
    fake = FakeEncoder(dimension=8)
    fragmentos, _ = preparar_fragmentos(ruta_chunks)

    dir_parciales = tmp_path / NOMBRE_PARCIALES
    firma = {"sha256_entrada": "x", "modelo": fake.nombre, "dimension": fake.dimension,
              "num_fragmentos": len(fragmentos)}

    # Primera pasada: bloques de 1, se interrumpe después del primer bloque
    # (se simula no llamando de nuevo, el checkpoint ya quedó en disco).
    _ = codificar_por_bloques(
        fragmentos[:1], fake, dir_parciales, tamano_bloque=1, firma=firma, reanudar=False
    )

    # Segunda pasada "reanudando": mismos fragmentos completos, debe detectar
    # el bloque ya hecho y solo codificar los que faltan.
    vectores_reanudado = codificar_por_bloques(
        fragmentos, fake, dir_parciales, tamano_bloque=1, firma=firma, reanudar=True
    )

    vectores_directo = fake.codificar_pasajes([f["texto"] for f in fragmentos])
    np.testing.assert_allclose(vectores_reanudado, vectores_directo, atol=1e-6)