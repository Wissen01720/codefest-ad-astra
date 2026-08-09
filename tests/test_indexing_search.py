from __future__ import annotations

import json
from pathlib import Path

import pytest

from codefest_ad_astra.indexing.build_index import construir_base_vectorial
from codefest_ad_astra.indexing.encoders import FakeEncoder
from codefest_ad_astra.indexing.search import Buscador, main

TEXTOS = [
    "La congestión de la órbita baja terrestre aumenta el riesgo de colisiones.",
    "Los satélites obsoletos y los restos de cohetes forman la basura espacial.",
    "La inteligencia artificial se adopta en el sector defensa a distinto ritmo.",
    "Los observatorios de derechos humanos documentan dinámicas territoriales.",
    "La gobernanza del espacio ultraterrestre depende de acuerdos multilaterales.",
]


def escribir_chunks(path: Path) -> Path:
    registros = [
        {
            "doc_id": f"DOC-{i:03d}",
            "chunk_id": f"DOC-{i:03d}-chunk-000",
            "fuente": f"F2_Espacio/informe_{i}.pdf",
            "formato": "pdf",
            "fenomeno": 2,
            "idioma": "es",
            "posicion": 0,
            "num_tokens": 40,
            "texto": texto,
        }
        for i, texto in enumerate(TEXTOS)
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in registros) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def base(tmp_path: Path) -> Path:
    chunks = escribir_chunks(tmp_path / "chunks.jsonl")
    return construir_base_vectorial(chunks, tmp_path / "bv", encoder=FakeEncoder(dimension=16))


def test_buscar_devuelve_k_resultados(base: Path):
    buscador = Buscador(base, encoder=FakeEncoder(dimension=16))
    resultados = buscador.buscar("basura espacial", k=3)
    assert len(resultados) == 3
    assert [r.rank for r in resultados] == [1, 2, 3]


def test_scores_ordenados_de_mayor_a_menor(base: Path):
    buscador = Buscador(base, encoder=FakeEncoder(dimension=16))
    scores = [r.score for r in buscador.buscar("órbita", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_consulta_identica_a_un_fragmento_lo_recupera_primero(base: Path):
    """Verifica de punta a punta que el vector recuperado corresponde a su texto."""
    buscador = Buscador(base, encoder=FakeEncoder(dimension=16))
    resultados = buscador.buscar(TEXTOS[1], k=1)
    assert resultados[0].texto == TEXTOS[1]
    assert resultados[0].doc_id == "DOC-001"
    assert resultados[0].score == pytest.approx(1.0, abs=1e-5)


def test_k_mayor_que_el_indice_no_falla(base: Path):
    buscador = Buscador(base, encoder=FakeEncoder(dimension=16))
    resultados = buscador.buscar("cualquier cosa", k=50)
    assert len(resultados) == len(TEXTOS)


def test_k_invalido_falla(base: Path):
    buscador = Buscador(base, encoder=FakeEncoder(dimension=16))
    with pytest.raises(ValueError):
        buscador.buscar("consulta", k=0)


def test_resultado_expone_metadata_completa(base: Path):
    buscador = Buscador(base, encoder=FakeEncoder(dimension=16))
    resultado = buscador.buscar("órbita", k=1)[0]
    for campo in ("doc_id", "chunk_id", "fuente", "formato", "fenomeno", "posicion", "num_tokens", "texto"):
        assert campo in resultado.metadata


def test_buscar_lote_equivale_a_buscar_una_por_una(base: Path):
    buscador = Buscador(base, encoder=FakeEncoder(dimension=16))
    consultas = ["basura espacial", "inteligencia artificial"]
    en_lote = buscador.buscar_lote(consultas, k=3)
    individuales = [buscador.buscar(c, k=3) for c in consultas]

    assert len(en_lote) == 2
    for lote, individual in zip(en_lote, individuales):
        assert [r.chunk_id for r in lote] == [r.chunk_id for r in individual]


def test_buscar_lote_vacio_devuelve_vacio(base: Path):
    buscador = Buscador(base, encoder=FakeEncoder(dimension=16))
    assert buscador.buscar_lote([], k=3) == []


def test_encoder_de_dimension_distinta_es_rechazado(base: Path):
    with pytest.raises(ValueError, match="dimensión"):
        Buscador(base, encoder=FakeEncoder(dimension=32))


def test_falla_si_el_manifiesto_no_dice_el_modelo(tmp_path: Path, base: Path):
    (base / "manifest.json").write_text(json.dumps({"tipo_indice": "IndexFlatIP"}), encoding="utf-8")
    with pytest.raises(ValueError, match="modelo"):
        Buscador(base)


def test_cli_json_imprime_resultados(base: Path, capsys):
    codigo = main(["--base", str(base), "--consulta", TEXTOS[0], "-k", "2", "--json"])
    assert codigo == 0
    salida = json.loads(capsys.readouterr().out)
    assert len(salida) == 2
    assert salida[0]["texto"] == TEXTOS[0]


def test_cli_texto_legible(base: Path, capsys):
    codigo = main(["--base", str(base), "--consulta", "órbita baja", "-k", "2"])
    assert codigo == 0
    salida = capsys.readouterr().out
    assert "score=" in salida
    assert "fragmentos indexados" in salida


def test_cli_devuelve_uno_si_la_base_no_existe(tmp_path: Path, capsys):
    codigo = main(["--base", str(tmp_path / "no_existe"), "--consulta", "algo"])
    assert codigo == 1
    assert "ERROR" in capsys.readouterr().out
