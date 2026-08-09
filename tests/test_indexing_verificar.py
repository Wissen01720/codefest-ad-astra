from __future__ import annotations

import json
from pathlib import Path

import pytest

from codefest_ad_astra.indexing.build_index import construir_base_vectorial
from codefest_ad_astra.indexing.encoders import FakeEncoder
from codefest_ad_astra.indexing.verificar import main, verificar


def chunk(i: int, **extra) -> dict:
    registro = {
        "doc_id": f"DOC-{i // 2:03d}",
        "chunk_id": f"DOC-{i // 2:03d}-chunk-{i:04d}",
        "fuente": f"F{(i % 3) + 1}_Fenomeno/archivo_{i // 2}.pdf",
        "formato": "pdf",
        "fenomeno": (i % 3) + 1,
        "idioma": ["es", "en", "pt"][i % 3],
        "posicion": i % 2,
        "num_tokens": 100 + i,
        "texto": f"Fragmento {i} sobre el entorno orbital.",
    }
    registro.update(extra)
    return registro


def construir(tmp_path: Path, registros: list[dict], **kwargs) -> Path:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in registros) + "\n", encoding="utf-8"
    )
    return construir_base_vectorial(chunks, tmp_path / "bv", encoder=FakeEncoder(dimension=8), **kwargs)


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return construir(tmp_path, [chunk(i) for i in range(9)])


def test_base_correcta_no_reporta_errores(base: Path):
    errores, _, _ = verificar(base)
    assert errores == []


def test_estadisticas_describen_el_contenido(base: Path):
    _, _, stats = verificar(base)
    assert stats["num_vectores"] == 9
    assert stats["dimension"] == 8
    assert stats["tipo_indice"] == "IndexFlatIP"
    assert stats["num_documentos"] == 5
    assert sum(stats["por_idioma"].values()) == 9
    assert sum(stats["por_fenomeno"].values()) == 9


def test_detecta_metadata_desalineada(base: Path):
    """Quitar una línea de metadata debe hacer fallar la verificación."""
    path = base / "metadata.jsonl"
    lineas = path.read_text(encoding="utf-8").strip().split("\n")
    path.write_text("\n".join(lineas[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(Exception, match="metadata"):
        verificar(base)


def test_detecta_indice_parcial(tmp_path: Path):
    base = construir(tmp_path, [chunk(i) for i in range(9)], limite=4)
    errores, _, _ = verificar(base)
    assert any("PARCIAL" in e for e in errores)


def test_advierte_fragmentos_sobre_250_palabras(tmp_path: Path):
    largo = chunk(0, texto=" ".join(["palabra"] * 300) + ".")
    base = construir(tmp_path, [largo, chunk(1)])
    _, advertencias, stats = verificar(base)
    assert stats["fragmentos_sobre_250_palabras"] == 1
    assert any("250 palabras" in a for a in advertencias)


def test_advierte_chunk_ids_duplicados(tmp_path: Path):
    base = construir(tmp_path, [chunk(0), chunk(1, chunk_id=chunk(0)["chunk_id"])])
    _, advertencias, _ = verificar(base)
    assert any("repetidos" in a for a in advertencias)


def test_main_devuelve_cero_en_base_valida(base: Path, capsys):
    assert main(["--base", str(base)]) == 0
    assert "OK:" in capsys.readouterr().out


def test_main_devuelve_uno_en_indice_parcial(tmp_path: Path, capsys):
    base = construir(tmp_path, [chunk(i) for i in range(9)], limite=3)
    assert main(["--base", str(base)]) == 1
    assert "ERRORES BLOQUEANTES" in capsys.readouterr().out


def test_main_devuelve_uno_si_no_existe(tmp_path: Path, capsys):
    assert main(["--base", str(tmp_path / "nada")]) == 1
    assert "ERROR" in capsys.readouterr().out
