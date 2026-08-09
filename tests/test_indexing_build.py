from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from codefest_ad_astra.indexing.build_index import (
    construir_base_vectorial,
    construir_metadata,
    main,
    preparar_fragmentos,
    sha256_archivo,
)
from codefest_ad_astra.indexing.encoders import FakeEncoder
from codefest_ad_astra.indexing.faiss_store import CAMPOS_OBLIGATORIOS, cargar_base_vectorial


def chunk(i: int, texto: str | None = None, **extra) -> dict:
    registro = {
        "doc_id": f"DOC-{i // 3:03d}",
        "chunk_id": f"DOC-{i // 3:03d}-chunk-{i:04d}",
        "fuente": f"F{(i % 3) + 1}_Fenomeno/archivo_{i // 3}.pdf",
        "formato": "pdf",
        "fenomeno": (i % 3) + 1,
        "idioma": "es",
        "posicion": i % 3,
        "num_tokens": 120 + i,
        "num_palabras": 90 + i,
        "char_start": i * 500,
        "char_end": (i + 1) * 500,
        "texto": texto if texto is not None else f"Fragmento número {i} sobre seguridad espacial.",
    }
    registro.update(extra)
    return registro


def escribir_chunks(path: Path, registros: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in registros) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def chunks_path(tmp_path: Path) -> Path:
    return escribir_chunks(tmp_path / "chunks.jsonl", [chunk(i) for i in range(10)])


def test_preparar_fragmentos_conserva_el_orden_del_archivo(chunks_path: Path):
    fragmentos, descartados = preparar_fragmentos(chunks_path)
    assert [f["chunk_id"] for f in fragmentos] == [chunk(i)["chunk_id"] for i in range(10)]
    assert descartados == []


def test_preparar_fragmentos_descarta_texto_vacio(tmp_path: Path):
    path = escribir_chunks(
        tmp_path / "chunks.jsonl", [chunk(0), chunk(1, texto="   "), chunk(2, texto=""), chunk(3)]
    )
    fragmentos, descartados = preparar_fragmentos(path)
    assert len(fragmentos) == 2
    assert [d["motivo"] for d in descartados] == ["texto_vacio", "texto_vacio"]
    assert [d["linea"] for d in descartados] == [2, 3]


def test_preparar_fragmentos_falla_si_faltan_campos_obligatorios(tmp_path: Path):
    incompleto = chunk(0)
    del incompleto["fenomeno"]
    path = escribir_chunks(tmp_path / "chunks.jsonl", [incompleto])
    with pytest.raises(Exception, match="fenomeno"):
        preparar_fragmentos(path)


def test_preparar_fragmentos_respeta_el_limite(chunks_path: Path):
    fragmentos, _ = preparar_fragmentos(chunks_path, limite=4)
    assert len(fragmentos) == 4


def test_detectar_truncamiento_marca_los_fragmentos_que_exceden_el_limite():
    from codefest_ad_astra.indexing.build_index import detectar_truncamiento

    encoder = FakeEncoder(dimension=8)  # cuenta tokens = palabras, límite 512
    corto = chunk(0, texto="tres palabras aqui.")
    largo = chunk(1, texto=" ".join(["palabra"] * 600) + ".")

    truncados = detectar_truncamiento([corto, largo], encoder)
    assert [t["chunk_id"] for t in truncados] == [largo["chunk_id"]]
    assert truncados[0]["tokens_reales"] == 600
    assert truncados[0]["limite"] == 512


def test_detectar_truncamiento_no_reporta_nada_si_todo_cabe():
    from codefest_ad_astra.indexing.build_index import detectar_truncamiento

    assert detectar_truncamiento([chunk(0), chunk(1)], FakeEncoder(dimension=8)) == []


def test_manifiesto_registra_los_fragmentos_truncados(tmp_path: Path):
    path = escribir_chunks(
        tmp_path / "chunks.jsonl", [chunk(0), chunk(1, texto=" ".join(["palabra"] * 600) + ".")]
    )
    dir_encoder = construir_base_vectorial(path, tmp_path / "bv", encoder=FakeEncoder(dimension=8))
    manifiesto = json.loads((dir_encoder / "manifest.json").read_text(encoding="utf-8"))
    assert manifiesto["fragmentos_truncados_detectados"] == 1


def test_construir_metadata_pone_primero_los_campos_de_la_tabla_1():
    metadata = construir_metadata([chunk(0)])
    assert list(metadata[0])[: len(CAMPOS_OBLIGATORIOS)] == list(CAMPOS_OBLIGATORIOS)


def test_construir_metadata_conserva_campos_extra_de_fase_3():
    metadata = construir_metadata([chunk(0)])
    for extra in ("idioma", "num_palabras", "char_start", "char_end"):
        assert extra in metadata[0]


def test_construir_base_vectorial_escribe_los_dos_archivos(tmp_path: Path, chunks_path: Path):
    salida = tmp_path / "base_vectorial"
    dir_encoder = construir_base_vectorial(
        chunks_path, salida, encoder=FakeEncoder(dimension=8), tamano_bloque=4
    )

    assert dir_encoder == salida / "encoder_encoder-test"
    assert (dir_encoder / "index.faiss").exists()
    assert (dir_encoder / "metadata.jsonl").exists()
    assert (dir_encoder / "manifest.json").exists()


def test_metadata_alineada_con_los_ids_del_indice(tmp_path: Path, chunks_path: Path):
    dir_encoder = construir_base_vectorial(
        chunks_path, tmp_path / "bv", encoder=FakeEncoder(dimension=8), tamano_bloque=3
    )
    indice, metadata, _ = cargar_base_vectorial(dir_encoder)

    assert indice.ntotal == len(metadata) == 10

    # Cada fragmento, re-codificado, debe recuperarse a sí mismo en la posición
    # que dice su metadata.
    encoder = FakeEncoder(dimension=8)
    for esperado, registro in enumerate(metadata):
        vector = encoder.codificar_pasajes([registro["texto"]])
        _, ids = indice.search(vector, 1)
        assert int(ids[0][0]) == esperado


def test_los_vectores_quedan_normalizados(tmp_path: Path, chunks_path: Path):
    dir_encoder = construir_base_vectorial(
        chunks_path, tmp_path / "bv", encoder=FakeEncoder(dimension=8), tamano_bloque=4
    )
    indice, _, _ = cargar_base_vectorial(dir_encoder)
    reconstruidos = np.vstack([indice.reconstruct(i) for i in range(indice.ntotal)])
    assert np.allclose(np.linalg.norm(reconstruidos, axis=1), 1.0, atol=1e-5)


def test_manifiesto_registra_las_decisiones_de_diseno(tmp_path: Path, chunks_path: Path):
    dir_encoder = construir_base_vectorial(
        chunks_path, tmp_path / "bv", encoder=FakeEncoder(dimension=8), tamano_bloque=4
    )
    manifiesto = json.loads((dir_encoder / "manifest.json").read_text(encoding="utf-8"))

    assert manifiesto["tipo_indice"] == "IndexFlatIP"
    assert manifiesto["vectores_normalizados"] is True
    assert manifiesto["dimension"] == 8
    assert manifiesto["num_vectores"] == 10
    assert manifiesto["sha256_entrada"] == sha256_archivo(chunks_path)
    assert manifiesto["indice_parcial"] is False


def test_limite_marca_el_indice_como_parcial(tmp_path: Path, chunks_path: Path):
    dir_encoder = construir_base_vectorial(
        chunks_path, tmp_path / "bv", encoder=FakeEncoder(dimension=8), limite=3
    )
    manifiesto = json.loads((dir_encoder / "manifest.json").read_text(encoding="utf-8"))
    assert manifiesto["indice_parcial"] is True
    assert manifiesto["limite_aplicado"] == 3
    assert manifiesto["num_vectores"] == 3


def test_descartados_se_registran_en_archivo(tmp_path: Path):
    path = escribir_chunks(tmp_path / "chunks.jsonl", [chunk(0), chunk(1, texto="  "), chunk(2)])
    dir_encoder = construir_base_vectorial(path, tmp_path / "bv", encoder=FakeEncoder(dimension=8))

    descartados = (dir_encoder / "descartados.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(descartados) == 1
    assert json.loads(descartados[0])["motivo"] == "texto_vacio"


def test_parciales_se_borran_al_terminar(tmp_path: Path, chunks_path: Path):
    dir_encoder = construir_base_vectorial(
        chunks_path, tmp_path / "bv", encoder=FakeEncoder(dimension=8), tamano_bloque=4
    )
    assert not (dir_encoder / "_parciales").exists()


def test_conservar_parciales_los_deja_en_disco(tmp_path: Path, chunks_path: Path):
    dir_encoder = construir_base_vectorial(
        chunks_path,
        tmp_path / "bv",
        encoder=FakeEncoder(dimension=8),
        tamano_bloque=4,
        conservar_parciales=True,
    )
    shards = sorted((dir_encoder / "_parciales").glob("emb_*.npy"))
    assert len(shards) == 3  # 10 fragmentos en bloques de 4


def test_reanudar_reutiliza_los_bloques_ya_codificados(tmp_path: Path, chunks_path: Path):
    salida = tmp_path / "bv"
    construir_base_vectorial(
        chunks_path,
        salida,
        encoder=FakeEncoder(dimension=8),
        tamano_bloque=4,
        conservar_parciales=True,
    )

    class EncoderQueCuenta(FakeEncoder):
        llamadas = 0

        def codificar_pasajes(self, textos):
            EncoderQueCuenta.llamadas += 1
            return super().codificar_pasajes(textos)

    dir_encoder = construir_base_vectorial(
        chunks_path, salida, encoder=EncoderQueCuenta(dimension=8), tamano_bloque=4, reanudar=True
    )
    assert EncoderQueCuenta.llamadas == 0  # todo venía del checkpoint

    indice, metadata, _ = cargar_base_vectorial(dir_encoder)
    assert indice.ntotal == len(metadata) == 10


def test_reanudar_rechaza_checkpoint_de_otra_entrada(tmp_path: Path, chunks_path: Path):
    salida = tmp_path / "bv"
    construir_base_vectorial(
        chunks_path,
        salida,
        encoder=FakeEncoder(dimension=8),
        tamano_bloque=4,
        conservar_parciales=True,
    )

    otros = escribir_chunks(tmp_path / "otros.jsonl", [chunk(i) for i in range(10, 16)])
    with pytest.raises(SystemExit, match="checkpoint"):
        construir_base_vectorial(
            otros, salida, encoder=FakeEncoder(dimension=8), tamano_bloque=4, reanudar=True
        )


def test_sin_reanudar_descarta_parciales_viejos(tmp_path: Path, chunks_path: Path):
    salida = tmp_path / "bv"
    construir_base_vectorial(
        chunks_path,
        salida,
        encoder=FakeEncoder(dimension=8),
        tamano_bloque=4,
        conservar_parciales=True,
    )
    otros = escribir_chunks(tmp_path / "otros.jsonl", [chunk(i) for i in range(10, 16)])
    dir_encoder = construir_base_vectorial(
        otros, salida, encoder=FakeEncoder(dimension=8), tamano_bloque=4
    )
    indice, metadata, _ = cargar_base_vectorial(dir_encoder)
    assert indice.ntotal == len(metadata) == 6


def test_falla_si_la_entrada_no_existe(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        construir_base_vectorial(
            tmp_path / "no_existe.jsonl", tmp_path / "bv", encoder=FakeEncoder(dimension=8)
        )


def test_falla_si_no_queda_ningun_fragmento(tmp_path: Path):
    path = escribir_chunks(tmp_path / "chunks.jsonl", [chunk(0, texto="   ")])
    with pytest.raises(SystemExit, match="fragmento"):
        construir_base_vectorial(path, tmp_path / "bv", encoder=FakeEncoder(dimension=8))


def test_main_devuelve_cero_con_encoder_fake(tmp_path: Path, chunks_path: Path):
    codigo = main(
        [
            "--entrada", str(chunks_path),
            "--salida", str(tmp_path / "bv"),
            "--modelo", "fake",
            "--tamano-bloque", "4",
        ]
    )
    assert codigo == 0
    assert (tmp_path / "bv" / "encoder_encoder-test" / "index.faiss").exists()


def test_main_devuelve_uno_si_la_entrada_no_existe(tmp_path: Path):
    codigo = main(["--entrada", str(tmp_path / "nada.jsonl"), "--salida", str(tmp_path / "bv"),
                   "--modelo", "fake"])
    assert codigo == 1


def test_cli_ejecutable_como_modulo(tmp_path: Path, chunks_path: Path):
    resultado = subprocess.run(
        [sys.executable, "-m", "codefest_ad_astra.indexing.build_index",
         "--entrada", str(chunks_path), "--salida", str(tmp_path / "bv"), "--modelo", "fake"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env={**__import__("os").environ, "PYTHONPATH": "src"},
    )
    assert resultado.returncode == 0, resultado.stderr
    assert (tmp_path / "bv" / "encoder_encoder-test" / "metadata.jsonl").exists()
