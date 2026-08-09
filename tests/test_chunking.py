from __future__ import annotations

import json
from pathlib import Path

import pytest

from codefest_ad_astra.ingest.chunking import (
    ChunkingConfig,
    FakeTokenCounter,
    build_chunks_for_document,
    process_chunking,
    split_block_into_sentences,
    split_text_into_blocks,
)


class TokenCounterByLength:
    def count(self, text: str) -> int:
        return len(text)


def make_config(**overrides) -> ChunkingConfig:
    config = ChunkingConfig(
        input_path=Path("input.jsonl"),
        output_path=Path("output.jsonl"),
        error_path=Path("errors.jsonl"),
        tokenizer_model="dummy",
        max_tokens=10,
        max_words=10,
        overlap_sentences=1,
        on_oversize="fail",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_split_block_heading_and_sentence_boundaries() -> None:
    texto = "Título de sección\n\nPrimera oración. Segunda oración. Dr. Pérez llegó.\n\nOtra oración."
    blocks = split_text_into_blocks(texto)

    assert len(blocks) == 2
    assert blocks[0].is_heading_block

    sentences = split_block_into_sentences(blocks[0])
    assert sentences[0].is_heading
    assert sentences[0].text == "Título de sección\n"
    assert sentences[1].text.strip() == "Primera oración."
    assert sentences[2].text.strip() == "Segunda oración."
    assert sentences[3].text.strip() == "Dr. Pérez llegó."


def test_build_chunks_for_document_creates_overlapping_chunks() -> None:
    record = {
        "doc_id": "doc1",
        "fuente": "fuente",
        "formato": "txt",
        "fenomeno": 1,
        "idioma": "es",
        "texto": "Primera oración. Segunda oración. Tercera oración.",
    }
    config = ChunkingConfig(
        input_path=Path("input.jsonl"),
        output_path=Path("output.jsonl"),
        error_path=Path("errors.jsonl"),
        tokenizer_model="dummy",
        max_tokens=4,
        max_words=4,
        overlap_sentences=1,
        on_oversize="fail",
    )

    chunks, errors = build_chunks_for_document(1, record, config, FakeTokenCounter())

    assert not errors
    assert [chunk.posicion for chunk in chunks] == [0, 1]
    assert chunks[0].chunk_id == "doc1-chunk-0000"
    assert chunks[1].chunk_id == "doc1-chunk-0001"
    assert chunks[0].texto.strip() == "Primera oración. Segunda oración."
    assert chunks[1].texto.strip() == "Segunda oración. Tercera oración."


def test_build_chunks_for_document_reports_oversize_sentence_on_skip_document() -> None:
    record = {
        "doc_id": "doc2",
        "fuente": "fuente",
        "formato": "txt",
        "fenomeno": 2,
        "idioma": "es",
        "texto": "Oración oversize que excede los límites.",
    }
    config = ChunkingConfig(
        input_path=Path("input.jsonl"),
        output_path=Path("output.jsonl"),
        error_path=Path("errors.jsonl"),
        tokenizer_model="dummy",
        max_tokens=3,
        max_words=3,
        overlap_sentences=1,
        on_oversize="skip-document",
    )

    chunks, errors = build_chunks_for_document(1, record, config, FakeTokenCounter())

    assert chunks == []
    assert len(errors) == 1
    assert errors[0].motivo == "oración oversize"
    assert errors[0].doc_id == "doc2"
    assert errors[0].num_palabras > config.max_words


def test_build_chunks_for_document_preserves_unlisted_formato_and_idioma() -> None:
    record = {
        "doc_id": "doc_unlisted",
        "fuente": "fuente",
        "formato": "parquet",
        "fenomeno": 1,
        "idioma": "ca",
        "texto": "Primera oración. Segunda oración.",
    }
    config = ChunkingConfig(
        input_path=Path("input.jsonl"),
        output_path=Path("output.jsonl"),
        error_path=Path("errors.jsonl"),
        tokenizer_model="dummy",
        max_tokens=10,
        max_words=10,
        overlap_sentences=1,
        on_oversize="fail",
    )

    chunks, errors = build_chunks_for_document(1, record, config, FakeTokenCounter())
    assert not errors
    assert len(chunks) == 1
    assert chunks[0].formato == "parquet"
    assert chunks[0].idioma == "ca"


def test_build_chunks_for_document_rejects_missing_fields() -> None:
    required_fields = ["doc_id", "fuente", "formato", "idioma", "texto"]
    for field in required_fields:
        record = {
            "doc_id": "doc_missing",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Una oración válida.",
        }
        record[field] = ""
        with pytest.raises(Exception, match=field):
            build_chunks_for_document(1, record, make_config(), FakeTokenCounter())


def test_build_chunks_for_document_rejects_wrong_types() -> None:
    record = {
        "doc_id": "doc_type",
        "fuente": "fuente",
        "formato": 123,
        "fenomeno": "1",
        "idioma": ["es"],
        "texto": True,
    }
    with pytest.raises(Exception):
        build_chunks_for_document(1, record, make_config(), FakeTokenCounter())


def test_build_chunks_for_document_rejects_invalid_fenomeno() -> None:
    record = {
        "doc_id": "doc_bad_fen",
        "fuente": "fuente",
        "formato": "txt",
        "fenomeno": 4,
        "idioma": "es",
        "texto": "Una oración.",
    }
    with pytest.raises(Exception, match="fenomeno debe ser 1, 2 o 3"):
        build_chunks_for_document(1, record, make_config(), FakeTokenCounter())


def test_build_chunks_for_document_slices_match_original_text() -> None:
    record = {
        "doc_id": "doc_slices",
        "fuente": "fuente",
        "formato": "txt",
        "fenomeno": 1,
        "idioma": "es",
        "texto": "Texto exacto. Sigue exacto.",
    }
    config = make_config()

    chunks, _ = build_chunks_for_document(1, record, config, FakeTokenCounter())
    for chunk in chunks:
        assert chunk.texto == record["texto"][chunk.char_start:chunk.char_end]
        assert 0 <= chunk.char_start < chunk.char_end <= len(record["texto"])


def test_build_chunks_for_document_deterministic_ids() -> None:
    record = {
        "doc_id": "doc_deterministic",
        "fuente": "fuente",
        "formato": "txt",
        "fenomeno": 1,
        "idioma": "es",
        "texto": "Primera oración. Segunda oración.",
    }
    config = make_config()

    chunks1, _ = build_chunks_for_document(1, record, config, FakeTokenCounter())
    chunks2, _ = build_chunks_for_document(1, record, config, FakeTokenCounter())
    assert [chunk.chunk_id for chunk in chunks1] == [chunk.chunk_id for chunk in chunks2]
    assert [chunk.posicion for chunk in chunks1] == [chunk.posicion for chunk in chunks2]


def test_build_chunks_for_document_skip_document_oversize() -> None:
    record = {
        "doc_id": "doc_skip",
        "fuente": "fuente",
        "formato": "txt",
        "fenomeno": 1,
        "idioma": "es",
        "texto": "Oración oversize que excede los límites.",
    }
    config = make_config(max_tokens=1, max_words=1, on_oversize="skip-document")

    chunks, errors = build_chunks_for_document(1, record, config, FakeTokenCounter())
    assert chunks == []
    assert len(errors) == 1
    assert errors[0].motivo == "oración oversize"


def test_build_chunks_for_document_overlap_zero() -> None:
    record = {
        "doc_id": "doc_overlap_zero",
        "fuente": "fuente",
        "formato": "txt",
        "fenomeno": 1,
        "idioma": "es",
        "texto": "A. B. C. D.",
    }
    config = make_config(max_tokens=10, max_words=10, overlap_sentences=0)

    chunks, errors = build_chunks_for_document(1, record, config, FakeTokenCounter())
    assert not errors
    assert [chunk.texto.strip() for chunk in chunks] == ["A. B. C. D."]
    assert [chunk.posicion for chunk in chunks] == [0]


def test_build_chunks_for_document_combines_short_paragraphs_in_same_chunk() -> None:
    record = {
        "doc_id": "doc_paragraphs",
        "fuente": "fuente",
        "formato": "txt",
        "fenomeno": 1,
        "idioma": "es",
        "texto": "Primer párrafo.\n\nSegundo párrafo.",
    }
    config = make_config(max_tokens=100, max_words=100)

    chunks, errors = build_chunks_for_document(1, record, config, FakeTokenCounter())
    assert not errors
    assert len(chunks) == 1
    assert chunks[0].texto == record["texto"][chunks[0].char_start:chunks[0].char_end]
    assert "Primer párrafo" in chunks[0].texto and "Segundo párrafo" in chunks[0].texto


def test_build_chunks_for_document_preserves_headers_lists_and_bullets() -> None:
    record = {
        "doc_id": "doc_lists",
        "fuente": "fuente",
        "formato": "txt",
        "fenomeno": 1,
        "idioma": "es",
        "texto": (
            "Sección\n\n"
            "1. Item uno.\n"
            "2. Item dos.\n"
            "- Viñeta uno.\n"
            "- Viñeta dos.\n"
            "Otra frase."
        ),
    }
    config = make_config(max_tokens=100, max_words=100)

    chunks, errors = build_chunks_for_document(1, record, config, FakeTokenCounter())
    assert not errors
    assert len(chunks) == 1
    assert "1. Item uno." in chunks[0].texto
    assert "- Viñeta uno." in chunks[0].texto
    assert chunks[0].texto == record["texto"][chunks[0].char_start:chunks[0].char_end]


def test_build_chunks_for_document_oversize_by_tokens_and_words() -> None:
    record = {
        "doc_id": "doc_oversize",
        "fuente": "fuente",
        "formato": "txt",
        "fenomeno": 1,
        "idioma": "es",
        "texto": "Una oración muy larga que cuenta como muchos tokens y muchas palabras.",
    }

    for max_tokens, max_words, on_oversize, should_fail in [
        (1, 100, "fail", True),
        (100, 1, "fail", True),
        (1, 100, "skip-document", False),
        (100, 1, "skip-document", False),
    ]:
        config = make_config(max_tokens=max_tokens, max_words=max_words, on_oversize=on_oversize)
        if should_fail:
            with pytest.raises(Exception):
                build_chunks_for_document(1, record, config, TokenCounterByLength())
        else:
            chunks, errors = build_chunks_for_document(1, record, config, TokenCounterByLength())
            assert chunks == []
            assert len(errors) == 1
            assert errors[0].motivo == "oración oversize"


def test_process_chunking_multiple_documents_preserve_order_and_chunk_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "multi.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"
    records = [
        {
            "doc_id": "order1",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Primera oración. Segunda oración.",
        },
        {
            "doc_id": "order2",
            "fuente": "fuente",
            "formato": "parquet",
            "fenomeno": 1,
            "idioma": "und",
            "texto": "Otra oración. Y otra más.",
        },
    ]
    input_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    config = make_config(input_path=input_path, output_path=output_path, error_path=error_path)
    assert process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter()) == 0

    chunks = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line]
    assert [chunk["doc_id"] for chunk in chunks] == ["order1", "order2"]
    assert [chunk["posicion"] for chunk in chunks if chunk["doc_id"] == "order1"] == [0]
    assert [chunk["chunk_id"] for chunk in chunks if chunk["doc_id"] == "order2"] == ["order2-chunk-0000"]
    ranges = [(chunk["char_start"], chunk["char_end"]) for chunk in chunks]
    assert all(ranges[i] != ranges[i + 1] for i in range(len(ranges) - 1))


def test_process_chunking_output_is_bytewise_deterministic(tmp_path: Path) -> None:
    input_path = tmp_path / "deterministic.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"
    record = {
        "doc_id": "det1",
        "fuente": "fuente",
        "formato": "txt",
        "fenomeno": 1,
        "idioma": "es",
        "texto": "Primera oración. Segunda oración.",
    }
    input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    config = make_config(input_path=input_path, output_path=output_path, error_path=error_path)
    assert process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter()) == 0
    first = output_path.read_bytes()

    assert process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter()) == 0
    second = output_path.read_bytes()

    assert first == second


def test_process_chunking_failure_removes_temporary_files(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text("{ invalid json }\n", encoding="utf-8")
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"

    created_paths = []
    original_named_temp = __import__('tempfile').NamedTemporaryFile

    def spy_named_temp(*args, **kwargs):
        f = original_named_temp(*args, **kwargs)
        created_paths.append(Path(f.name))
        return f

    import codefest_ad_astra.ingest.chunking as chunking_module
    monkeypatch.setattr(chunking_module.tempfile, 'NamedTemporaryFile', spy_named_temp)

    config = make_config(input_path=input_path, output_path=output_path, error_path=error_path)
    with pytest.raises(Exception):
        process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())

    assert created_paths, "No temporary files were created"
    assert not any(path.exists() for path in created_paths)


def test_process_chunking_writes_output_and_error_files(tmp_path: Path) -> None:
    source_fixture = Path(__file__).resolve().parent / "fixtures" / "mini_corpus.jsonl"
    input_path = tmp_path / "mini_corpus.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"

    input_path.write_text(source_fixture.read_text(), encoding="utf-8")

    config = ChunkingConfig(
        input_path=input_path,
        output_path=output_path,
        error_path=error_path,
        tokenizer_model="dummy",
        max_tokens=10,
        max_words=10,
        overlap_sentences=1,
        on_oversize="skip-document",
    )

    exit_code = process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())
    assert exit_code == 0

    chunks = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line]
    errors = [json.loads(line) for line in error_path.read_text(encoding="utf-8").splitlines() if line]

    assert len(chunks) == 3
    assert [chunk["doc_id"] for chunk in chunks] == ["doc1", "doc1", "doc3"]
    assert errors == [
        {
            "line_number": 2,
            "doc_id": "doc2",
            "fuente": "fuente",
            "idioma": "en",
            "char_start": 41,
            "char_end": 85,
            "num_tokens": 11,
            "num_palabras": 11,
            "max_tokens": 10,
            "max_words": 10,
            "motivo": "oración oversize",
        }
    ]
