from pathlib import Path
import json
import pytest

from codefest_ad_astra.ingest.chunking import (
    build_chunks_for_document,
    ChunkingConfig,
    FakeTokenCounter,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_WORDS,
    FatalChunkingError,
    process_chunking,
    create_transformer_token_counter,
)


class WeirdTokenCounter(FakeTokenCounter):
    def count(self, text: str) -> int:
        # simulate a tokenizer that counts ~1 token per 10 chars + words
        return max(1, len(text) // 10)


def test_structured_pair_split_keeps_pairs():
    text = "ciudad: Bogotá D.C. poblacion: 7000000 area_km2: 1587"
    record = {
        "doc_id": "doc_csv",
        "fuente": "f",
        "formato": "csv",
        "fenomeno": 1,
        "idioma": "es",
        "texto": text,
    }
    config = ChunkingConfig(
        input_path=Path("in"),
        output_path=Path("out"),
        error_path=Path("err"),
        tokenizer_model="fake",
        max_tokens=10,
        max_words=5,
        overlap_sentences=0,
        on_oversize="fail",
    )
    warnings: list[dict] = []
    chunks, errors = build_chunks_for_document(1, record, config, FakeTokenCounter(), warnings_out=warnings)
    assert errors == []
    # Each chunk must contain whole pairs (no ':' split)
    for c in chunks:
        assert ":" in c.texto
        # no chunk starts or ends with a ':'
        assert not c.texto.strip().startswith(":")
        assert not c.texto.strip().endswith(":")
    # reconstruction exact
    reconstructed = "".join(c.texto for c in chunks)
    assert reconstructed == text


def test_pdf_no_fallback_oversize_raises():
    # long sentence without punctuation in pdf should not fallback
    text = "palabra " * 600
    record = {"doc_id": "doc_pdf", "fuente": "f", "formato": "pdf", "fenomeno": 1, "idioma": "es", "texto": text}
    config = ChunkingConfig(input_path=Path("in"), output_path=Path("out"), error_path=Path("err"), tokenizer_model="fake", max_tokens=100, max_words=50, overlap_sentences=0, on_oversize="fail")
    with pytest.raises(FatalChunkingError):
        build_chunks_for_document(1, record, config, FakeTokenCounter())


def test_pathological_single_pair_hard_split():
    # single key:value where value is huge with no inner separators
    value = "x" * 5000
    text = f"geom: {value}"
    record = {"doc_id": "doc_geom", "fuente": "f", "formato": "pbf", "fenomeno": 1, "idioma": "es", "texto": text}
    config = ChunkingConfig(input_path=Path("in"), output_path=Path("out"), error_path=Path("err"), tokenizer_model="fake", max_tokens=1, max_words=1000, overlap_sentences=0, on_oversize="fail")
    warnings: list[dict] = []
    chunks, errors = build_chunks_for_document(1, record, config, FakeTokenCounter(), warnings_out=warnings)
    assert errors == []
    # ensure motivo contains 'hard_split_by_chars' for at least one warning
    assert any("hard_split_by_chars" in w.get("motivo", "") for w in warnings)
    # reconstruction exact
    reconstructed = "".join(c.texto for c in chunks)
    assert reconstructed == text


def test_token_counter_influence_on_splitting():
    # using WeirdTokenCounter where tokens are based on char count
    words = ["palabra"] * 200
    text = " ".join(words)
    record = {"doc_id": "doc_weird", "fuente": "f", "formato": "csv", "fenomeno": 1, "idioma": "es", "texto": text}
    config = ChunkingConfig(input_path=Path("in"), output_path=Path("out"), error_path=Path("err"), tokenizer_model="fake", max_tokens=30, max_words=250, overlap_sentences=0, on_oversize="fail")
    warnings: list[dict] = []
    chunks, errors = build_chunks_for_document(1, record, config, WeirdTokenCounter(), warnings_out=warnings)
    assert errors == []
    assert warnings
    # verify each chunk's num_tokens recalculated by counter fits limit
    for c in chunks:
        assert c.num_tokens <= config.max_tokens
    reconstructed = "".join(c.texto for c in chunks)
    assert reconstructed == text


def test_overlap_with_fallback_chunks():
    # create several pairs to produce multiple chunks and test overlap
    pairs = [f"k{i}: v{i}" for i in range(8)]
    text = " ".join(pairs)
    record = {"doc_id": "doc_overlap", "fuente": "f", "formato": "csv", "fenomeno": 1, "idioma": "es", "texto": text}
    config = ChunkingConfig(input_path=Path("in"), output_path=Path("out"), error_path=Path("err"), tokenizer_model="fake", max_tokens=20, max_words=3, overlap_sentences=1, on_oversize="fail")
    warnings: list[dict] = []
    chunks, errors = build_chunks_for_document(1, record, config, FakeTokenCounter(), warnings_out=warnings)
    assert errors == []
    assert warnings
    # ensure overlap: last word(s) of chunk i should appear at start of chunk i+1
    for i in range(len(chunks) - 1):
        # char-level overlap: the end index of chunk i should be greater or equal than the start of chunk i+1
        assert chunks[i].char_end >= chunks[i + 1].char_start


def test_pdf_skip_document_writes_error_file(tmp_path):
    # Long no-punct sentence in pdf should not be split; skip-document writes an error
    text = "palabra " * 600
    record = {"doc_id": "doc_pdf", "fuente": "f", "formato": "pdf", "fenomeno": 1, "idioma": "es", "texto": text}
    input_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    err_path = tmp_path / "err.jsonl"
    input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    config = ChunkingConfig(input_path=input_path, output_path=out_path, error_path=err_path, tokenizer_model="fake", max_tokens=100, max_words=50, overlap_sentences=0, on_oversize="skip-document")
    # run processing with FakeTokenCounter via factory
    exit_code = process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())
    # output file should exist (may be empty if skipped), error file must contain entry
    assert err_path.exists()
    contents = err_path.read_text(encoding="utf-8")
    assert "oración oversize" in contents or "oraci" in contents


def test_advertencias_written_for_fallback(tmp_path):
    # Create a CSV-like document that triggers structured fallback and ensure advertencias.jsonl is written
    text = "ciudad: Bogotá D.C. poblacion: 7000000 area_km2: 1587"
    record = {"doc_id": "doc_csv", "fuente": "f", "formato": "csv", "fenomeno": 1, "idioma": "es", "texto": text}
    input_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    err_path = tmp_path / "err.jsonl"
    input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    config = ChunkingConfig(input_path=input_path, output_path=out_path, error_path=err_path, tokenizer_model="fake", max_tokens=10, max_words=5, overlap_sentences=0, on_oversize="fail")
    process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())
    advert_path = out_path.parent / "advertencias.jsonl"
    assert advert_path.exists()
    lines = advert_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1


def test_transformer_token_counter_respects_limits():
    transformers = pytest.importorskip("transformers")
    # try to create a small tokenizer; skip the test if download fails
    try:
        from codefest_ad_astra.ingest.chunking import create_transformer_token_counter
        tc = create_transformer_token_counter("distilbert-base-uncased")
    except Exception:
        pytest.skip("Modelo de tokenizer no disponible localmente; este test verifica el conteo de tokens con un tokenizer real y no debe considerarse opcional — ejecutar al menos una vez con el modelo descargado antes de dar por cerrada la Fase 3.")
    # build a structured text and ensure token counts after split respect limits
    text = "k0: " + ("word " * 200)
    record = {"doc_id": "doc_tf", "fuente": "f", "formato": "csv", "fenomeno": 1, "idioma": "es", "texto": text}
    config = ChunkingConfig(input_path=Path("in"), output_path=Path("out"), error_path=Path("err"), tokenizer_model="distilbert-base-uncased", max_tokens=50, max_words=1000, overlap_sentences=0, on_oversize="fail")
    chunks, errors = build_chunks_for_document(1, record, config, tc)
    assert errors == []
    for c in chunks:
        assert c.num_tokens <= config.max_tokens


def test_separator_in_value_generates_advertencia(tmp_path):
    # value contains the separator ' | ' itself; should generate advertencia motivo
    text = "a: value with | inside | b: v"
    record = {"doc_id": "doc_sep_val", "fuente": "f", "formato": "csv", "fenomeno": 1, "idioma": "es", "texto": text}
    input_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    err_path = tmp_path / "err.jsonl"
    input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    config = ChunkingConfig(input_path=input_path, output_path=out_path, error_path=err_path, tokenizer_model="fake", max_tokens=5, max_words=100, overlap_sentences=0, on_oversize="fail")
    process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())
    advert_path = out_path.parent / "advertencias.jsonl"
    assert advert_path.exists()
    data = [json.loads(l) for l in advert_path.read_text(encoding="utf-8").strip().splitlines()]
    assert any(w.get("motivo") == "posible_separador_en_valor" for w in data)