from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from codefest_ad_astra.ingest.chunking import (
    ChunkingConfig,
    FakeTokenCounter,
    build_chunks_for_document,
    create_transformer_token_counter,
    parse_args,
    process_chunking,
    main,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def test_cli_help_returns_zero() -> None:
    result = os.system(
        ".venv/bin/python -m codefest_ad_astra.ingest.chunking --help > /dev/null 2>&1"
    )
    assert result == 0


def test_missing_arguments_return_nonzero() -> None:
    result = os.system(
        ".venv/bin/python -m codefest_ad_astra.ingest.chunking --entrada input.jsonl > /dev/null 2>&1"
    )
    assert result != 0


def test_cli_invalid_numeric_config_returns_one_and_no_output(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "corpus.jsonl"
    input_path.write_text(
        json.dumps([
            {
                "doc_id": "cli_bad",
                "fuente": "fuente",
                "formato": "txt",
                "fenomeno": 1,
                "idioma": "es",
                "texto": "Texto corto.",
            }
        ][0], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"

    monkeypatch.setattr(
        "codefest_ad_astra.ingest.chunking.create_transformer_token_counter",
        lambda _: FakeTokenCounter(),
    )

    result = main([
        "--entrada",
        str(input_path),
        "--salida",
        str(output_path),
        "--errores",
        str(error_path),
        "--tokenizer-model",
        "dummy",
        "--max-tokens",
        "0",
    ])

    assert result == 1
    assert not output_path.exists()
    assert not error_path.exists()


def test_valid_execution_returns_zero_and_writes_files(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"
    records = [
        {
            "doc_id": "cli1",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Primera oración. Segunda oración.",
        }
    ]
    write_jsonl(input_path, records)

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
    assert process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter()) == 0
    assert output_path.exists()
    assert error_path.exists()
    assert output_path.read_text(encoding="utf-8").strip()
    assert error_path.read_text(encoding="utf-8").strip() == ""


def test_json_invalid_returns_one(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text("{ invalid json }\n", encoding="utf-8")
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"

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

    with pytest.raises(Exception):
        process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())


def test_duplicate_doc_id_returns_exception(tmp_path: Path) -> None:
    input_path = tmp_path / "duplicate.jsonl"
    write_jsonl(input_path, [
        {
            "doc_id": "cli_dup",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Primera oración.",
        },
        {
            "doc_id": "cli_dup",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Segunda oración.",
        },
    ])
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"

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

    with pytest.raises(Exception, match="doc_id duplicado"):
        process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())


def test_tokenizer_load_error_returns_one(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    write_jsonl(input_path, [
        {
            "doc_id": "cli2",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Una oración.",
        }
    ])
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"
    config = ChunkingConfig(
        input_path=input_path,
        output_path=output_path,
        error_path=error_path,
        tokenizer_model="missing-model-abc-123",
        max_tokens=10,
        max_words=10,
        overlap_sentences=1,
        on_oversize="skip-document",
    )

    with pytest.raises(Exception) as excinfo:
        process_chunking(config, token_counter_factory=lambda name: (_ for _ in ()).throw(RuntimeError(f"No se pudo cargar tokenizer '{name}'")))
    assert "No se pudo cargar tokenizer" in str(excinfo.value)


def test_create_transformer_token_counter_uses_offline_flag(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        def __call__(self, text, add_special_tokens=True, truncation=False):
            return {"input_ids": [1, 2, 3]}

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda model_name, **kwargs: captured.update({"model_name": model_name, "kwargs": kwargs}) or DummyTokenizer(),
    )

    token_counter = create_transformer_token_counter("dummy-model")

    assert token_counter.count("hola mundo") == 3
    assert captured["model_name"] == "dummy-model"
    assert captured["kwargs"]["use_fast"] is True
    assert captured["kwargs"]["local_files_only"] is True


def test_on_oversize_fail_returns_one(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    write_jsonl(input_path, [
        {
            "doc_id": "cli3",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Oración oversize que excede los límites.",
        }
    ])
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"

    config = ChunkingConfig(
        input_path=input_path,
        output_path=output_path,
        error_path=error_path,
        tokenizer_model="dummy",
        max_tokens=1,
        max_words=1,
        overlap_sentences=1,
        on_oversize="fail",
    )

    with pytest.raises(Exception):
        process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())
    assert not output_path.exists()
    assert not error_path.exists()


def test_on_oversize_skip_document_writes_error(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    write_jsonl(input_path, [
        {
            "doc_id": "cli4",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Oración oversize que excede los límites.",
        }
    ])
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"

    config = ChunkingConfig(
        input_path=input_path,
        output_path=output_path,
        error_path=error_path,
        tokenizer_model="dummy",
        max_tokens=1,
        max_words=1,
        overlap_sentences=1,
        on_oversize="skip-document",
    )

    assert process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter()) == 0
    assert output_path.exists()
    assert error_path.exists()
    assert output_path.read_text(encoding="utf-8").strip() == ""
    assert "oración oversize" in error_path.read_text(encoding="utf-8")


def test_invalid_max_tokens_returns_exception(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    write_jsonl(input_path, [
        {
            "doc_id": "cli5",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Una oración.",
        }
    ])
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"

    config = ChunkingConfig(
        input_path=input_path,
        output_path=output_path,
        error_path=error_path,
        tokenizer_model="dummy",
        max_tokens=0,
        max_words=10,
        overlap_sentences=1,
        on_oversize="fail",
    )

    with pytest.raises(Exception):
        process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())


def test_invalid_max_words_returns_exception(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    write_jsonl(input_path, [
        {
            "doc_id": "cli6",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Una oración.",
        }
    ])
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"

    config = ChunkingConfig(
        input_path=input_path,
        output_path=output_path,
        error_path=error_path,
        tokenizer_model="dummy",
        max_tokens=10,
        max_words=0,
        overlap_sentences=1,
        on_oversize="fail",
    )

    with pytest.raises(Exception):
        process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())


def test_negative_overlap_sentences_returns_exception(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    write_jsonl(input_path, [
        {
            "doc_id": "cli7",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Una oración.",
        }
    ])
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"

    config = ChunkingConfig(
        input_path=input_path,
        output_path=output_path,
        error_path=error_path,
        tokenizer_model="dummy",
        max_tokens=10,
        max_words=10,
        overlap_sentences=-1,
        on_oversize="fail",
    )

    with pytest.raises(Exception):
        process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())


def test_input_file_missing_returns_exception(tmp_path: Path) -> None:
    input_path = tmp_path / "missing.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"

    config = ChunkingConfig(
        input_path=input_path,
        output_path=output_path,
        error_path=error_path,
        tokenizer_model="dummy",
        max_tokens=10,
        max_words=10,
        overlap_sentences=1,
        on_oversize="fail",
    )

    with pytest.raises(Exception):
        process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())


def test_output_directories_are_created(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    nested = tmp_path / "nested" / "out"
    output_path = nested / "chunks.jsonl"
    error_path = nested / "errors.jsonl"
    write_jsonl(input_path, [
        {
            "doc_id": "cli8",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Una oración. Otra oración.",
        }
    ])

    config = ChunkingConfig(
        input_path=input_path,
        output_path=output_path,
        error_path=error_path,
        tokenizer_model="dummy",
        max_tokens=10,
        max_words=10,
        overlap_sentences=1,
        on_oversize="fail",
    )

    assert process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter()) == 0
    assert output_path.exists()
    assert error_path.exists()


def test_previous_valid_output_preserved_on_failure(tmp_path: Path) -> None:
    output_path = tmp_path / "chunks.jsonl"
    error_path = tmp_path / "errors.jsonl"
    output_path.write_text('{"existing": true}\n', encoding="utf-8")
    error_path.write_text('{"existing": true}\n', encoding="utf-8")

    input_path = tmp_path / "corpus.jsonl"
    write_jsonl(input_path, [
        {
            "doc_id": "cli9",
            "fuente": "fuente",
            "formato": "txt",
            "fenomeno": 1,
            "idioma": "es",
            "texto": "Oración oversize que excede los límites.",
        }
    ])

    config = ChunkingConfig(
        input_path=input_path,
        output_path=output_path,
        error_path=error_path,
        tokenizer_model="dummy",
        max_tokens=1,
        max_words=1,
        overlap_sentences=1,
        on_oversize="fail",
    )

    with pytest.raises(Exception):
        process_chunking(config, token_counter_factory=lambda _: FakeTokenCounter())

    assert output_path.read_text(encoding="utf-8") == '{"existing": true}\n'
    assert error_path.read_text(encoding="utf-8") == '{"existing": true}\n'
