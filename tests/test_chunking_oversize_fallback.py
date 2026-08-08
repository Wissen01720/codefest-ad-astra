from pathlib import Path

from codefest_ad_astra.ingest.chunking import (
    build_chunks_for_document,
    ChunkingConfig,
    FakeTokenCounter,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_WORDS,
)


def test_oversize_sentence_fallback_splits_by_words() -> None:
    # Create a single document whose "texto" is a very long row without
    # terminal punctuation (simulating a CSV/XLSX/PBF "column: value" line).
    # Using many words so FakeTokenCounter (word-based) will treat it as
    # oversized and trigger the whitespace-splitting fallback.
    long_word_count = 600
    words = ["palabra"] * long_word_count
    text = " ".join(words)

    record = {
        "doc_id": "doc_long_row",
        "fuente": "fuente",
        "formato": "csv",
        "fenomeno": 1,
        "idioma": "es",
        "texto": text,
    }

    config = ChunkingConfig(
        input_path=Path("in.jsonl"),
        output_path=Path("out.jsonl"),
        error_path=Path("err.jsonl"),
        tokenizer_model="fake",
        max_tokens=DEFAULT_MAX_TOKENS,
        max_words=250,
        overlap_sentences=0,
        on_oversize="fail",
    )

    warnings: list[dict] = []
    chunks, errors = build_chunks_for_document(1, record, config, FakeTokenCounter(), warnings_out=warnings)

    assert errors == []
    assert warnings
    # Expect the long row to be split into multiple chunks (600/250 -> 3 chunks)
    assert len(chunks) >= 3

    # Reconstruct the original text from chunk slices and validate equality
    reconstructed = "".join([c.texto for c in chunks])
    assert reconstructed == text

    # Positions should be sequential and num_palabras should respect the limit
    for i, c in enumerate(chunks):
        assert c.posicion == i
        assert c.num_palabras <= config.max_words
