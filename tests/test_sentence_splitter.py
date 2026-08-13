from __future__ import annotations

from codefest_ad_astra.ingest.chunking import split_block_into_sentences, split_text_into_blocks


def test_split_sentences_preserves_abbreviations() -> None:
    texto = "Esto es un Sr. Pérez. Sigue aquí."
    blocks = split_text_into_blocks(texto)
    assert len(blocks) == 1

    sentences = split_block_into_sentences(blocks[0])
    assert [sentence.text.strip() for sentence in sentences] == [
        "Esto es un Sr. Pérez.",
        "Sigue aquí.",
    ]


def test_split_sentences_ignores_decimal_periods() -> None:
    texto = "El precio es 1.234,56. Otra frase."
    blocks = split_text_into_blocks(texto)
    sentences = split_block_into_sentences(blocks[0])

    assert [sentence.text.strip() for sentence in sentences] == [
        "El precio es 1.234,56.",
        "Otra frase.",
    ]


def test_split_sentences_handles_spanish_question_exclamation() -> None:
    texto = "¿Qué tal? ¡Muy bien! Esto sigue."
    blocks = split_text_into_blocks(texto)
    sentences = split_block_into_sentences(blocks[0])

    assert [sentence.text.strip() for sentence in sentences] == [
        "¿Qué tal?",
        "¡Muy bien!",
        "Esto sigue.",
    ]


def test_split_sentences_handles_english_initials_and_us_abbreviation() -> None:
    texto = "Mr. Smith viaja a los U.S. hoy. Esto continúa."
    blocks = split_text_into_blocks(texto)
    sentences = split_block_into_sentences(blocks[0])

    assert [sentence.text.strip() for sentence in sentences] == [
        "Mr. Smith viaja a los U.S. hoy.",
        "Esto continúa.",
    ]


def test_split_sentences_handles_portuguese_dr_and_exclamation() -> None:
    texto = "Dr. Silva disse: Olá! Tudo bem? Sim."
    blocks = split_text_into_blocks(texto)
    sentences = split_block_into_sentences(blocks[0])

    assert [sentence.text.strip() for sentence in sentences] == [
        "Dr. Silva disse: Olá!",
        "Tudo bem?",
        "Sim.",
    ]


def test_split_sentences_handles_ascii_ellipsis() -> None:
    texto = "Esto... ¿Qué sigue? Otra frase."
    blocks = split_text_into_blocks(texto)
    sentences = split_block_into_sentences(blocks[0])

    assert [sentence.text.strip() for sentence in sentences] == [
        "Esto...",
        "¿Qué sigue?",
        "Otra frase.",
    ]


def test_split_sentences_handles_unicode_ellipsis() -> None:
    texto = "Esto… sigue. Otra frase."
    blocks = split_text_into_blocks(texto)
    sentences = split_block_into_sentences(blocks[0])

    assert [sentence.text.strip() for sentence in sentences] == [
        "Esto…",
        "sigue.",
        "Otra frase.",
    ]


def test_split_sentences_handles_numbered_lists_and_bullets() -> None:
    texto = (
        "1. Item uno.\n"
        "2. Item dos.\n"
        "- Viñeta uno.\n"
        "- Viñeta dos.\n\n"
        "Otra frase."
    )
    blocks = split_text_into_blocks(texto)
    assert len(blocks) == 2

    sentences = split_block_into_sentences(blocks[0])
    texts = [sentence.text.strip() for sentence in sentences]
    assert texts[0] == "1."
    assert "Item uno." in texts[1]
    assert texts[2] == "2."
    assert "Item dos." in texts[3]
    assert any(item.startswith("- Viñeta uno") for item in texts)
    assert any(item.startswith("- Viñeta dos") for item in texts)

    sentences = split_block_into_sentences(blocks[1])
    assert [sentence.text.strip() for sentence in sentences] == ["Otra frase."]


def test_split_text_blocks_separate_multiple_headers() -> None:
    texto = "Sección uno\n\nPrimera frase.\n\nSección dos\n\nSegunda frase."
    blocks = split_text_into_blocks(texto)

    assert len(blocks) == 2
    assert blocks[0].is_heading_block
    assert blocks[1].is_heading_block
    assert split_block_into_sentences(blocks[0])[0].text.strip() == "Sección uno"
    assert split_block_into_sentences(blocks[1])[0].text.strip() == "Sección dos"
