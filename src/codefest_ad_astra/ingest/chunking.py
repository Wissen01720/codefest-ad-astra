from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

DEFAULT_MAX_TOKENS = 450
DEFAULT_MAX_WORDS = 250
DEFAULT_OVERLAP_SENTENCES = 1
DEFAULT_ON_OVERSIZE = "fail"
VALID_ON_OVERSIZE = {"fail", "skip-document"}
REQUIRED_FIELDS = ["doc_id", "fuente", "formato", "fenomeno", "idioma", "texto"]

ABBREVIATIONS = {
    "sr",
    "sra",
    "dr",
    "dra",
    "lic",
    "ing",
    "prof",
    "st",
    "srta",
    "srs",
    "sres",
    "etc",
    "p.ej",
    "eg",
    "i.e",
    "e.g",
    "vs",
    "a.m",
    "p.m",
    "u.s",
    "e.u",
    "s.a",
    "s.l",
    "c.v",
    "mr",
    "mrs",
    "ms",
}

LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?P<punct>[\.\?\!…]+[\"')\]]*)\s*")


class ChunkingError(Exception):
    pass


class FatalChunkingError(ChunkingError):
    pass


class TokenCounter:
    def count(self, text: str) -> int:
        raise NotImplementedError


class FakeTokenCounter(TokenCounter):
    def count(self, text: str) -> int:
        return count_words(text)


class TransformerTokenCounter(TokenCounter):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        encoded = self._tokenizer(text, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])


def create_transformer_token_counter(model_name: str) -> TokenCounter:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return TransformerTokenCounter(tokenizer)


@dataclass(slots=True)
class ChunkingConfig:
    input_path: Path
    output_path: Path
    error_path: Path
    tokenizer_model: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_words: int = DEFAULT_MAX_WORDS
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES
    on_oversize: str = DEFAULT_ON_OVERSIZE


@dataclass(slots=True)
class TextBlock:
    text: str
    start: int
    end: int
    is_heading_block: bool


@dataclass(slots=True)
class Sentence:
    text: str
    char_start: int
    char_end: int
    block_index: int
    is_heading: bool


@dataclass(slots=True)
class ChunkRecord:
    doc_id: str
    chunk_id: str
    fuente: str
    formato: str
    fenomeno: int
    idioma: str
    posicion: int
    num_tokens: int
    num_palabras: int
    char_start: int
    char_end: int
    texto: str

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "fuente": self.fuente,
            "formato": self.formato,
            "fenomeno": self.fenomeno,
            "idioma": self.idioma,
            "posicion": self.posicion,
            "num_tokens": self.num_tokens,
            "num_palabras": self.num_palabras,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "texto": self.texto,
        }


@dataclass(slots=True)
class ErrorRecord:
    line_number: int
    doc_id: Optional[str]
    fuente: Optional[str]
    idioma: Optional[str]
    char_start: Optional[int]
    char_end: Optional[int]
    num_tokens: Optional[int]
    num_palabras: Optional[int]
    max_tokens: Optional[int]
    max_words: Optional[int]
    motivo: str

    def to_dict(self) -> dict:
        return {
            "line_number": self.line_number,
            "doc_id": self.doc_id,
            "fuente": self.fuente,
            "idioma": self.idioma,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "num_tokens": self.num_tokens,
            "num_palabras": self.num_palabras,
            "max_tokens": self.max_tokens,
            "max_words": self.max_words,
            "motivo": self.motivo,
        }


def count_words(text: str) -> int:
    return len(text.split())


def _is_blank_line(line: str) -> bool:
    return not line.strip()


def _normalize_line_text(line: str) -> str:
    return line.strip()


def _is_list_marker(line: str) -> bool:
    return bool(LIST_MARKER_RE.match(line))


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _is_list_marker(stripped):
        return False
    if stripped.endswith((".", "?", "!", "…")):
        return False
    if len(stripped.split()) > 10:
        return False
    if len(stripped) > 120:
        return False
    return True


def _get_previous_token(text: str, index: int) -> str:
    prefix = text[:index].rstrip()
    if not prefix:
        return ""
    match = re.search(r"([\w\.]+)$", prefix)
    return match.group(1) if match else ""


def _get_next_char(text: str, index: int) -> Optional[str]:
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        return char
    return None


def _is_decimal_boundary(text: str, dot_index: int) -> bool:
    prev_token = _get_previous_token(text, dot_index)
    if not prev_token.isdigit():
        return False

    suffix = text[dot_index:]
    return bool(re.match(r"^[\.,]\d+(?:[\.,]\d+)*", suffix))


def _is_abbreviation_token(token: str) -> bool:
    token = token.rstrip(".").lower()
    return token in ABBREVIATIONS


def _has_initials(token: str) -> bool:
    return bool(re.fullmatch(r"(?:[A-Za-zÁÉÍÓÚáéíóúÑñ]\.){2,}", token))


def _split_sentences_in_text(text: str, block_index: int, block_start: int) -> list[Sentence]:
    sentences: list[Sentence] = []
    start = 0

    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        candidate_end = match.end()
        punct = match.group("punct")
        prev_token = _get_previous_token(text, match.start())
        next_char = _get_next_char(text, candidate_end)

        if "?" in punct or "!" in punct or "…" in punct:
            boundary = True
        elif "." in punct:
            if _is_decimal_boundary(text, match.start()):
                boundary = False
            elif _is_abbreviation_token(prev_token):
                boundary = False
            elif _has_initials(prev_token):
                boundary = False
            elif len(prev_token) == 1 and prev_token.isalpha() and next_char is not None and next_char.isupper():
                boundary = False
            elif next_char is not None and next_char.islower():
                boundary = False
            else:
                boundary = True
        else:
            boundary = True

        if not boundary:
            continue

        sentence_text = text[start:candidate_end]
        sentences.append(
            Sentence(
                text=sentence_text,
                char_start=block_start + start,
                char_end=block_start + candidate_end,
                block_index=block_index,
                is_heading=False,
            )
        )
        start = candidate_end

    if start < len(text):
        residue = text[start:]
        if residue.strip():
            sentences.append(
                Sentence(
                    text=residue,
                    char_start=block_start + start,
                    char_end=block_start + len(text),
                    block_index=block_index,
                    is_heading=False,
                )
            )
    return sentences


def split_text_into_blocks(text: str) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    boundaries: list[tuple[int, int]] = []
    start = 0
    for separator in re.finditer(r"\n[ \t]*\n+", text):
        end = separator.start()
        if start < end:
            boundaries.append((start, end))
        start = separator.end()
    if start < len(text):
        boundaries.append((start, len(text)))

    i = 0
    while i < len(boundaries):
        block_start, block_end = boundaries[i]
        block_text = text[block_start:block_end]
        block_lines = block_text.splitlines()
        is_heading_block = False
        merge_next = False

        if len(block_lines) == 1 and _is_heading_line(block_lines[0]):
            merge_next = True

        if len(block_lines) > 1 and _is_heading_line(block_lines[0]):
            is_heading_block = True

        if merge_next and i + 1 < len(boundaries):
            next_start, next_end = boundaries[i + 1]
            merged_text = text[block_start:next_end]
            blocks.append(TextBlock(text=merged_text, start=block_start, end=next_end, is_heading_block=True))
            i += 2
            continue

        blocks.append(TextBlock(text=block_text, start=block_start, end=block_end, is_heading_block=is_heading_block))
        i += 1

    return blocks


def split_block_into_sentences(block: TextBlock) -> list[Sentence]:
    if block.is_heading_block:
        lines = block.text.splitlines(True)
        if len(lines) > 1:
            first_line = lines[0]
            heading_end = len(first_line)
            heading_sentence = Sentence(
                text=first_line,
                char_start=block.start,
                char_end=block.start + heading_end,
                block_index=0,
                is_heading=True,
            )
            remainder = block.text[heading_end:]
            remainder_sentences = _split_sentences_in_text(remainder, 0, block.start + heading_end)
            return [heading_sentence] + remainder_sentences

    return _split_sentences_in_text(block.text, 0, block.start)


def _validate_input_document(record: dict, line_number: int) -> dict:
    if not isinstance(record, dict):
        raise FatalChunkingError(f"L{line_number}: input debe ser un objeto JSON.")

    for field in REQUIRED_FIELDS:
        if field not in record:
            raise FatalChunkingError(f"L{line_number}: falta campo obligatorio '{field}'.")

    doc_id = record["doc_id"]
    fuente = record["fuente"]
    formato = record["formato"]
    idioma = record["idioma"]
    texto = record["texto"]
    fenomeno = record["fenomeno"]

    if not isinstance(doc_id, str) or not doc_id.strip():
        raise FatalChunkingError(f"L{line_number}: doc_id vacío o no es texto.")
    if not isinstance(fuente, str) or not fuente.strip():
        raise FatalChunkingError(f"L{line_number}: fuente vacía o no es texto.")
    if not isinstance(formato, str) or not formato.strip():
        raise FatalChunkingError(f"L{line_number}: formato vacío o no es texto.")
    if not isinstance(idioma, str) or not idioma.strip():
        raise FatalChunkingError(f"L{line_number}: idioma vacío o no es texto.")
    if not isinstance(texto, str) or not texto.strip():
        raise FatalChunkingError(f"L{line_number}: texto vacío o solo espacios.")
    if not isinstance(fenomeno, int) or fenomeno not in {1, 2, 3}:
        raise FatalChunkingError(f"L{line_number}: fenomeno debe ser 1, 2 o 3.")

    return record


def _validate_config(config: ChunkingConfig) -> None:
    if config.max_tokens <= 0:
        raise FatalChunkingError("max_tokens debe ser mayor que 0.")
    if config.max_words <= 0:
        raise FatalChunkingError("max_words debe ser mayor que 0.")
    if config.overlap_sentences < 0:
        raise FatalChunkingError("overlap_sentences debe ser 0 o mayor.")
    if config.on_oversize not in VALID_ON_OVERSIZE:
        raise FatalChunkingError(f"Valor desconocido para --on-oversize: {config.on_oversize}")


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise FatalChunkingError(f"L{line_number}: JSON inválido: {exc}")
            yield line_number, record, stripped


def _chunk_id_for(doc_id: str, position: int) -> str:
    return f"{doc_id}-chunk-{position:04d}"


def _is_heading_span(sentence: Sentence) -> bool:
    return sentence.is_heading


def _select_overlap(prev_sentences: Sequence[Sentence], desired: int) -> int:
    if desired <= 0 or len(prev_sentences) <= 1:
        return 0
    for overlap in range(min(desired, len(prev_sentences) - 1), 0, -1):
        overlap_sentences = prev_sentences[-overlap:]
        if len(overlap_sentences) == 1 and overlap_sentences[0].is_heading:
            continue
        if all(s.is_heading for s in overlap_sentences):
            continue
        return overlap
    return 0


def _is_block_end(sentences: Sequence[Sentence], index: int) -> bool:
    return index + 1 == len(sentences) or sentences[index + 1].block_index != sentences[index].block_index


def _choose_chunk_indices(
    sentences: Sequence[Sentence],
    start_index: int,
    original_text: str,
    config: ChunkingConfig,
    token_counter: TokenCounter,
) -> list[int]:
    if start_index >= len(sentences):
        return []

    chunk_indices: list[int] = []
    chunk_start = sentences[start_index].char_start
    last_preferred_break: Optional[int] = None

    broken_by_limit = False
    for index in range(start_index, len(sentences)):
        sentence = sentences[index]
        if index > start_index and sentence.is_heading:
            break

        candidate_text = original_text[chunk_start:sentence.char_end]
        candidate_tokens = token_counter.count(candidate_text)
        candidate_words = count_words(candidate_text)

        if candidate_tokens > config.max_tokens or candidate_words > config.max_words:
            broken_by_limit = True
            break

        chunk_indices.append(index)

        if _is_block_end(sentences, index):
            last_preferred_break = index

    if not chunk_indices:
        return []

    if broken_by_limit and last_preferred_break is not None and last_preferred_break < chunk_indices[-1]:
        if last_preferred_break >= start_index:
            chunk_indices = [i for i in chunk_indices if i <= last_preferred_break]

    return chunk_indices


def _is_oversize_sentence(sentence: Sentence, original_text: str, config: ChunkingConfig, token_counter: TokenCounter) -> bool:
    text = original_text[sentence.char_start:sentence.char_end]
    return token_counter.count(text) > config.max_tokens or count_words(text) > config.max_words


def _validate_chunk_sequence(chunks: Sequence[ChunkRecord], original_text: str, config: ChunkingConfig) -> None:
    if not chunks:
        return

    seen_ids: set[str] = set()
    expected_position = 0
    prev_end = 0
    covered: list[tuple[int, int]] = []

    for chunk in chunks:
        if not chunk.texto:
            raise FatalChunkingError(f"Chunk vacío en {chunk.chunk_id}.")
        if chunk.chunk_id in seen_ids:
            raise FatalChunkingError(f"Chunk_id duplicado: {chunk.chunk_id}.")
        seen_ids.add(chunk.chunk_id)
        if chunk.posicion != expected_position:
            raise FatalChunkingError(f"Posición inválida para {chunk.chunk_id}: {chunk.posicion} espera {expected_position}.")
        expected_position += 1
        if chunk.num_tokens > config.max_tokens:
            raise FatalChunkingError(f"Chunk {chunk.chunk_id} supera max_tokens.")
        if chunk.num_palabras > config.max_words:
            raise FatalChunkingError(f"Chunk {chunk.chunk_id} supera max_words.")
        if chunk.texto != original_text[chunk.char_start:chunk.char_end]:
            raise FatalChunkingError(f"Texto del chunk {chunk.chunk_id} no coincide con slice original.")
        if chunk.char_start < 0 or chunk.char_end > len(original_text) or chunk.char_start >= chunk.char_end:
            raise FatalChunkingError(f"Offsets inválidos en chunk {chunk.chunk_id}.")
        covered.append((chunk.char_start, chunk.char_end))

    covered.sort()
    merged: list[tuple[int, int]] = []
    for start, end in covered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    for start, end in merged:
        for pos in range(start, end):
            if original_text[pos].strip():
                break
        else:
            continue

    # check coverage of non-whitespace characters
    current = 0
    for start, end in merged:
        while current < start:
            if original_text[current].strip():
                raise FatalChunkingError("Contenido no vacío no cubierto por los chunks.")
            current += 1
        current = max(current, end)
    while current < len(original_text):
        if original_text[current].strip():
            raise FatalChunkingError("Contenido no vacío no cubierto por los chunks.")
        current += 1


def build_chunks_for_document(
    line_number: int,
    record: dict,
    config: ChunkingConfig,
    token_counter: TokenCounter,
) -> tuple[list[ChunkRecord], list[ErrorRecord]]:
    validated = _validate_input_document(record, line_number)
    doc_id = validated["doc_id"]
    texto = validated["texto"]
    blocks = split_text_into_blocks(texto)
    sentences: list[Sentence] = []
    for block_index, block in enumerate(blocks):
        block_sentences = split_block_into_sentences(block)
        for sentence in block_sentences:
            sentence.block_index = block_index
        sentences.extend(block_sentences)

    if not sentences:
        return [], []

    chunks: list[ChunkRecord] = []
    errors: list[ErrorRecord] = []
    prev_chunk_sentences: list[Sentence] = []
    cursor = 0
    position = 0

    while cursor < len(sentences):
        overlap = _select_overlap(prev_chunk_sentences, config.overlap_sentences)
        start_index = cursor - overlap if overlap > 0 else cursor
        if start_index < 0:
            start_index = 0

        chunk_indices = _choose_chunk_indices(sentences, start_index, texto, config, token_counter)
        if not chunk_indices or chunk_indices[-1] < cursor:
            if overlap > 0:
                overlap = _select_overlap(prev_chunk_sentences, overlap - 1)
                start_index = cursor - overlap if overlap > 0 else cursor
                chunk_indices = _choose_chunk_indices(sentences, start_index, texto, config, token_counter)

        if not chunk_indices or chunk_indices[-1] < cursor:
            sentence = sentences[cursor]
            if _is_oversize_sentence(sentence, texto, config, token_counter):
                fragment_text = texto[sentence.char_start:sentence.char_end]
                fragment_tokens = token_counter.count(fragment_text)
                fragment_words = count_words(fragment_text)
                error = ErrorRecord(
                    line_number=line_number,
                    doc_id=doc_id,
                    fuente=record["fuente"],
                    idioma=record["idioma"],
                    char_start=sentence.char_start,
                    char_end=sentence.char_end,
                    num_tokens=fragment_tokens,
                    num_palabras=fragment_words,
                    max_tokens=config.max_tokens,
                    max_words=config.max_words,
                    motivo="oración oversize",
                )
                if config.on_oversize == "fail":
                    raise FatalChunkingError(
                        f"L{line_number}: oración oversize en documento {doc_id}."
                    )
                errors.append(error)
                return [], errors
            raise FatalChunkingError(
                f"L{line_number}: no se pudo construir chunk válido para documento {doc_id}."
            )

        if chunk_indices[0] < start_index:
            chunk_indices = [idx for idx in chunk_indices if idx >= start_index]

        if not chunk_indices:
            raise FatalChunkingError(
                f"L{line_number}: no se pudo construir chunk válido para documento {doc_id}."
            )

        chunk_start = sentences[chunk_indices[0]].char_start
        chunk_end = sentences[chunk_indices[-1]].char_end
        chunk_text = texto[chunk_start:chunk_end]
        num_tokens = token_counter.count(chunk_text)
        num_palabras = count_words(chunk_text)

        if num_tokens > config.max_tokens or num_palabras > config.max_words:
            raise FatalChunkingError(
                f"L{line_number}: chunk generado excede límites para documento {doc_id}."
            )

        chunk_id = _chunk_id_for(doc_id, position)
        chunk = ChunkRecord(
            doc_id=doc_id,
            chunk_id=chunk_id,
            fuente=record["fuente"],
            formato=record["formato"],
            fenomeno=record["fenomeno"],
            idioma=record["idioma"],
            posicion=position,
            num_tokens=num_tokens,
            num_palabras=num_palabras,
            char_start=chunk_start,
            char_end=chunk_end,
            texto=chunk_text,
        )
        chunks.append(chunk)
        prev_chunk_sentences = [sentences[idx] for idx in chunk_indices]
        cursor = chunk_indices[-1] + 1
        position += 1

    _validate_chunk_sequence(chunks, texto, config)
    return chunks, errors


def process_chunking(
    config: ChunkingConfig,
    token_counter_factory: Callable[[str], TokenCounter] = create_transformer_token_counter,
) -> int:
    if config.on_oversize not in VALID_ON_OVERSIZE:
        raise FatalChunkingError(f"Valor desconocido para --on-oversize: {config.on_oversize}")

    _validate_config(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.error_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        token_counter = token_counter_factory(config.tokenizer_model)
    except Exception as exc:
        raise FatalChunkingError(f"No se pudo cargar tokenizer '{config.tokenizer_model}': {exc}") from exc

    output_tmp = None
    error_tmp = None
    try:
        output_tmp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(config.output_path.parent),
            prefix=config.output_path.name + ".",
            suffix=".tmp",
        )
        error_tmp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(config.error_path.parent),
            prefix=config.error_path.name + ".",
            suffix=".tmp",
        )
        output_tmp = Path(output_tmp_file.name)
        error_tmp = Path(error_tmp_file.name)

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.error_path.parent.mkdir(parents=True, exist_ok=True)

        line_number = 0
        doc_ids: set[str] = set()
        any_chunks_written = False
        any_errors_written = False

        for line_number, record, _ in _read_jsonl(config.input_path):
            validated = _validate_input_document(record, line_number)
            doc_id = validated["doc_id"]
            if doc_id in doc_ids:
                raise FatalChunkingError(f"L{line_number}: doc_id duplicado '{doc_id}'.")
            doc_ids.add(doc_id)
            chunks, errors = build_chunks_for_document(line_number, validated, config, token_counter)
            for error in errors:
                error_tmp_file.write(json.dumps(error.to_dict(), ensure_ascii=False) + "\n")
                any_errors_written = True
            for chunk in chunks:
                output_tmp_file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                any_chunks_written = True

        output_tmp_file.close()
        error_tmp_file.close()

        if line_number == 0:
            raise FatalChunkingError("El archivo de entrada está vacío o no contiene registros JSON válidos.")

        os.replace(str(output_tmp), str(config.output_path))
        os.replace(str(error_tmp), str(config.error_path))
        return 0
    except Exception:
        if output_tmp is not None and output_tmp.exists():
            try:
                output_tmp.unlink()
            except OSError:
                pass
        if error_tmp is not None and error_tmp.exists():
            try:
                error_tmp.unlink()
            except OSError:
                pass
        raise


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fase 3: chunking de documentos JSONL ya procesados.")
    parser.add_argument("--entrada", type=Path, required=True, help="Archivo JSONL de entrada con documentos procesados.")
    parser.add_argument("--salida", type=Path, required=True, help="Archivo JSONL de salida con chunks.")
    parser.add_argument("--errores", type=Path, required=True, help="Archivo JSONL de errores.")
    parser.add_argument("--tokenizer-model", type=str, required=True, help="Nombre del tokenizer para conteo de tokens.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Máximo de tokens por chunk.")
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS, help="Máximo de palabras por chunk.")
    parser.add_argument("--overlap-sentences", type=int, default=DEFAULT_OVERLAP_SENTENCES, help="Número de oraciones de solapamiento entre chunks.")
    parser.add_argument("--on-oversize", choices=sorted(VALID_ON_OVERSIZE), default=DEFAULT_ON_OVERSIZE, help="Política para oraciones oversize.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = ChunkingConfig(
        input_path=args.entrada,
        output_path=args.salida,
        error_path=args.errores,
        tokenizer_model=args.tokenizer_model,
        max_tokens=args.max_tokens,
        max_words=args.max_words,
        overlap_sentences=args.overlap_sentences,
        on_oversize=args.on_oversize,
    )
    try:
        return process_chunking(config, token_counter_factory=create_transformer_token_counter)
    except FatalChunkingError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
