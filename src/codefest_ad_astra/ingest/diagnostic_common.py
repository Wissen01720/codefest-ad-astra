"""Shared helpers for corpus diagnostics and JSON schema analysis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import sha1, sha256
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse


HTTP_URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


def iter_corpus_files(corpus: Path, suffixes: set[str]) -> list[Path]:
    """Return sorted files in a corpus filtered by suffix."""

    return sorted(
        path
        for path in corpus.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def read_json(path: Path) -> Any:
    """Load a JSON document from disk using a permissive UTF-8 read."""

    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def file_sha256(path: Path) -> str:
    """Compute a stable SHA-256 hash for a file."""

    digest = sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def is_http_url(value: Any) -> bool:
    """Return True when a value looks like an HTTP or HTTPS URL."""

    if not isinstance(value, str):
        return False

    try:
        parsed = urlparse(value.strip())
    except ValueError:
        return False

    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def looks_like_pdf_reference(value: Any) -> bool:
    """Return True when a value looks like a PDF reference or PDF URL."""

    if not isinstance(value, str):
        return False

    text = value.strip().lower()

    if not text:
        return False

    return text.endswith(".pdf") or ".pdf?" in text or ".pdf#" in text


def string_kind(value: str) -> str:
    """Classify a string value for schema statistics."""

    text = value.strip()

    if not text:
        return "empty_string"

    if is_http_url(text):
        return "url"

    if looks_like_pdf_reference(text):
        return "pdf_url"

    if len(text) >= 160:
        return "long_text"

    if len(text) >= 40:
        return "text"

    return "short_text"


def value_kind(value: Any) -> str:
    """Classify a JSON value into a compact kind label."""

    if isinstance(value, str):
        return string_kind(value)

    if isinstance(value, dict):
        return "dict"

    if isinstance(value, list):
        return "list"

    if value is None:
        return "null"

    if isinstance(value, bool):
        return "bool"

    if isinstance(value, int):
        return "int"

    if isinstance(value, float):
        return "float"

    return type(value).__name__


def scalar_text(value: Any) -> str:
    """Return the text payload that should count as textual content."""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return "\n\n".join(item.strip() for item in value if item.strip())

    return ""


def flatten_json(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield flattened JSON paths and values.

    Lists are normalized to a ``[]`` suffix so repeated items collapse into a
    single field path for profiling purposes.
    """

    if isinstance(value, dict):
        if path:
            yield path, value
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from flatten_json(child, child_path)
        return

    if isinstance(value, list):
        if path:
            yield path, value
        for child in value:
            child_path = f"{path}[]" if path else "[]"
            yield from flatten_json(child, child_path)
        return

    yield (path or "$"), value


def stable_schema_signature(value: Any, depth: int = 5, breadth: int = 12) -> str:
    """Build a stable fingerprint for a JSON document structure."""

    def build(node: Any, current_depth: int) -> Any:
        if current_depth <= 0:
            return "..."

        if isinstance(node, dict):
            items = sorted(node.items(), key=lambda item: str(item[0]))[:breadth]
            return {
                "type": "dict",
                "fields": [
                    [str(key), build(child, current_depth - 1)]
                    for key, child in items
                ],
            }

        if isinstance(node, list):
            return {
                "type": "list",
                "length": len(node),
                "items": [build(child, current_depth - 1) for child in node[:breadth]],
            }

        return {"type": value_kind(node)}

    serial = json.dumps(build(value, depth), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha1(serial.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class FieldStatistics:
    """Aggregate statistics for a JSON path."""

    path: str
    documents: int = 0
    occurrences: int = 0
    value_kinds: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    text_characters: int = 0
    min_characters: int | None = None
    max_characters: int = 0
    url_values: int = 0
    pdf_values: int = 0
    sample_values: list[str] = field(default_factory=list)

    def register_document(self) -> None:
        self.documents += 1

    def register_value(self, value: Any) -> None:
        self.occurrences += 1

        kind = value_kind(value)
        self.value_kinds[kind] = self.value_kinds.get(kind, 0) + 1

        text = scalar_text(value)

        if not text:
            return

        characters = len(text)
        self.text_characters += characters

        if self.min_characters is None or characters < self.min_characters:
            self.min_characters = characters

        if characters > self.max_characters:
            self.max_characters = characters

        if is_http_url(text):
            self.url_values += 1

        if looks_like_pdf_reference(text):
            self.pdf_values += 1

        if len(self.sample_values) < 3:
            self.sample_values.append(text[:200])

    @property
    def mean_characters(self) -> float:
        if self.documents == 0:
            return 0.0
        return self.text_characters / self.documents


@dataclass(slots=True)
class JsonDocument:
    """Parsed JSON document metadata used by the analyzers."""

    path: Path
    data: Any
    root_kind: str
    fingerprint: str
    top_level_keys: tuple[str, ...]


def load_json_document(path: Path) -> JsonDocument:
    """Parse a JSON document and attach structural metadata."""

    data = read_json(path)
    root_kind = type(data).__name__
    fingerprint = stable_schema_signature(data)
    top_level_keys = tuple(sorted(data.keys())) if isinstance(data, dict) else ()

    return JsonDocument(
        path=path,
        data=data,
        root_kind=root_kind,
        fingerprint=fingerprint,
        top_level_keys=top_level_keys,
    )