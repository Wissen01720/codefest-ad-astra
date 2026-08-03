"""Modelo de fragmento (chunk) — metadata obligatoria de la Tabla 1 (spec Sección 3.4)."""
from dataclasses import dataclass, asdict
import json


@dataclass(slots=True)
class Fragment:
    doc_id: str
    chunk_id: str
    fuente: str
    formato: str
    fenomeno: int
    posicion: int
    num_tokens: int
    texto: str
    idioma: str = ""

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
