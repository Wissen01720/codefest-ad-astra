"""Modelo de documento usado en las Fases 1 y 2 del pipeline (extracción + limpieza)."""
from dataclasses import dataclass, asdict
import json


@dataclass
class Document:
    doc_id: str
    fuente: str      # ruta relativa del archivo original dentro del corpus
    formato: str      # pdf, html, json, csv, xlsx, imagen
    fenomeno: int      # 1, 2 o 3
    idioma: str        # es, en, pt... (detectado automáticamente)
    texto: str          # texto limpio y completo del documento (aún sin fragmentar)
    
    def to_json_line(self) -> str:
        """Convert the Document instance to a JSON string in a single line."""
        return json.dumps(asdict(self), ensure_ascii=False)
    