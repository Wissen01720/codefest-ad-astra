"""Fase 4 — Embeddings e índice FAISS.

Convierte el `chunks.jsonl` de la Fase 3 en la base vectorial entregable:
`index.faiss` + `metadata.jsonl` alineados línea a línea con los identificadores
internos de FAISS.
"""

from .encoders import (
    Encoder,
    FakeEncoder,
    SentenceTransformerEncoder,
    crear_encoder,
    prefijos_para_modelo,
    slug_modelo,
)
from .faiss_store import (
    IndexAlignmentError,
    construir_indice,
    cargar_base_vectorial,
    guardar_base_vectorial,
    validar_alineacion,
)

__all__ = [
    "Encoder",
    "FakeEncoder",
    "SentenceTransformerEncoder",
    "crear_encoder",
    "prefijos_para_modelo",
    "slug_modelo",
    "IndexAlignmentError",
    "construir_indice",
    "cargar_base_vectorial",
    "guardar_base_vectorial",
    "validar_alineacion",
]
