"""Wrapper delgado sobre sentence-transformers para la Fase 4.

Aísla al resto del código de la librería concreta: si Dupla B cambia de
encoder o de librería de embeddings, solo este módulo cambia.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_ENCODER = "BAAI/bge-m3"


def load_encoder(model_name: str = DEFAULT_ENCODER) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_texts(model, textos: list[str], batch_size: int = 32) -> np.ndarray:
    """Codifica `textos` y normaliza cada vector a norma unitaria: requisito
    para que faiss.IndexFlatIP equivalga a similitud coseno (spec 6 y 8.2)."""
    vectores = model.encode(
        textos,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectores, dtype="float32")
