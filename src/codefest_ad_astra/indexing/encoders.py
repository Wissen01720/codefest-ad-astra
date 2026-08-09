"""Wrapper de encoders para la Fase 4.

Regla dura del reto (Sección 4.2 y 8.3 de la especificación): solo se permiten
modelos *encoder* (familia BERT). Nada de decoders/LLMs generativos en ninguna
etapa de indexación o recuperación.

El mismo encoder que construye el índice debe codificar las consultas
(Sección 8.1), incluyendo los prefijos de instrucción si el modelo los exige.
Por eso `prefijos_para_modelo` es la única fuente de verdad y su resultado se
persiste en el manifiesto de la base vectorial.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

# Modelo por defecto: multilingüe (ES/EN/PT), licencia MIT, 1024 dimensiones,
# buen desempeño en recuperación densa (MTEB/BEIR) y contexto amplio.
MODELO_POR_DEFECTO = "BAAI/bge-m3"

# Longitud máxima de entrada. Los chunks de la Fase 3 vienen acotados a
# --max-tokens (450 por defecto), así que 512 cubre el caso con margen y evita
# que un modelo de contexto largo (bge-m3 admite 8192) desperdicie cómputo.
MAX_SEQ_LENGTH_POR_DEFECTO = 512

# Algunas familias de encoders fueron entrenadas con prefijos de instrucción y
# pierden calidad de forma notable si se omiten. La clave es un patrón sobre el
# nombre del modelo; el valor es (prefijo_consulta, prefijo_pasaje).
_PREFIJOS: tuple[tuple[re.Pattern[str], tuple[str, str]], ...] = (
    (re.compile(r"(^|/)(multilingual-)?e5", re.IGNORECASE), ("query: ", "passage: ")),
    (re.compile(r"gtr|sentence-t5", re.IGNORECASE), ("", "")),
    (re.compile(r"bge-m3", re.IGNORECASE), ("", "")),
)


def prefijos_para_modelo(nombre_modelo: str) -> tuple[str, str]:
    """Devuelve (prefijo_consulta, prefijo_pasaje) para el modelo indicado.

    Para los modelos que no requieren prefijo devuelve ("", ""). El resultado se
    guarda en el manifiesto para que la recuperación use exactamente lo mismo.
    """
    for patron, prefijos in _PREFIJOS:
        if patron.search(nombre_modelo):
            return prefijos
    return ("", "")


def slug_modelo(nombre_modelo: str) -> str:
    """Nombre de carpeta seguro a partir del identificador de HuggingFace.

    'BAAI/bge-m3' -> 'bge-m3', usado como `base_vectorial/encoder_bge-m3/`.
    """
    base = nombre_modelo.rstrip("/").split("/")[-1]
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", base).strip("-._")
    if not slug:
        raise ValueError(f"No se pudo derivar un nombre de carpeta desde {nombre_modelo!r}")
    return slug


def normalizar(vectores: np.ndarray) -> np.ndarray:
    """Normaliza a norma unitaria en float32.

    Con vectores unitarios, el producto interno de `IndexFlatIP` equivale a la
    similitud coseno (Sección 8.2, ecuación 4). Un vector nulo se deja en cero
    en lugar de dividir por cero: no puede ser el más similar a nada porque su
    producto interno con cualquier consulta es 0.
    """
    vectores = np.ascontiguousarray(vectores, dtype=np.float32)
    if vectores.ndim != 2:
        raise ValueError(f"Se esperaba una matriz 2-D de vectores, llegó shape={vectores.shape}")
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    seguras = np.where(normas == 0.0, 1.0, normas)
    return np.ascontiguousarray(vectores / seguras, dtype=np.float32)


class Encoder(Protocol):
    """Contrato mínimo que la Fase 4 necesita de un encoder."""

    nombre: str
    dimension: int

    def codificar_pasajes(self, textos: Sequence[str]) -> np.ndarray: ...

    def codificar_consultas(self, textos: Sequence[str]) -> np.ndarray: ...

    def contar_tokens(self, textos: Sequence[str]) -> list[int]: ...


class SentenceTransformerEncoder:
    """Encoder real respaldado por `sentence-transformers`.

    Siempre devuelve vectores normalizados en float32, listos para
    `IndexFlatIP`.
    """

    def __init__(
        self,
        nombre_modelo: str = MODELO_POR_DEFECTO,
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_seq_length: int | None = MAX_SEQ_LENGTH_POR_DEFECTO,
        mostrar_progreso: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.nombre = nombre_modelo
        self.batch_size = batch_size
        self.mostrar_progreso = mostrar_progreso
        self.prefijo_consulta, self.prefijo_pasaje = prefijos_para_modelo(nombre_modelo)

        self._modelo = SentenceTransformer(nombre_modelo, device=device)
        if max_seq_length is not None:
            self._modelo.max_seq_length = max_seq_length
        self.device = str(self._modelo.device)
        self.max_seq_length = int(self._modelo.max_seq_length)
        # sentence-transformers 5.x renombró el método; soportamos ambos para no
        # atarnos a una versión concreta de la librería.
        obtener_dim = getattr(
            self._modelo, "get_embedding_dimension", None
        ) or self._modelo.get_sentence_embedding_dimension
        self.dimension = int(obtener_dim())

    def _codificar(self, textos: Sequence[str], prefijo: str) -> np.ndarray:
        if not textos:
            return np.zeros((0, self.dimension), dtype=np.float32)
        entradas = [prefijo + t for t in textos] if prefijo else list(textos)
        vectores = self._modelo.encode(
            entradas,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=self.mostrar_progreso and len(entradas) > self.batch_size,
        )
        # Re-normalizamos por si el modelo devuelve normas ligeramente distintas
        # de 1 (acumulación en fp16); es idempotente sobre vectores unitarios.
        return normalizar(vectores)

    def codificar_pasajes(self, textos: Sequence[str]) -> np.ndarray:
        return self._codificar(textos, self.prefijo_pasaje)

    def codificar_consultas(self, textos: Sequence[str]) -> np.ndarray:
        return self._codificar(textos, self.prefijo_consulta)

    def contar_tokens(self, textos: Sequence[str]) -> list[int]:
        """Cuenta tokens con el tokenizer *de este* modelo.

        La Fase 3 puede haber contado con otro tokenizer. Si el conteo real
        supera `max_seq_length`, el encoder trunca en silencio y se pierde texto
        del fragmento, así que conviene detectarlo antes de indexar.
        """
        if not textos:
            return []
        codificados = self._modelo.tokenizer(
            [self.prefijo_pasaje + t for t in textos] if self.prefijo_pasaje else list(textos),
            add_special_tokens=True,
            truncation=False,
        )
        return [len(ids) for ids in codificados["input_ids"]]


class FakeEncoder:
    """Encoder determinístico para tests, sin descargar pesos.

    Proyecta un hash estable del texto a la esfera unitaria. No tiene ninguna
    propiedad semántica: sirve para verificar alineación, persistencia y
    contratos de E/S, nunca calidad de recuperación.
    """

    def __init__(self, dimension: int = 8, nombre: str = "fake/encoder-test") -> None:
        self.nombre = nombre
        self.dimension = dimension
        self.prefijo_consulta, self.prefijo_pasaje = ("", "")
        self.device = "cpu"
        self.max_seq_length = 512

    def _codificar(self, textos: Sequence[str]) -> np.ndarray:
        if not textos:
            return np.zeros((0, self.dimension), dtype=np.float32)
        filas = []
        for texto in textos:
            # hashlib y no hash(): el hash de str está aleatorizado por proceso,
            # y estos vectores deben ser idénticos entre la corrida que indexa y
            # la que busca (los tests de CLI usan subprocesos distintos).
            digest = hashlib.blake2b(texto.encode("utf-8"), digest_size=4).digest()
            semilla = int.from_bytes(digest, "big")
            rng = np.random.default_rng(semilla)
            filas.append(rng.standard_normal(self.dimension))
        return normalizar(np.vstack(filas))

    def codificar_pasajes(self, textos: Sequence[str]) -> np.ndarray:
        return self._codificar(textos)

    def codificar_consultas(self, textos: Sequence[str]) -> np.ndarray:
        return self._codificar(textos)

    def contar_tokens(self, textos: Sequence[str]) -> list[int]:
        return [len(t.split()) for t in textos]


def crear_encoder(
    nombre_modelo: str = MODELO_POR_DEFECTO,
    *,
    device: str | None = None,
    batch_size: int = 32,
    max_seq_length: int | None = MAX_SEQ_LENGTH_POR_DEFECTO,
    mostrar_progreso: bool = True,
    dimension: int | None = None,
) -> Encoder:
    """Fábrica de encoders. `fake` construye el encoder de pruebas.

    También reconoce el nombre con el que `FakeEncoder` se identifica en el
    manifiesto (`fake/...`), para que una base construida con él pueda volver a
    cargarse desde disco sin salir a HuggingFace. `dimension` solo aplica a ese
    caso: en un encoder real la dimensión la fija el modelo.
    """
    if nombre_modelo == "fake" or nombre_modelo.startswith("fake/"):
        return FakeEncoder(
            dimension=dimension if dimension is not None else 8,
            nombre=nombre_modelo if nombre_modelo != "fake" else "fake/encoder-test",
        )
    return SentenceTransformerEncoder(
        nombre_modelo,
        device=device,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        mostrar_progreso=mostrar_progreso,
    )
