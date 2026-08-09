import numpy as np
import pytest

from codefest_ad_astra.indexing.encoder import DEFAULT_ENCODER, encode_texts, load_encoder


class _FakeSentenceTransformer:
    """Simula la interfaz de sentence_transformers.SentenceTransformer.encode()
    lo suficiente para probar el wrapper sin descargar ningún modelo."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.llamadas: list[dict] = []

    def encode(self, textos, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        self.llamadas.append(dict(
            n_textos=len(textos),
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=convert_to_numpy,
        ))
        vectores = np.array(
            [[float(len(t) + i) for i in range(self.dim)] for t in textos],
            dtype="float32",
        )
        if normalize_embeddings:
            normas = np.linalg.norm(vectores, axis=1, keepdims=True)
            vectores = vectores / normas
        return vectores


def test_encode_texts_pide_normalizacion_y_numpy():
    fake = _FakeSentenceTransformer()
    encode_texts(fake, ["hola", "mundo distinto"])

    assert fake.llamadas[0]["normalize_embeddings"] is True
    assert fake.llamadas[0]["convert_to_numpy"] is True
    assert fake.llamadas[0]["n_textos"] == 2


def test_encode_texts_devuelve_vectores_de_norma_unitaria():
    fake = _FakeSentenceTransformer()
    vectores = encode_texts(fake, ["hola", "mundo distinto", "x"])

    normas = np.linalg.norm(vectores, axis=1)
    np.testing.assert_allclose(normas, 1.0, atol=1e-6)


def test_encode_texts_devuelve_float32():
    fake = _FakeSentenceTransformer()
    vectores = encode_texts(fake, ["hola"])
    assert vectores.dtype == np.float32


def test_default_encoder_es_bge_m3():
    assert DEFAULT_ENCODER == "BAAI/bge-m3"


@pytest.mark.integration
def test_load_encoder_y_encode_texts_con_modelo_real_pequeno():
    modelo = load_encoder("sentence-transformers/paraphrase-MiniLM-L3-v2")
    vectores = encode_texts(modelo, ["hola mundo", "hello world"])

    assert vectores.shape[0] == 2
    normas = np.linalg.norm(vectores, axis=1)
    np.testing.assert_allclose(normas, 1.0, atol=1e-3)
