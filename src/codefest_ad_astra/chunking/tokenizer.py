"""Conteo de tokens usando el tokenizer real del encoder elegido.

El límite de entrada del encoder (spec Sección 4.3, comúnmente 512 tokens)
se mide en tokens de su propio vocabulario, no en palabras -- por eso el
chunking (chunker.py) debe usar este conteo, no una aproximación.
"""
from functools import lru_cache

from transformers import AutoTokenizer

DEFAULT_ENCODER = "BAAI/bge-m3"


@lru_cache(maxsize=4)
def _cargar_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, tokenizer=None, model_name: str = DEFAULT_ENCODER) -> int:
    """Cuenta los tokens que produciría el tokenizer de `model_name` al
    codificar `text`, incluyendo tokens especiales ([CLS]/[SEP] o equivalente).

    Pasa `tokenizer` explícitamente en tests para no depender de red/descarga.
    """
    tokenizer = tokenizer or _cargar_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=True))
