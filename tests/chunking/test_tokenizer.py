import pytest

from codefest_ad_astra.chunking.tokenizer import count_tokens, DEFAULT_ENCODER


class _FakeTokenizer:
    """Simula un tokenizer real: 1 token por palabra + 2 tokens especiales
    (equivalente a [CLS]/[SEP]), suficiente para probar la lógica de
    conteo sin descargar nada de HuggingFace."""

    def encode(self, text, add_special_tokens=True):
        tokens = text.split()
        return tokens + (["<esp>", "<esp>"] if add_special_tokens else [])


def test_cuenta_tokens_con_tokenizer_inyectado():
    fake = _FakeTokenizer()
    assert count_tokens("hola mundo", tokenizer=fake) == 4  # 2 palabras + 2 especiales


def test_texto_vacio_da_solo_tokens_especiales():
    fake = _FakeTokenizer()
    assert count_tokens("", tokenizer=fake) == 2


def test_default_encoder_es_bge_m3():
    assert DEFAULT_ENCODER == "BAAI/bge-m3"


@pytest.mark.integration
def test_tokenizer_real_cuenta_mas_de_cero_tokens():
    assert count_tokens("La inteligencia artificial en la defensa nacional.") > 0
