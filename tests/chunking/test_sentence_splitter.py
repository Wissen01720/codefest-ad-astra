from codefest_ad_astra.chunking.sentence_splitter import split_sentences


def test_dos_oraciones_simples_es():
    texto = "La IA transforma la defensa. Los riesgos son reales."
    assert split_sentences(texto) == [
        "La IA transforma la defensa.",
        "Los riesgos son reales.",
    ]


def test_oracion_en_ingles_con_pregunta_y_exclamacion():
    texto = "Is this safe? It might not be! We must check."
    assert split_sentences(texto) == [
        "Is this safe?",
        "It might not be!",
        "We must check.",
    ]


def test_no_corta_en_abreviatura_conocida():
    texto = "El Dr. Pérez firmó el informe. Luego lo publicó el CENIA."
    assert split_sentences(texto) == [
        "El Dr. Pérez firmó el informe.",
        "Luego lo publicó el CENIA.",
    ]


def test_no_corta_en_numero_decimal():
    texto = "El índice subió a 3.14 puntos este año. Es un récord."
    assert split_sentences(texto) == [
        "El índice subió a 3.14 puntos este año.",
        "Es un récord.",
    ]


def test_no_corta_en_puntos_suspensivos():
    texto = "Y entonces... nadie lo esperaba. El resultado fue sorprendente."
    assert split_sentences(texto) == [
        "Y entonces... nadie lo esperaba.",
        "El resultado fue sorprendente.",
    ]


def test_texto_vacio_devuelve_lista_vacia():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_texto_de_una_sola_oracion():
    assert split_sentences("Solo una oración sin punto final") == [
        "Solo una oración sin punto final"
    ]


def test_oracion_en_portugues():
    texto = "A segurança espacial é crítica. O lixo orbital cresce rápido."
    assert split_sentences(texto) == [
        "A segurança espacial é crítica.",
        "O lixo orbital cresce rápido.",
    ]
