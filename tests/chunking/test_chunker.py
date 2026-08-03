from codefest_ad_astra.chunking.chunker import chunk_document
from codefest_ad_astra.ingest.validation import Document


def _contar_palabras(texto: str) -> int:
    """Fake determinista: 1 token por palabra, sin red ni modelo real."""
    return len(texto.split())


def _doc(texto: str, **overrides) -> Document:
    base = dict(
        doc_id="DOC-1",
        fuente="F1_IA/reporte.pdf",
        formato="pdf",
        fenomeno=1,
        idioma="es",
        texto=texto,
    )
    base.update(overrides)
    return Document(**base)


def test_parrafo_unico_que_cabe_completo_da_un_solo_chunk():
    doc = _doc("Primera oración corta. Segunda oración corta también.")
    fragmentos = chunk_document(doc, max_tokens=100, contar_tokens=_contar_palabras)

    assert len(fragmentos) == 1
    assert fragmentos[0].texto == "Primera oración corta. Segunda oración corta también."
    assert fragmentos[0].posicion == 0
    assert fragmentos[0].chunk_id == "DOC-1-chunk-000"
    assert fragmentos[0].doc_id == "DOC-1"
    assert fragmentos[0].fuente == "F1_IA/reporte.pdf"
    assert fragmentos[0].formato == "pdf"
    assert fragmentos[0].fenomeno == 1
    assert fragmentos[0].idioma == "es"


def test_dos_parrafos_que_no_caben_juntos_dan_dos_chunks():
    parrafo_1 = "Uno dos tres cuatro cinco. Seis siete ocho nueve diez."
    parrafo_2 = "Once doce trece catorce quince. Dieciseis diecisiete dieciocho."
    doc = _doc(f"{parrafo_1}\n\n{parrafo_2}")

    # cada párrafo tiene 10 palabras (incluye puntuación pegada, cuenta por
    # split en espacios); presupuesto de 12 obliga a separarlos
    fragmentos = chunk_document(doc, max_tokens=12, overlap_sentences=0, contar_tokens=_contar_palabras)

    assert len(fragmentos) == 2
    assert fragmentos[0].posicion == 0
    assert fragmentos[1].posicion == 1
    assert fragmentos[0].chunk_id == "DOC-1-chunk-000"
    assert fragmentos[1].chunk_id == "DOC-1-chunk-001"


def test_ninguna_oracion_queda_cortada_entre_chunks():
    texto = (
        "Primera oración del documento. Segunda oración del documento. "
        "Tercera oración un poco más larga que las anteriores. "
        "Cuarta oración también presente aquí. Quinta y última oración."
    )
    doc = _doc(texto)
    fragmentos = chunk_document(doc, max_tokens=8, overlap_sentences=0, contar_tokens=_contar_palabras)

    assert len(fragmentos) > 1
    for fragmento in fragmentos:
        assert fragmento.texto.strip().endswith((".", "!", "?"))
        # cada fragmento es una concatenación de oraciones completas
        assert not fragmento.texto.strip().startswith((",", ";"))


def test_solapamiento_repite_la_ultima_oracion_del_chunk_anterior():
    texto = (
        "Oración A del primer párrafo. Oración B del primer párrafo.\n\n"
        "Oración C del segundo párrafo. Oración D del segundo párrafo."
    )
    doc = _doc(texto)
    fragmentos = chunk_document(doc, max_tokens=8, overlap_sentences=1, contar_tokens=_contar_palabras)

    assert len(fragmentos) >= 2
    # la primera oración del segundo chunk debe ser la última del primero
    ultima_oracion_chunk_0 = fragmentos[0].texto.split(". ")[-1].strip()
    assert fragmentos[1].texto.startswith(ultima_oracion_chunk_0.rstrip("."))


def test_oracion_individual_mas_grande_que_el_presupuesto_se_emite_sola():
    oracion_larga = "Palabra " * 20 + "final."
    doc = _doc(oracion_larga.strip())
    fragmentos = chunk_document(doc, max_tokens=5, overlap_sentences=0, contar_tokens=_contar_palabras)

    assert len(fragmentos) == 1
    assert fragmentos[0].texto == oracion_larga.strip()


def test_documento_vacio_no_produce_fragmentos():
    doc = _doc("")
    assert chunk_document(doc, contar_tokens=_contar_palabras) == []


def test_num_tokens_usa_la_funcion_de_conteo_inyectada():
    doc = _doc("Una oración con cinco palabras exactas.")
    fragmentos = chunk_document(doc, max_tokens=100, contar_tokens=_contar_palabras)

    assert fragmentos[0].num_tokens == _contar_palabras(fragmentos[0].texto)
