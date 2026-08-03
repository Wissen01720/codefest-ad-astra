from codefest_ad_astra.chunking.chunker import chunk_document
from codefest_ad_astra.ingest.validation import Document


def _contar_palabras(texto: str) -> int:
    """Fake determinista: 1 token por palabra, sin red ni modelo real."""
    return len(texto.split())


def _contar_caracteres(texto: str) -> int:
    """Fake determinista: 1 token por caracter -- útil para simular una
    única "palabra" (sin espacios) que por sí sola supera max_tokens,
    algo que _contar_palabras no puede representar (cuenta 1 token por
    palabra sin importar su longitud)."""
    return len(texto)


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
    # max_tokens=12 (en vez de 8): con el fix del hallazgo #1, una cola de
    # overlap que ya no cabe junto a la siguiente oración se descarta en vez
    # de mezclarse por encima del presupuesto -- por eso el solapamiento
    # necesita margen real (tail + oración siguiente <= max_tokens) para
    # sobrevivir. Con oraciones de 5 tokens cada una, max_tokens=8 no dejaba
    # margen (5+5=10>8) y el fix correctamente descartaba la cola siempre;
    # max_tokens=12 sí deja margen (5+5=10<=12) y el solapamiento se
    # preserva como se espera.
    fragmentos = chunk_document(doc, max_tokens=12, overlap_sentences=1, contar_tokens=_contar_palabras)

    assert len(fragmentos) >= 2
    # la primera oración del segundo chunk debe ser la última del primero
    ultima_oracion_chunk_0 = fragmentos[0].texto.split(". ")[-1].strip()
    assert fragmentos[1].texto.startswith(ultima_oracion_chunk_0.rstrip("."))


def test_oracion_individual_mas_grande_que_el_presupuesto_se_divide_por_palabras():
    """Una "oración" sobredimensionada (aquí, una oración real y larga, pero
    el mismo código de fallback también cubre el caso de texto tabular sin
    puntuación -- ver test_documento_tabular_sin_puntuacion_se_divide_en_...)
    ya no se emite verbatim por encima del presupuesto: se reparte en varias
    piezas por palabras, cada una dentro de max_tokens, sin perder contenido."""
    oracion_larga = "Palabra " * 20 + "final."
    doc = _doc(oracion_larga.strip())
    fragmentos = chunk_document(doc, max_tokens=5, overlap_sentences=0, contar_tokens=_contar_palabras)

    assert len(fragmentos) > 1
    for fragmento in fragmentos:
        assert fragmento.num_tokens <= 5
    reconstruido = " ".join(f.texto for f in fragmentos)
    assert reconstruido.split() == oracion_larga.strip().split()


def test_palabra_unica_mas_grande_que_el_presupuesto_se_emite_sola():
    """Una única "palabra" (sin espacios) que por sí sola supera max_tokens
    no tiene forma de dividirse por límites de palabra -- ese es el único
    caso legítimo que queda "sola" por encima del presupuesto."""
    palabra_larga = "x" * 50
    doc = _doc(palabra_larga)
    fragmentos = chunk_document(doc, max_tokens=5, overlap_sentences=0, contar_tokens=_contar_caracteres)

    assert len(fragmentos) == 1
    assert fragmentos[0].texto == palabra_larga


def test_documento_vacio_no_produce_fragmentos():
    doc = _doc("")
    assert chunk_document(doc, contar_tokens=_contar_palabras) == []


def test_num_tokens_usa_la_funcion_de_conteo_inyectada():
    doc = _doc("Una oración con cinco palabras exactas.")
    fragmentos = chunk_document(doc, max_tokens=100, contar_tokens=_contar_palabras)

    assert fragmentos[0].num_tokens == _contar_palabras(fragmentos[0].texto)


def test_oraciones_sobredimensionadas_consecutivas_no_duplican_chunk():
    """Regresión: dos oraciones sobredimensionadas seguidas no deben hacer
    que el buffer congelado (la cola de overlap) se reemita dos veces como
    chunks duplicados con texto idéntico.
    """
    oracion_corta = "Uno dos tres."
    oracion_grande_1 = "Palabra " * 10 + "fin uno."
    oracion_grande_2 = "Palabra " * 10 + "fin dos."
    texto = f"{oracion_corta} {oracion_grande_1} {oracion_grande_2}"
    doc = _doc(texto)

    fragmentos = chunk_document(doc, max_tokens=5, overlap_sentences=1, contar_tokens=_contar_palabras)

    textos = [f.texto for f in fragmentos]
    assert len(textos) == len(set(textos)), f"chunks duplicados en: {textos}"


def test_cola_de_overlap_mas_oracion_siguiente_no_excede_max_tokens():
    """Regresión del hallazgo crítico de la revisión de rama completa: en el
    camino por defecto (overlap_sentences=1), tras `_cerrar_chunk_actual()`
    la cola de overlap queda en `buffer` con `buffer_tiene_contenido_nuevo =
    False`. Si la oración siguiente no cabe junto a esa cola, el segundo
    `_cerrar_chunk_actual()` es un no-op (correcto, evita duplicados) pero
    la oración se agregaba al buffer sin verificar que aún cupiera --
    dejando un buffer por encima de max_tokens hasta el próximo cierre real,
    que lo emitía tal cual (hasta ~2x max_tokens en el peor caso).

    Con S1=4, S2=6, S3=6, S4=2 tokens y max_tokens=10: S1+S2 llenan el
    primer chunk (10 tokens) y cierran; la cola es S2 (6 tokens). S3 (6
    tokens) no cabe junto a la cola (6+6=12>10) -- exactamente el escenario
    que dispara el bug. Antes del fix, el segundo chunk emitido terminaba
    siendo S2+S3 = 12 tokens > max_tokens=10.
    """
    s1 = "Uno dos tres cuatro."
    s2 = "Cinco seis siete ocho nueve diez."
    s3 = "Once doce trece catorce quince dieciseis."
    s4 = "Diecisiete dieciocho."
    texto = f"{s1} {s2} {s3} {s4}"
    doc = _doc(texto)

    max_tokens = 10
    fragmentos = chunk_document(doc, max_tokens=max_tokens, overlap_sentences=1, contar_tokens=_contar_palabras)

    assert len(fragmentos) >= 2
    for fragmento in fragmentos:
        assert fragmento.num_tokens <= max_tokens, (
            f"chunk excede max_tokens={max_tokens}: {fragmento.num_tokens} tokens -> {fragmento.texto!r}"
        )


def test_documento_tabular_sin_puntuacion_se_divide_en_varios_chunks():
    """Simula el escenario de extractors.py para CSV/XLSX/PBF: filas tipo
    'columna: valor' unidas con un solo '\\n' (no crea párrafos separados,
    _dividir_en_parrafos solo corta en '\\n{2,}') y sin puntuación de cierre
    de oración en ninguna parte -- split_sentences no encuentra ningún punto
    de corte y todo el documento se convierte en una única "oración"
    gigante. Sin el fallback de la fase 4, esto se emitía verbatim como un
    único chunk masivamente sobredimensionado que el encoder truncaba en
    silencio. Con el fallback, se debe partir en varios chunks por límites
    de palabra, cada uno dentro del presupuesto, sin perder contenido.
    """
    filas = [f"campo{i}: valor{i} campo{i}b: valor{i}b" for i in range(20)]
    texto = "\n".join(filas)
    doc = _doc(texto, formato="csv")

    max_tokens = 10
    fragmentos = chunk_document(doc, max_tokens=max_tokens, overlap_sentences=0, contar_tokens=_contar_palabras)

    assert len(fragmentos) > 1
    for fragmento in fragmentos:
        assert fragmento.num_tokens <= max_tokens

    reconstruido = " ".join(f.texto for f in fragmentos).split()
    assert reconstruido == texto.split()
