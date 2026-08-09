import json

from codefest_ad_astra.chunking.fragment import Fragment


def _fragmento_de_prueba() -> Fragment:
    return Fragment(
        doc_id="DOC-abc123",
        chunk_id="DOC-abc123-chunk-000",
        fuente="F1_IA/CSET/reporte.pdf",
        formato="pdf",
        fenomeno=1,
        posicion=0,
        num_tokens=42,
        texto="La IA transforma la defensa.",
        idioma="es",
    )


def test_to_json_line_incluye_los_campos_obligatorios_de_la_tabla_1():
    linea = _fragmento_de_prueba().to_json_line()
    datos = json.loads(linea)

    for campo in ("doc_id", "chunk_id", "fuente", "formato", "fenomeno", "posicion", "num_tokens", "texto"):
        assert campo in datos

    assert datos["doc_id"] == "DOC-abc123"
    assert datos["chunk_id"] == "DOC-abc123-chunk-000"
    assert datos["fenomeno"] == 1
    assert datos["posicion"] == 0
    assert datos["num_tokens"] == 42


def test_to_json_line_es_una_sola_linea_sin_saltos():
    linea = _fragmento_de_prueba().to_json_line()
    assert "\n" not in linea


def test_idioma_es_opcional_y_por_defecto_vacio():
    fragmento = Fragment(
        doc_id="DOC-x", chunk_id="DOC-x-chunk-000", fuente="f.pdf",
        formato="pdf", fenomeno=2, posicion=0, num_tokens=5, texto="Texto.",
    )
    assert fragmento.idioma == ""
