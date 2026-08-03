import json
from pathlib import Path

from codefest_ad_astra.chunking.pipeline import procesar_documentos
from codefest_ad_astra.ingest.validation import Document


def _contar_palabras(texto: str) -> int:
    return len(texto.split())


def _escribir_documentos_jsonl(tmp_path: Path, documentos: list[Document]) -> Path:
    ruta = tmp_path / "documentos.jsonl"
    with open(ruta, "w", encoding="utf-8") as f:
        for doc in documentos:
            f.write(doc.to_json_line() + "\n")
    return ruta


def test_procesar_documentos_genera_fragmentos_para_cada_documento(tmp_path):
    documentos = [
        Document(doc_id="DOC-1", fuente="a.pdf", formato="pdf", fenomeno=1, idioma="es",
                  texto="Primera oración. Segunda oración."),
        Document(doc_id="DOC-2", fuente="b.pdf", formato="pdf", fenomeno=2, idioma="en",
                  texto="First sentence. Second sentence."),
    ]
    ruta = _escribir_documentos_jsonl(tmp_path, documentos)

    fragmentos = list(procesar_documentos(ruta, max_tokens=100, contar_tokens=_contar_palabras))

    assert len(fragmentos) == 2
    ids_doc = {f.doc_id for f in fragmentos}
    assert ids_doc == {"DOC-1", "DOC-2"}


def test_procesar_documentos_con_archivo_vacio_no_produce_fragmentos(tmp_path):
    ruta = _escribir_documentos_jsonl(tmp_path, [])
    fragmentos = list(procesar_documentos(ruta, contar_tokens=_contar_palabras))
    assert fragmentos == []


def test_main_escribe_fragments_jsonl_con_una_linea_por_fragmento(tmp_path, monkeypatch, capsys):
    documentos = [
        Document(doc_id="DOC-1", fuente="a.pdf", formato="pdf", fenomeno=1, idioma="es",
                  texto="Primera oración. Segunda oración."),
    ]
    entrada = _escribir_documentos_jsonl(tmp_path, documentos)
    salida = tmp_path / "fragments.jsonl"

    import sys
    from codefest_ad_astra.chunking import pipeline as modulo_pipeline

    monkeypatch.setattr(
        modulo_pipeline, "count_tokens",
        lambda texto, model_name=None: _contar_palabras(texto),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["pipeline.py", "--documentos", str(entrada), "--salida", str(salida)],
    )

    modulo_pipeline.main()

    assert salida.exists()
    lineas = salida.read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 1
    datos = json.loads(lineas[0])
    assert datos["doc_id"] == "DOC-1"
