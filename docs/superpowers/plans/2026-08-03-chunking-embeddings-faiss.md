# Fase 3 (Chunking) + Fase 4 (Embeddings + Índice FAISS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convertir `data/processed/documentos.jsonl` (salida ya construida de Fase 1+2) en `fragments.jsonl` (chunks con metadata obligatoria) y luego en una base vectorial FAISS persistida (`base_vectorial/encoder_<nombre>/index.faiss` + `metadata.jsonl`), lista para que la Fase 6 (recuperación, plan aparte) la consuma.

**Architecture:** Dos paquetes nuevos, hermanos de `ingest/`: `chunking/` (paragraph-aware sentence-boundary chunking sobre el texto ya limpio) e `indexing/` (encoder wrapper + construcción/persistencia del índice FAISS). Cada pieza es una función pura testeable con inyección de dependencias (tokenizer y modelo de embeddings se pasan como parámetros), más un `pipeline.py`/`build_index.py` como CLI delgado encima. Los tests unitarios usan fakes (sin red); un puñado de tests marcados `integration` validan contra el tokenizer/modelo real de HuggingFace.

**Tech Stack:** Python 3.12, `transformers` (AutoTokenizer), `sentence-transformers`, `faiss-cpu`, `pytest` (nuevo, dev-only).

## Global Constraints

- Ninguna oración puede quedar cortada entre dos chunks consecutivos — el corte debe caer siempre en un límite oracional completo (spec Sección 3.3, "Requisito obligatorio").
- Metadata obligatoria por fragmento (Tabla 1, spec Sección 3.4): `doc_id`, `chunk_id`, `fuente`, `formato`, `fenomeno`, `posicion` (entero, empieza en 0), `num_tokens`, `texto`. Campos extra están permitidos.
- Los chunks no deben superar el límite de entrada del encoder elegido (spec Sección 4.3, "comúnmente 512 tokens").
- Prohibido usar cualquier arquitectura decoder/LLM generativo en indexación o recuperación (spec Sección 4.2 y 8.3) — el pipeline de esta fase solo usa encoders (familia BERT/XLM-R) y FAISS.
- El encoder debe ser un modelo público de HuggingFace bajo licencia libre (Apache 2.0 / MIT / CC BY preferidas) — spec Sección 4.3. Encoder de arranque acordado con el equipo: `BAAI/bge-m3` (multilingüe ES/EN/PT, buen MTEB/BEIR). Es un parámetro de CLI, no está hardcodeado — Dupla B puede cambiarlo sin tocar el código de chunking/indexing.
- Antes de insertar vectores en el índice deben normalizarse a norma unitaria, para que `IndexFlatIP` equivalga a similitud coseno (spec Sección 6, paso 5, y Sección 8.2).
- El índice se serializa con `faiss.write_index()` (spec Sección 1.4) y debe ser cargable después con `faiss.read_index()` sin dependencias adicionales.
- El orden de las líneas de `metadata.jsonl` debe coincidir exactamente con los IDs internos asignados por FAISS al indexar (spec Sección 1.4 y 5.3) — como se insertan los vectores con `index.add()` en el mismo orden en que se escriben las líneas, los IDs internos son 0..N-1 en ese mismo orden.
- Estructura de entrega esperada para esta fase (spec Sección 1.4): `base_vectorial/encoder_<nombre>/index.faiss` y `.../metadata.jsonl`.

---

## Notas de diseño (léelas antes de empezar a picar tareas)

**Por qué "paragraph-aware" y no "jerárquica pura":** la especificación sugiere usar encabezados de Markdown/HTML como señal estructural (Sección 3.2, "Jerárquica o estructural"). Pero `ingest/extractors.py` y `ingest/cleaning.py` (Fase 1+2, ya construidas y **no se tocan** en este plan) aplanan todo a texto plano — no preservan qué línea era un `<h1>` o un `#`. Lo que sí preservan es la separación de párrafos como doble salto de línea (`cleaning.py:43`, `_LINEAS_VACIAS_MULTIPLES` colapsa 3+ saltos a exactamente 2, nunca a menos). Por eso la estrategia real e implementable con lo que ya existe es: **empaquetar párrafos completos dentro del presupuesto de tokens, y solo bajar a nivel oración cuando un párrafo no cabe solo**. Esto es la estrategia híbrida "estructural + tamaño fijo con corte en oración" que el equipo ya había decidido, adaptada a lo que el pipeline real produce.

**Por qué el conteo de tokens usa el tokenizer del encoder real (vía `transformers.AutoTokenizer`) y no una aproximación por palabras:** el límite duro que hay que respetar es el límite de entrada del encoder (spec 4.3), que se mide en tokens de su propio vocabulario (subpalabras), no en palabras. Contar con el tokenizer real es la única forma de que el chunking y el límite de 512 tokens del encoder sean consistentes de verdad.

**Inyección de dependencias para tests rápidos sin red:** tanto `chunk_document()` como `encode_texts()` reciben la función de conteo de tokens / el modelo de embeddings como parámetro, con un default que sí usa la librería real. Los tests unitarios pasan fakes deterministas (cuentan palabras, o devuelven vectores fijos) — corren en milisegundos y no necesitan descargar nada. Un puñado de tests marcados `@pytest.mark.integration` sí cargan el tokenizer/modelo real (primera vez descargan de HuggingFace, después quedan cacheados) para confirmar que la integración real funciona.

**Qué NO cubre este plan (queda para el plan de Fase 6):** el módulo de recuperación (consulta → encoder → búsqueda FAISS → agregación a nivel documento → fusión multi-encoder → `resultados.jsonl`). Este plan termina en el índice FAISS persistido y su metadata.

---

## Mapa de archivos

```
pyproject.toml                                    (modificar: +pytest dev, +transformers)

src/codefest_ad_astra/chunking/
├── __init__.py                                   (nuevo)
├── sentence_splitter.py                          (nuevo)
├── tokenizer.py                                  (nuevo)
├── fragment.py                                   (nuevo)
├── chunker.py                                     (nuevo)
└── pipeline.py                                    (nuevo, CLI Fase 3)

src/codefest_ad_astra/indexing/
├── __init__.py                                   (nuevo)
├── encoder.py                                     (nuevo)
└── build_index.py                                 (nuevo, CLI Fase 4)

tests/
├── __init__.py                                    (nuevo)
├── chunking/
│   ├── __init__.py                                (nuevo)
│   ├── test_sentence_splitter.py                  (nuevo)
│   ├── test_tokenizer.py                          (nuevo)
│   ├── test_fragment.py                           (nuevo)
│   ├── test_chunker.py                             (nuevo)
│   └── test_pipeline.py                            (nuevo)
└── indexing/
    ├── __init__.py                                 (nuevo)
    ├── test_encoder.py                              (nuevo)
    └── test_build_index.py                          (nuevo)
```

---

### Task 1: Dependencias — pytest (dev) + transformers (runtime)

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `pytest` ejecutable vía `uv run pytest`; `transformers.AutoTokenizer` importable en el entorno.

- [ ] **Step 1: Agregar `transformers` a las dependencias y `pytest` como grupo de desarrollo**

Edita `pyproject.toml` así:

```toml
[project]
name = "codefest-ad-astra"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "beautifulsoup4>=4.15.0",
    "faiss-cpu>=1.14.3",
    "ipykernel>=7.3.0",
    "jupyter>=1.1.1",
    "langdetect>=1.0.9",
    "mapbox-vector-tile>=2.2.0",
    "numpy>=2.5.1",
    "openpyxl>=3.1.5",
    "pandas>=3.0.5",
    "pdfplumber>=0.11.10",
    "pillow>=12.3.0",
    "pytesseract>=0.3.13",
    "sentence-transformers>=5.6.1",
    "transformers>=4.44.0",
]

[project.scripts]
codefest-ad-astra = "codefest_ad_astra:main"

[dependency-groups]
dev = ["pytest>=8.0.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: prueba contra modelos/tokenizers reales de HuggingFace (lenta, requiere internet la primera vez que descarga)",
]

[build-system]
requires = ["uv_build>=0.12.0,<0.13.0"]
build-backend = "uv_build"
```

- [ ] **Step 2: Sincronizar el entorno**

Run: `uv sync --group dev`
Expected: instala `pytest` y `transformers` sin errores.

- [ ] **Step 3: Verificar que pytest corre (sin tests todavía, debe reportar "no tests ran")**

Run: `uv run pytest`
Expected: `no tests ran` (todavía no existe `tests/`, es esperado en este punto).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: agregar pytest (dev) y transformers para Fase 3/4"
```

---

### Task 2: Divisor de oraciones multilingüe (ES/EN/PT)

**Files:**
- Create: `src/codefest_ad_astra/chunking/__init__.py`
- Create: `src/codefest_ad_astra/chunking/sentence_splitter.py`
- Test: `tests/__init__.py`
- Test: `tests/chunking/__init__.py`
- Test: `tests/chunking/test_sentence_splitter.py`

**Interfaces:**
- Produces: `split_sentences(text: str) -> list[str]` — usado por `chunker.py` (Task 5).

- [ ] **Step 1: Crear paquetes vacíos**

`src/codefest_ad_astra/chunking/__init__.py`:
```python
"""Fase 3: fragmentación (chunking) de documentos limpios en fragmentos indexables."""
```

`tests/__init__.py`:
```python
```

`tests/chunking/__init__.py`:
```python
```

- [ ] **Step 2: Escribir los tests (deben fallar: el módulo no existe)**

`tests/chunking/test_sentence_splitter.py`:
```python
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
```

- [ ] **Step 3: Correr los tests, confirmar que fallan por `ModuleNotFoundError`**

Run: `uv run pytest tests/chunking/test_sentence_splitter.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'codefest_ad_astra.chunking.sentence_splitter'`

- [ ] **Step 4: Implementar `sentence_splitter.py`**

```python
"""Divisor de oraciones para ES/EN/PT sobre texto ya limpio (Fase 2).

No usa un modelo de NLP: es una heurística basada en puntuación, con una
lista corta de abreviaturas comunes y protección de números decimales y
puntos suspensivos, para evitar el error más común de un split ingenuo
(cortar 'Dr. Pérez' o '3.14' en dos oraciones).
"""
import re

_MARCADOR = " "

_ABREVIATURAS = (
    "sr", "sra", "srta", "dr", "dra", "ing", "lic", "prof",
    "av", "no", "art", "pp", "ed", "fig", "ref", "vs", "etc",
    "ee", "uu", "eu", "cap", "vol", "num", "núm",
)

_PATRON_ABREVIATURA = re.compile(
    r"\b(?:" + "|".join(_ABREVIATURAS) + r")\.",
    re.IGNORECASE,
)
_PATRON_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_PATRON_ELIPSIS = re.compile(r"\.\.\.")
_PATRON_CORTE = re.compile(
    r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑÀ-Ý0-9¿¡"“(])'
)


def split_sentences(text: str) -> list[str]:
    """Divide `text` en oraciones completas. Nunca corta a mitad de una
    abreviatura conocida, un número decimal o unos puntos suspensivos."""
    text = text.strip()
    if not text:
        return []

    protegido = _PATRON_ABREVIATURA.sub(lambda m: m.group(0).replace(".", _MARCADOR), text)
    protegido = _PATRON_DECIMAL.sub(_MARCADOR, protegido)
    protegido = _PATRON_ELIPSIS.sub(_MARCADOR * 3, protegido)

    partes = _PATRON_CORTE.split(protegido)

    return [
        p.replace(_MARCADOR, ".").strip()
        for p in partes
        if p.strip()
    ]
```

- [ ] **Step 5: Correr los tests, confirmar que pasan**

Run: `uv run pytest tests/chunking/test_sentence_splitter.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add src/codefest_ad_astra/chunking/__init__.py src/codefest_ad_astra/chunking/sentence_splitter.py tests/__init__.py tests/chunking/__init__.py tests/chunking/test_sentence_splitter.py
git commit -m "feat(chunking): divisor de oraciones ES/EN/PT sin cortar abreviaturas/decimales"
```

---

### Task 3: Conteo de tokens con el tokenizer real del encoder

**Files:**
- Create: `src/codefest_ad_astra/chunking/tokenizer.py`
- Test: `tests/chunking/test_tokenizer.py`

**Interfaces:**
- Consumes: nada de tasks anteriores.
- Produces: `DEFAULT_ENCODER: str`, `count_tokens(text: str, tokenizer=None, model_name: str = DEFAULT_ENCODER) -> int` — usado por `chunker.py` (Task 5) y `pipeline.py` (Task 6).

- [ ] **Step 1: Escribir los tests**

`tests/chunking/test_tokenizer.py`:
```python
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
```

- [ ] **Step 2: Correr, confirmar fallo por import inexistente**

Run: `uv run pytest tests/chunking/test_tokenizer.py -v -m "not integration"`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `tokenizer.py`**

```python
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
```

- [ ] **Step 4: Correr los tests unitarios (sin red)**

Run: `uv run pytest tests/chunking/test_tokenizer.py -v -m "not integration"`
Expected: 3 passed

- [ ] **Step 5: Correr el test de integración (descarga el tokenizer real la primera vez)**

Run: `uv run pytest tests/chunking/test_tokenizer.py -v -m integration`
Expected: 1 passed (puede tardar en la primera corrida mientras descarga `BAAI/bge-m3` desde HuggingFace)

- [ ] **Step 6: Commit**

```bash
git add src/codefest_ad_astra/chunking/tokenizer.py tests/chunking/test_tokenizer.py
git commit -m "feat(chunking): conteo de tokens vía tokenizer real del encoder (BAAI/bge-m3 por defecto)"
```

---

### Task 4: `Fragment` — modelo de metadata obligatoria por chunk

**Files:**
- Create: `src/codefest_ad_astra/chunking/fragment.py`
- Test: `tests/chunking/test_fragment.py`

**Interfaces:**
- Produces: `@dataclass Fragment(doc_id, chunk_id, fuente, formato, fenomeno, posicion, num_tokens, texto, idioma="")` con `.to_json_line() -> str` — usado por `chunker.py` (Task 5), `pipeline.py` (Task 6) y `build_index.py` (Task 8).

- [ ] **Step 1: Escribir los tests**

`tests/chunking/test_fragment.py`:
```python
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
```

- [ ] **Step 2: Correr, confirmar fallo**

Run: `uv run pytest tests/chunking/test_fragment.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `fragment.py`**

```python
"""Modelo de fragmento (chunk) — metadata obligatoria de la Tabla 1 (spec Sección 3.4)."""
from dataclasses import dataclass, asdict
import json


@dataclass(slots=True)
class Fragment:
    doc_id: str
    chunk_id: str
    fuente: str
    formato: str
    fenomeno: int
    posicion: int
    num_tokens: int
    texto: str
    idioma: str = ""

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
```

- [ ] **Step 4: Correr los tests**

Run: `uv run pytest tests/chunking/test_fragment.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/codefest_ad_astra/chunking/fragment.py tests/chunking/test_fragment.py
git commit -m "feat(chunking): modelo Fragment con metadata obligatoria de la Tabla 1"
```

---

### Task 5: Chunker — empaquetado por párrafo con corte en oración

**Files:**
- Create: `src/codefest_ad_astra/chunking/chunker.py`
- Test: `tests/chunking/test_chunker.py`

**Interfaces:**
- Consumes: `split_sentences(text: str) -> list[str]` (Task 2), `Fragment` (Task 4), `codefest_ad_astra.ingest.validation.Document(doc_id, fuente, formato, fenomeno, idioma, texto)` (ya existe).
- Produces: `chunk_document(doc: Document, max_tokens: int = 400, overlap_sentences: int = 1, contar_tokens: Callable[[str], int] = count_tokens) -> list[Fragment]` — usado por `pipeline.py` (Task 6).

- [ ] **Step 1: Escribir los tests**

`tests/chunking/test_chunker.py`:
```python
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
```

- [ ] **Step 2: Correr, confirmar fallo**

Run: `uv run pytest tests/chunking/test_chunker.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `chunker.py`**

```python
"""Fase 3: empaqueta párrafos completos dentro del presupuesto de tokens del
encoder; solo baja a nivel oración cuando un párrafo entero no cabe.

Nunca corta una oración a mitad (spec Sección 3.3): el único lugar donde se
decide un corte es entre oraciones completas de `split_sentences()`.
"""
from __future__ import annotations

import re
from typing import Callable

from ..ingest.validation import Document
from .fragment import Fragment
from .sentence_splitter import split_sentences
from .tokenizer import count_tokens

_SEPARADOR_PARRAFOS = re.compile(r"\n{2,}")


def _dividir_en_parrafos(texto: str) -> list[str]:
    return [p.strip() for p in _SEPARADOR_PARRAFOS.split(texto) if p.strip()]


def chunk_document(
    doc: Document,
    max_tokens: int = 400,
    overlap_sentences: int = 1,
    contar_tokens: Callable[[str], int] = count_tokens,
) -> list[Fragment]:
    """Fragmenta `doc.texto` respetando completitud lingüística.

    Estrategia: por párrafo (separado por línea en blanco), empaquetando
    párrafos completos mientras quepan en `max_tokens`; si un párrafo no
    cabe junto al buffer actual, se cierra el chunk actual y se procesa
    ese párrafo oración por oración. `overlap_sentences` controla cuántas
    oraciones del final de un chunk se repiten al inicio del siguiente.
    """
    parrafos = _dividir_en_parrafos(doc.texto)
    if not parrafos:
        return []

    buffer: list[str] = []
    buffer_tokens = 0
    textos_de_chunks: list[str] = []

    def _cerrar_chunk_actual() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        textos_de_chunks.append(" ".join(buffer))
        cola = buffer[-overlap_sentences:] if overlap_sentences > 0 else []
        buffer = list(cola)
        buffer_tokens = sum(contar_tokens(o) for o in buffer)

    for parrafo in parrafos:
        oraciones_parrafo = split_sentences(parrafo)
        if not oraciones_parrafo:
            continue

        tokens_parrafo = sum(contar_tokens(o) for o in oraciones_parrafo)

        if buffer_tokens + tokens_parrafo <= max_tokens:
            buffer.extend(oraciones_parrafo)
            buffer_tokens += tokens_parrafo
            continue

        for oracion in oraciones_parrafo:
            tokens_oracion = contar_tokens(oracion)

            if tokens_oracion > max_tokens:
                _cerrar_chunk_actual()
                textos_de_chunks.append(oracion)
                continue

            if buffer_tokens + tokens_oracion > max_tokens:
                _cerrar_chunk_actual()

            buffer.append(oracion)
            buffer_tokens += tokens_oracion

    _cerrar_chunk_actual()

    return [
        Fragment(
            doc_id=doc.doc_id,
            chunk_id=f"{doc.doc_id}-chunk-{posicion:03d}",
            fuente=doc.fuente,
            formato=doc.formato,
            fenomeno=doc.fenomeno,
            posicion=posicion,
            num_tokens=contar_tokens(texto_chunk),
            texto=texto_chunk,
            idioma=doc.idioma,
        )
        for posicion, texto_chunk in enumerate(textos_de_chunks)
    ]
```

- [ ] **Step 4: Correr los tests**

Run: `uv run pytest tests/chunking/test_chunker.py -v`
Expected: 8 passed

Si `test_dos_parrafos_que_no_caben_juntos_dan_dos_chunks` u otro test de conteo falla por un desfase de +/-1 o 2 tokens (el conteo por `split()` de `_contar_palabras` puede diferir de tu intuición contando puntuación pegada a palabras), ajusta el `max_tokens` del test, no la lógica del chunker — el objetivo del test es la frontera párrafo/oración, no el número exacto.

- [ ] **Step 5: Commit**

```bash
git add src/codefest_ad_astra/chunking/chunker.py tests/chunking/test_chunker.py
git commit -m "feat(chunking): chunk_document con empaquetado por parrafo y corte en oracion"
```

---

### Task 6: Pipeline CLI de Fase 3 — `documentos.jsonl` → `fragments.jsonl`

**Files:**
- Create: `src/codefest_ad_astra/chunking/pipeline.py`
- Test: `tests/chunking/test_pipeline.py`

**Interfaces:**
- Consumes: `chunk_document(...)` (Task 5), `codefest_ad_astra.ingest.validation.Document` (ya existe, con `to_json_line()`).
- Produces: `procesar_documentos(documentos: Path, max_tokens=400, overlap_sentences=1, model_name=DEFAULT_ENCODER, contar_tokens=None) -> Iterator[Fragment]` — usado por `main()` de este mismo módulo; `main()` es el entry point CLI de Fase 3.

- [ ] **Step 1: Escribir los tests**

`tests/chunking/test_pipeline.py`:
```python
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
```

- [ ] **Step 2: Correr, confirmar fallo**

Run: `uv run pytest tests/chunking/test_pipeline.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `pipeline.py`**

```python
"""Orquestador de la Fase 3: lee documentos.jsonl (salida de
ingest.pipeline, Fase 1+2) y genera fragments.jsonl, un chunk por línea,
listo para la Fase 4 (embeddings + índice FAISS).

Uso:
    uv run python -m codefest_ad_astra.chunking.pipeline \
        --documentos data/processed/documentos.jsonl \
        --salida data/processed/fragments.jsonl
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

from ..ingest.validation import Document
from .chunker import chunk_document
from .fragment import Fragment
from .tokenizer import DEFAULT_ENCODER, count_tokens


def _leer_documentos(path: Path) -> Iterator[Document]:
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                yield Document(**json.loads(linea))


def procesar_documentos(
    documentos: Path,
    max_tokens: int = 400,
    overlap_sentences: int = 1,
    model_name: str = DEFAULT_ENCODER,
    contar_tokens: Callable[[str], int] | None = None,
) -> Iterator[Fragment]:
    contar = contar_tokens or partial(count_tokens, model_name=model_name)
    for doc in _leer_documentos(documentos):
        yield from chunk_document(
            doc,
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
            contar_tokens=contar,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 3: chunking de documentos.jsonl -> fragments.jsonl")
    parser.add_argument("--documentos", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--overlap-sentences", type=int, default=1)
    parser.add_argument("--modelo", type=str, default=DEFAULT_ENCODER)
    args = parser.parse_args()

    args.salida.parent.mkdir(parents=True, exist_ok=True)

    total_fragmentos = 0
    docs_vistos: set[str] = set()
    with open(args.salida, "w", encoding="utf-8") as f:
        for fragmento in procesar_documentos(
            args.documentos, args.max_tokens, args.overlap_sentences, args.modelo,
        ):
            f.write(fragmento.to_json_line() + "\n")
            total_fragmentos += 1
            docs_vistos.add(fragmento.doc_id)

    print(f"\nListo: {len(docs_vistos)} documentos -> {total_fragmentos} fragmentos -> {args.salida}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Correr los tests**

Run: `uv run pytest tests/chunking/test_pipeline.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/codefest_ad_astra/chunking/pipeline.py tests/chunking/test_pipeline.py
git commit -m "feat(chunking): CLI de Fase 3, documentos.jsonl -> fragments.jsonl"
```

---

### Task 7: Encoder wrapper — embeddings normalizados

**Files:**
- Create: `src/codefest_ad_astra/indexing/__init__.py`
- Create: `src/codefest_ad_astra/indexing/encoder.py`
- Test: `tests/indexing/__init__.py`
- Test: `tests/indexing/test_encoder.py`

**Interfaces:**
- Produces: `DEFAULT_ENCODER: str`, `load_encoder(model_name=DEFAULT_ENCODER)`, `encode_texts(model, textos: list[str], batch_size: int = 32) -> np.ndarray` (float32, norma unitaria por fila) — usado por `build_index.py` (Task 8).

- [ ] **Step 1: Crear paquete e escribir los tests**

`src/codefest_ad_astra/indexing/__init__.py`:
```python
"""Fase 4: embeddings + índice FAISS."""
```

`tests/indexing/__init__.py`:
```python
```

`tests/indexing/test_encoder.py`:
```python
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
```

- [ ] **Step 2: Correr, confirmar fallo**

Run: `uv run pytest tests/indexing/test_encoder.py -v -m "not integration"`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `encoder.py`**

```python
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
```

- [ ] **Step 4: Correr los tests unitarios**

Run: `uv run pytest tests/indexing/test_encoder.py -v -m "not integration"`
Expected: 4 passed

- [ ] **Step 5: Correr el test de integración (descarga un modelo pequeño la primera vez)**

Run: `uv run pytest tests/indexing/test_encoder.py -v -m integration`
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add src/codefest_ad_astra/indexing/__init__.py src/codefest_ad_astra/indexing/encoder.py tests/indexing/__init__.py tests/indexing/test_encoder.py
git commit -m "feat(indexing): wrapper de encoder con embeddings normalizados a norma unitaria"
```

---

### Task 8: Construcción y persistencia del índice FAISS + metadata

**Files:**
- Create: `src/codefest_ad_astra/indexing/build_index.py`
- Test: `tests/indexing/test_build_index.py`

**Interfaces:**
- Consumes: `encode_texts`, `load_encoder`, `DEFAULT_ENCODER` (Task 7).
- Produces: `construir_indice(vectores: np.ndarray) -> faiss.Index`, `guardar_base_vectorial(fragmentos: list[dict], vectores: np.ndarray, carpeta_salida: Path, nombre_encoder: str) -> None` — dejan `base_vectorial/encoder_<nombre>/index.faiss` y `.../metadata.jsonl` listos para la Fase 6 (plan aparte).

- [ ] **Step 1: Escribir los tests**

`tests/indexing/test_build_index.py`:
```python
import json

import faiss
import numpy as np
import pytest

from codefest_ad_astra.indexing.build_index import construir_indice, guardar_base_vectorial


def _vectores_normalizados(n: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    v = rng.random((n, dim)).astype("float32")
    normas = np.linalg.norm(v, axis=1, keepdims=True)
    return v / normas


def _fragmentos_de_prueba(n: int) -> list[dict]:
    return [
        {
            "doc_id": f"DOC-{i}",
            "chunk_id": f"DOC-{i}-chunk-000",
            "fuente": f"archivo-{i}.pdf",
            "formato": "pdf",
            "fenomeno": 1,
            "posicion": 0,
            "num_tokens": 10,
            "texto": f"Texto del fragmento {i}.",
        }
        for i in range(n)
    ]


def test_construir_indice_es_flatip_y_contiene_todos_los_vectores():
    vectores = _vectores_normalizados(5)
    indice = construir_indice(vectores)

    assert isinstance(indice, faiss.IndexFlatIP)
    assert indice.ntotal == 5
    assert indice.d == 4


def test_guardar_base_vectorial_escribe_index_y_metadata_en_orden(tmp_path):
    vectores = _vectores_normalizados(3)
    fragmentos = _fragmentos_de_prueba(3)

    guardar_base_vectorial(fragmentos, vectores, tmp_path, nombre_encoder="bge-m3")

    carpeta = tmp_path / "encoder_bge-m3"
    assert (carpeta / "index.faiss").exists()
    assert (carpeta / "metadata.jsonl").exists()

    indice_leido = faiss.read_index(str(carpeta / "index.faiss"))
    assert indice_leido.ntotal == 3

    lineas = (carpeta / "metadata.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 3
    for i, linea in enumerate(lineas):
        datos = json.loads(linea)
        assert datos["doc_id"] == f"DOC-{i}"  # orden de línea == orden de inserción == id interno FAISS


def test_guardar_base_vectorial_falla_si_hay_descuadre_fragmentos_vectores(tmp_path):
    vectores = _vectores_normalizados(2)
    fragmentos = _fragmentos_de_prueba(3)

    with pytest.raises(ValueError, match="Descuadre"):
        guardar_base_vectorial(fragmentos, vectores, tmp_path, nombre_encoder="bge-m3")


def test_indice_recuperado_devuelve_el_vecino_mas_cercano_esperado():
    vectores = _vectores_normalizados(4)
    indice = construir_indice(vectores)

    puntuaciones, ids = indice.search(vectores[[0]], k=1)

    assert ids[0][0] == 0  # el vecino más cercano de un vector es él mismo
    assert puntuaciones[0][0] == pytest.approx(1.0, abs=1e-4)  # coseno consigo mismo == 1
```

- [ ] **Step 2: Correr, confirmar fallo**

Run: `uv run pytest tests/indexing/test_build_index.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `build_index.py`**

```python
"""Fase 4: genera embeddings de fragments.jsonl y construye/persiste el
índice FAISS + su almacén de metadata, en la estructura de entrega exigida
(spec Sección 1.4): base_vectorial/encoder_<nombre>/{index.faiss,metadata.jsonl}

Uso:
    uv run python -m codefest_ad_astra.indexing.build_index \
        --fragmentos data/processed/fragments.jsonl \
        --salida base_vectorial \
        --encoder-nombre bge-m3 \
        --modelo BAAI/bge-m3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from .encoder import DEFAULT_ENCODER, encode_texts, load_encoder


def _leer_fragmentos(path: Path) -> list[dict]:
    fragmentos = []
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                fragmentos.append(json.loads(linea))
    return fragmentos


def construir_indice(vectores: np.ndarray) -> faiss.Index:
    """IndexFlatIP: búsqueda exacta por producto interno, equivalente a
    similitud coseno porque los vectores ya vienen normalizados (spec 5.2, 6, 8.2)."""
    dimension = vectores.shape[1]
    indice = faiss.IndexFlatIP(dimension)
    indice.add(vectores)
    return indice


def guardar_base_vectorial(
    fragmentos: list[dict],
    vectores: np.ndarray,
    carpeta_salida: Path,
    nombre_encoder: str,
) -> None:
    """Escribe index.faiss y metadata.jsonl en carpeta_salida/encoder_<nombre>/.

    La línea i de metadata.jsonl corresponde SIEMPRE al fragmento insertado
    en la posición i de `vectores`, que a su vez es el ID interno i que le
    asigna FAISS (los vectores se insertan con index.add() en ese mismo
    orden, sin IDs explícitos) -- así se cumple el requisito de la spec
    Sección 1.4/5.3 de que el orden de metadata.jsonl coincida con los IDs
    internos de FAISS.
    """
    if len(fragmentos) != vectores.shape[0]:
        raise ValueError(
            f"Descuadre entre fragmentos ({len(fragmentos)}) y vectores ({vectores.shape[0]}): "
            "el orden de metadata.jsonl debe coincidir exactamente con los IDs internos de FAISS."
        )

    carpeta_encoder = carpeta_salida / f"encoder_{nombre_encoder}"
    carpeta_encoder.mkdir(parents=True, exist_ok=True)

    indice = construir_indice(vectores)
    faiss.write_index(indice, str(carpeta_encoder / "index.faiss"))

    with open(carpeta_encoder / "metadata.jsonl", "w", encoding="utf-8") as f:
        for fragmento in fragmentos:
            f.write(json.dumps(fragmento, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 4: embeddings + índice FAISS")
    parser.add_argument("--fragmentos", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True, help="Carpeta base_vectorial/")
    parser.add_argument("--encoder-nombre", type=str, required=True, help="Subcarpeta encoder_<nombre>")
    parser.add_argument("--modelo", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    fragmentos = _leer_fragmentos(args.fragmentos)
    if not fragmentos:
        raise SystemExit(f"No se encontraron fragmentos en {args.fragmentos}")

    textos = [f["texto"] for f in fragmentos]

    modelo = load_encoder(args.modelo)
    vectores = encode_texts(modelo, textos, batch_size=args.batch_size)

    guardar_base_vectorial(fragmentos, vectores, args.salida, args.encoder_nombre)

    print(f"\nListo: {len(fragmentos)} fragmentos indexados -> {args.salida}/encoder_{args.encoder_nombre}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Correr los tests**

Run: `uv run pytest tests/indexing/test_build_index.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/codefest_ad_astra/indexing/build_index.py tests/indexing/test_build_index.py
git commit -m "feat(indexing): construir y persistir index.faiss + metadata.jsonl alineados"
```

---

### Task 9: Suite completa + smoke test opcional sobre el corpus real

**Files:**
- Modify: ninguno (solo verificación)

**Interfaces:** ninguna nueva.

- [ ] **Step 1: Correr toda la suite unitaria (rápida, sin red)**

Run: `uv run pytest -m "not integration" -v`
Expected: todos los tests de Tasks 2-8 en verde (26 tests unitarios en total).

- [ ] **Step 2: Correr la suite de integración completa (descarga modelos/tokenizers reales la primera vez)**

Run: `uv run pytest -m integration -v`
Expected: 2 passed (tokenizer real de `BAAI/bge-m3`, encoder real pequeño).

- [ ] **Step 3 (opcional, solo si `data/raw` o `data/raw_muestra` existen localmente): correr el pipeline de Fase 1+2 y luego Fase 3+4 de punta a punta**

`data/raw*` está en `.gitignore` — si no existe en tu máquina, salta este paso, no es bloqueante para el resto del equipo.

```bash
uv run python -m codefest_ad_astra.ingest.pipeline --corpus data/raw --salida data/processed/documentos.jsonl
uv run python -m codefest_ad_astra.chunking.pipeline --documentos data/processed/documentos.jsonl --salida data/processed/fragments.jsonl
uv run python -m codefest_ad_astra.indexing.build_index --fragmentos data/processed/fragments.jsonl --salida base_vectorial --encoder-nombre bge-m3
```

Verificar a ojo:
- `data/processed/fragments.jsonl` tiene más líneas que `documentos.jsonl` (cada doc se parte en 1+ fragmentos).
- Ningún `texto` de `fragments.jsonl` termina a mitad de palabra o de oración (revisar unas 10 líneas al azar).
- `base_vectorial/encoder_bge-m3/index.faiss` y `metadata.jsonl` existen y tienen el mismo número de vectores/líneas (puedes verificarlo con `faiss.read_index(...).ntotal` vs `wc -l metadata.jsonl`).

- [ ] **Step 4: Commit final (si el Step 3 generó cambios de código, no de datos — `base_vectorial/` y `data/processed/` deberían ir a `.gitignore` si aún no están)**

Revisa `.gitignore`: si `base_vectorial/` y `data/processed/` no están listados, agrégalos (son artefactos generados, no código):

```bash
git status
```

Si aparecen `base_vectorial/` o `data/processed/` como untracked y quieres ignorarlos, edita `.gitignore` y comitea solo ese cambio, no los datos generados.

---

## Resumen para Dupla B / handoff de Fase 4

- El encoder de arranque es `BAAI/bge-m3`, pero es un flag de CLI (`--modelo` en `build_index.py`) — cambiarlo no requiere tocar código.
- Si cambian de encoder, revisen si necesita prefijos especiales en el texto antes de codificar (p. ej. `intfloat/multilingual-e5-large` espera `"passage: "` / `"query: "` como prefijo) — eso NO está implementado aquí a propósito, porque afecta tanto indexación como recuperación (Fase 6) y es una decisión de Dupla B, no de este plan.
- `max_tokens=400` en el chunker deja margen bajo el límite típico de 512 tokens del encoder, y de paso deja los chunks cerca del límite de 250 palabras que exige el formato de salida final (spec Sección 9.2) — pensado para minimizar el re-trabajo de recorte en la Fase 7.
- Multi-encoder (spec Sección 4.4): `build_index.py` ya soporta correr una vez por encoder (cada corrida genera su propia `encoder_<nombre>/`); la fusión de resultados entre índices es trabajo de la Fase 6, no de este plan.
