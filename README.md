# CodeFest Ad Astra 2026 — Etapa 1: Base de Conocimiento Vectorial

Proyecto para la Etapa 1 (fase virtual) del CODEFEST AD ASTRA 2026 (Universidad de los Andes / Fuerza Aeroespacial Colombiana). Objetivo del reto: construir una base de conocimiento vectorial (FAISS + metadata + encoder(s), opcionalmente grafo de conocimiento) a partir de un corpus multiformato sobre tres fenómenos (IA en defensa, seguridad espacial/LEO, dinámicas territoriales en LATAM), y responder 50 consultas (`q001`-`q050`) devolviendo top-3 documentos y top-10 fragmentos por consulta en `resultados.jsonl`.

## Estado actual: Fases 1–4 implementadas

| Fase | Estado | Documento |
|---|---|---|
| 1–2 — Extracción y limpieza | Implementada | este README |
| 3 — Chunking | Implementada | [README_CHUNKING.md](README_CHUNKING.md) |
| 4 — Embeddings + índice FAISS | Implementada, **falta correrla sobre el corpus real** | [README_FASE_4.md](README_FASE_4.md) |
| 6 — Recuperación | Pendiente | — |
| 7 — `resultados.jsonl` + `generador.py` | Pendiente | — |
| 5 — Grafo de conocimiento (bonus) | Pendiente | — |

### Punto de entrada único para lo ya construido

```python
from codefest_ad_astra.ingest.extraer_y_limpiar import extraer_y_limpiar

resultado = extraer_y_limpiar(path)  # path: Path a cualquier archivo del corpus

resultado.texto                        # str, limpio y normalizado, listo para chunking
resultado.idioma                       # 'es' / 'en' / 'pt' / 'und'
resultado.paginas_originales           # int
resultado.lineas_boilerplate_removidas # bool
```

Soporta todos los formatos entregados por ADL: PDF, HTML, JSON, CSV/XLSX, TXT/Markdown, imágenes (OCR) y PBF (mapas por teselas).

### Cobertura del corpus

De 1835 recursos, **1754 (95.6%)** se extraen y limpian correctamente. Los 81 restantes están diagnosticados (no son un misterio):

| Categoría | Cantidad | Causa | ¿Recuperable? |
|---|---|---|---|
| Manifiestos/catálogos | 8 | Índices de scraping, no contenido — exclusión intencional | No aplica |
| PDFs escaneados sin OCR | ~45 | Concentrados en `Alertas_Tempranas/pdfs/Informes/` | Sí, con OCR (no implementado aún) |
| PDFs corruptos | 2 | Archivo fuente dañado | No |
| Imágenes sin texto | 4 | Fotografías, no infografías | No aplica |
| Contenido genuinamente corto | ~15 | Newsletters/artículos breves, xlsx con pocas filas | No, es el contenido real |

Detalle completo y evidencia en [README_CHUNKING.md](README_CHUNKING.md) (documento de handoff hacia la fase de chunking).

### Módulos implementados (`src/codefest_ad_astra/ingest/`)

| Módulo | Rol |
|---|---|
| `extractors.py` | Extracción cruda de texto por formato (PDF por página, HTML, JSON con heurística de campos, CSV/XLSX fila-a-fila, OCR de imágenes, PBF por capas/zoom) |
| `cleaning.py` | Limpieza y normalización: caracteres de control, encoding UTF-8, detección de idioma, remoción de headers/footers repetidos entre páginas de PDF |
| `validation.py` | Modelo `Document` (doc_id, fuente, formato, fenómeno, idioma, texto) y serialización a JSONL |
| `extraer_y_limpiar.py` | Punto de entrada único que combina extracción + limpieza para cualquier archivo del corpus |
| `pipeline.py` | Orquestador Fase 1+2: recorre `data/raw`, infiere `fenomeno` (F1/F2/F3) desde la ruta, filtra manifiestos y PBFs duplicados por zoom, genera `doc_id` determinístico (hash de la ruta relativa), escribe un `documentos.jsonl` |
| `quality_metrics.py` | `assess_text_quality`: score 0-1 y etiqueta alta/media/baja sobre el texto extraído (con caveat conocido para PDFs largos, ver handoff) |
| `pdf_validator.py` / `ocr_validator.py` | Clasificación de PDFs/imágenes individuales (`PDF_OK`, `PDF_ESCANEADO`, `PDF_CORRUPTO`, etc.) |
| `schema_detector.py` / `json_profiler.py` | Detección de esquema para los JSON heterogéneos del corpus (598 esquemas distintos mapeados en la muestra) |
| `batch_runner.py` | Re-validación de PDFs a gran escala sin bloqueos |
| `diagnostics.py` / `diagnostic_common.py` / `corpus_report.py` / `generar_reporte_final.py` / `resumen_checkpoint.py` | Capa de diagnóstico y generación de reportes de cobertura del corpus (`data/diagnostics/`) |
| `chunking.py` | Fase 3: fragmentación con corte en frontera de oración (ES/EN/PT), solapamiento, límites de tokens/palabras y fallbacks para formatos estructurados |

### Módulos implementados (`src/codefest_ad_astra/indexing/`)

| Módulo | Rol |
|---|---|
| `encoders.py` | Wrapper de encoders de HuggingFace: vectores normalizados en float32, prefijos de instrucción por familia de modelo, encoder falso determinístico para tests |
| `faiss_store.py` | Construcción del `IndexFlatIP`, persistencia atómica y carga de `index.faiss` + `metadata.jsonl`, con validación de la alineación entre ambos |
| `build_index.py` | Fase 4: CLI `chunks.jsonl` → base vectorial, con checkpointing y reanudación |
| `search.py` | Búsqueda mínima top-k por similitud coseno (verificación del índice; la recuperación completa es Fase 6) |
| `verificar.py` | Chequeo de una base vectorial ya construida contra los requisitos del entregable |

### Cómo correr el pipeline de extracción

```bash
uv run python -m codefest_ad_astra.ingest.pipeline --corpus data/raw --salida data/processed/documentos.jsonl
```

Re-validar PDFs a gran escala:

```bash
uv run python -m codefest_ad_astra.ingest.batch_runner --corpus <ruta>
```

## Cómo correr chunking (Fase 3) e indexación (Fase 4)

```bash
uv run python -m codefest_ad_astra.ingest.chunking \
  --entrada data/processed/documentos.jsonl \
  --salida data/processed/chunks.jsonl \
  --errores data/processed/chunking_errors.jsonl \
  --tokenizer-model BAAI/bge-m3

uv run python -m codefest_ad_astra.indexing.build_index \
  --entrada data/processed/chunks.jsonl \
  --salida entrega/base_vectorial \
  --modelo BAAI/bge-m3
```

Detalles en [README_CHUNKING.md](README_CHUNKING.md) y [README_FASE_4.md](README_FASE_4.md).

## Pendiente (no implementado)

Según la especificación técnica (`CODEFEST_2026-1.pdf`), falta:

1. **Correr la Fase 4 sobre el corpus real**: el módulo está listo y probado, pero en este repo no existen `data/raw` ni `data/processed/chunks.jsonl` (gitignored), así que aún no hay una base vectorial construida.
2. **Módulo de recuperación** (Sección 8): agregación a nivel documento, combinación de múltiples encoders si aplica (CombSUM/CombMNZ/RRF), post-filtros por metadata. La búsqueda top-k por similitud coseno ya existe en `codefest_ad_astra.indexing.search`.
3. **Formato de salida** (Sección 9): generar `resultados.jsonl` con las 50 consultas, top-3 documentos y top-10 fragmentos (≤250 palabras c/u).
4. **Script `generador.py`** que reproduzca `resultados.jsonl` a partir del índice.
5. **Documento técnico** (PDF, máx. 8 páginas) con justificación de decisiones de diseño.
6. *(Bonus, opcional)* **Grafo de conocimiento** (Sección 7): NER + extracción de relaciones + integración con la recuperación vectorial.

## Estructura de entrega esperada (aún no creada)

```
entrega/
  resultados.jsonl
  generador.py
  informe_tecnico.pdf
  base_vectorial/
    encoder_<nombre>/
      index.faiss
      metadata.jsonl
    grafo/            (bonus, si aplica)
      grafo.graphml
```

## Setup

```bash
uv sync
```

Requiere Python ≥3.12. Dependencias clave: `pdfplumber`, `beautifulsoup4`, `pandas`/`openpyxl`, `pytesseract`+`Pillow` (OCR), `mapbox-vector-tile` (PBF), `langdetect`, `faiss-cpu`, `sentence-transformers`.
