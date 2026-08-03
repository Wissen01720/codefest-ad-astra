# CodeFest Ad Astra 2026 — Etapa 1: Base de Conocimiento Vectorial

Proyecto para la Etapa 1 (fase virtual) del CODEFEST AD ASTRA 2026 (Universidad de los Andes / Fuerza Aeroespacial Colombiana). Objetivo del reto: construir una base de conocimiento vectorial (FAISS + metadata + encoder(s), opcionalmente grafo de conocimiento) a partir de un corpus multiformato sobre tres fenómenos (IA en defensa, seguridad espacial/LEO, dinámicas territoriales en LATAM), y responder 50 consultas (`q001`-`q050`) devolviendo top-3 documentos y top-10 fragmentos por consulta en `resultados.jsonl`.

## Estado actual: Fase 1 (extracción) + Fase 2 (limpieza) completas

Lo construido hasta ahora cubre el **preprocesamiento de fuentes** (Sección 2 de la especificación). Chunking, encoders, índice FAISS, recuperación y grafo de conocimiento **todavía no están implementados**.

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

### Cómo correr el pipeline de extracción

```bash
uv run python -m codefest_ad_astra.ingest.pipeline --corpus data/raw --salida data/processed/documentos.jsonl
```

Re-validar PDFs a gran escala:

```bash
uv run python -m codefest_ad_astra.ingest.batch_runner --corpus <ruta>
```

## Pendiente (no implementado)

Según la especificación técnica (`CODEFEST_2026-1.pdf`), falta:

1. **Chunking** (Sección 3): fragmentar `documentos.jsonl` con la estrategia elegida, respetando completitud lingüística (sin cortar oraciones) y generando la metadata obligatoria por fragmento (`doc_id`, `chunk_id`, `fuente`, `formato`, `fenomeno`, `posicion`, `num_tokens`, `texto`).
2. **Encoder(s)** (Sección 4): elegir modelo(s) encoder multilingüe(s) de HuggingFace (prohibido usar decoders/LLMs generativos en esta etapa).
3. **Índice FAISS** (Sección 5): construir `index.faiss` + `metadata.jsonl` por encoder (probablemente `IndexFlatIP` con vectores normalizados dado el volumen del corpus).
4. **Módulo de recuperación** (Sección 8): búsqueda por similitud coseno, combinación de múltiples encoders si aplica (CombSUM/CombMNZ/RRF), agregación a nivel documento, post-filtros por metadata.
5. **Formato de salida** (Sección 9): generar `resultados.jsonl` con las 50 consultas, top-3 documentos y top-10 fragmentos (≤250 palabras c/u).
6. **Script `generador.py`** que reproduzca `resultados.jsonl` a partir del índice.
7. **Documento técnico** (PDF, máx. 8 páginas) con justificación de decisiones de diseño.
8. *(Bonus, opcional)* **Grafo de conocimiento** (Sección 7): NER + extracción de relaciones + integración con la recuperación vectorial.

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
