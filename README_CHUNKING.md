# Handoff Fase 1 + 2 → Chunking — CodeFest Ad Astra

## Punto de entrada único

No necesitas conocer `extractors.py` ni `cleaning.py` por dentro. Solo esto:

```python
from codefest_ad_astra.ingest.extraer_y_limpiar import extraer_y_limpiar

resultado = extraer_y_limpiar(path)  # path: Path a cualquier archivo del corpus

resultado.texto                        # str, ya limpio y normalizado, listo para chunking
resultado.idioma                       # 'es' / 'en' / 'pt' / 'und'
resultado.paginas_originales           # int
resultado.lineas_boilerplate_removidas # bool
```

Funciona para todos los formatos del corpus (PDF, JSON, HTML, TXT, CSV, XLSX, imágenes, PBF) — internamente decide el tratamiento correcto según el tipo de archivo.

## Antes de chunkear un documento: revisa la calidad

```python
from codefest_ad_astra.ingest.quality_metrics import assess_text_quality

calidad = assess_text_quality(resultado.texto)
calidad.quality_score   # 0.0 - 1.0
calidad.quality_label   # 'alta' / 'media' / 'baja'
```

**Nota metodológica importante:** para PDFs largos y multipágina, `quality_score` puede aparecer más bajo de lo real. Esto es un artefacto conocido: `pdfplumber` no preserva separadores de párrafo dentro de una página, así que cada página termina como una sola línea larga, y el ratio de "líneas útiles" del validador termina midiendo aproximadamente el número de páginas, no la calidad real del texto. Si vas a filtrar documentos por `quality_score`, para PDFs conviene mirar también `characters` y `words`, no solo el score aislado.

## Qué NO vas a recibir (y por qué, con evidencia)

De los 1835 recursos del corpus, **1754 (95.6%) se extraen y limpian correctamente.** Los 81 restantes están completamente diagnosticados, no son un misterio:

| Categoría | Cantidad | Causa | ¿Recuperable? |
|---|---|---|---|
| Manifiestos/catálogos | 8 | Son índices, no contenido documental — exclusión intencional | No aplica (correcto que se excluyan) |
| PDFs escaneados sin OCR | ~45 | Concentrados casi todos en `Alertas_Tempranas/pdfs/Informes/` | Sí, con OCR — no implementado por tiempo |
| PDFs corruptos | 2 | Archivo fuente dañado (`No /Root object`) | No, sin el archivo original |
| Imágenes sin texto | 4 | Son fotografías (no infografías), correctamente clasificadas | No aplica |
| Contenido genuinamente corto | ~15 | Newsletters/artículos breves, JSON con poco cuerpo, xlsx con pocas filas | No, es el contenido real de la fuente |

**Si necesitas los ~45 PDFs de Alertas Tempranas**, avísame — es la única categoría con una vía de recuperación clara (OCR), pero es trabajo nuevo, no un ajuste rápido.

## Diagnóstico disponible por si necesitas más detalle

- `pdf_validator.py` → `validate_pdf(path)` clasifica cualquier PDF individual en `PDF_OK / PDF_ESCANEADO / PDF_SIN_OCR / PDF_CORRUPTO / PDF_VACIO / PDF_CIFRADO / PDF_INCOMPLETO / PDF_ERROR`.
- `ocr_validator.py` → `validate_image(path)` hace lo mismo para imágenes.
- `schema_detector.py` / `json_profiler.py` → si te encuentras un JSON con estructura rara, esto ya lo tiene mapeado (598 esquemas distintos detectados en la muestra).
- `batch_runner.py` → si necesitas re-validar PDFs a gran escala sin que se cuelgue: `uv run python -m codefest_ad_astra.ingest.batch_runner --corpus <ruta>`.

## Estado de las restricciones del proyecto

No se tocó `extractors.py` (salvo el fix puntual de una línea en `extract_json` para manejar JSON con raíz tipo lista), no se tocó `validate_extraction.py`, y toda la capa de diagnóstico vive separada en sus propios módulos.