# Fase 3 — Chunking de documentos

## 1. Objetivo

El chunking divide los textos procesados en fragmentos manejables que mantienen el orden, la estructura y la relación con el documento original.

En el pipeline de la solución:

- Fase 1: extracción
  - Se extrae texto en bruto de PDF, HTML, TXT, CSV, XLSX, imágenes, PBF, JSON, etc.
- Fase 2: limpieza y normalización
  - Se normaliza el texto, se elimina boilerplate, se corrigen saltos y se marca el idioma.
- Fase 3: chunking
  - Se segmenta el texto limpio en chunks orientados a embeddings y búsquedas.
- Fase 4: embeddings e índice FAISS
  - Cada chunk se convierte en embedding y se indexa en FAISS para recuperación.

El propósito de esta fase es producir un JSONL de chunks fiables y reproducibles para la integración con la fase 4.

## 2. Estado actual

- La implementación de `src/codefest_ad_astra/ingest/chunking.py` está terminada y validada con datos sintéticos.
- El módulo compila correctamente.
- Las `42` pruebas existentes pasan.
- La validación con una muestra real aún está pendiente.
- La selección del encoder definitivo para la fase 4 aún está pendiente.

> No se debe afirmar que la integración real está aprobada hasta que se ejecute en datos reales y con el encoder elegido.

## 3. Archivos implementados

- `src/codefest_ad_astra/ingest/chunking.py`
  - Responsable de la lógica de chunking, segmentación de texto, conteo de tokens, generación de chunks, manejo de errores, validación de entradas y escritura atómica.
- `tests/test_chunking.py`
  - Pruebas unitarias y de integración para la lógica de chunking, preservación de texto, oversize, posiciones, orden y determinismo.
- `tests/test_chunking_cli.py`
  - Pruebas de la interfaz de línea de comandos, comportamiento de salida, argumentos inválidos, ayuda y manejo de errores en ejecución.
- `tests/test_sentence_splitter.py`
  - Pruebas de la segmentación de oraciones para abreviaturas, decimales, iniciales, signos de interrogación/exclamación, elipsis, listas y encabezados.
- `tests/fixtures/mini_corpus.jsonl`
  - Fixture con `3` documentos multilingües que valida la integración de chunking con entradas reales sintéticas.
- `README_CHUNKING.md`
  - Documentación de uso del módulo, alcance de la fase 3 y notas de operación.
- `pyproject.toml`
  - Dependencias y configuración del proyecto.
- `uv.lock`
  - Lockfile que incluye las dependencias efectivas usadas para validar la ejecución.

## 4. Esquema de entrada

El chunking recibe un JSONL con objetos que contienen estas claves exactas:

- `doc_id`
- `fuente`
- `formato`
- `fenomeno`
- `idioma`
- `texto`

Ejemplo válido:

```json
{
  "doc_id": "doc1",
  "fuente": "fuente",
  "formato": "txt",
  "fenomeno": 1,
  "idioma": "es",
  "texto": "Primera oración. Segunda oración."
}
```

### Validación de `formato` e `idioma`

- Deben ser strings no vacíos.
- No están limitados por listas cerradas.
- Se conservan exactamente desde la Fase 2.
- Valores como `txt`, `parquet` o `und` son aceptados.

## 5. Esquema de salida

Cada chunk resultante produce un JSONL con los campos:

- `doc_id`
- `chunk_id`
- `fuente`
- `formato`
- `fenomeno`
- `idioma`
- `posicion`
- `num_tokens`
- `num_palabras`
- `char_start`
- `char_end`
- `texto`

Ejemplo válido (estructural):

```json
{
  "doc_id": "doc1",
  "chunk_id": "doc1-chunk-0000",
  "fuente": "fuente",
  "formato": "txt",
  "fenomeno": 1,
  "idioma": "es",
  "posicion": 0,
  "num_tokens": 10,
  "num_palabras": 8,
  "char_start": 0,
  "char_end": 45,
  "texto": "Primera oración. Segunda oración."
}
```

### Significado de cada campo

- `doc_id`: identificador original del documento.
- `chunk_id`: identificador secuencial del chunk, construido como `doc_id-chunk-XXXX`.
- `fuente`: valor original de la fuente.
- `formato`: valor original del formato.
- `fenomeno`: valor original del fenómeno.
- `idioma`: valor original del idioma.
- `posicion`: índice del chunk dentro del documento, empezando en `0`.
- `num_tokens`: número de tokens calculados por el tokenizer elegido.
- `num_palabras`: número de palabras en el chunk, calculado con `len(text.split())`.
- `char_start`: offset inclusivo dentro del texto original.
- `char_end`: offset exclusivo dentro del texto original.
- `texto`: slice exacto del texto original correspondiente al chunk.

## 6. Estrategia de chunking

La estrategia se basa en reglas estructurales y en la conservación de oraciones completas.

- Detección de bloques:
  - El texto se divide por separadores de párrafo (`\n\n+`).
- Detección de encabezados:
  - Bloques cortos sin puntuación final se marcan como encabezados.
  - Un encabezado puede fusionarse con el párrafo siguiente cuando ambos caben en el mismo bloque.
- Tratamiento de párrafos:
  - Los párrafos cortos de una misma sección pueden combinarse en un solo chunk si caben.
  - Los párrafos largos se dividen únicamente en límites de oraciones.
- Segmentación por oraciones:
  - El texto se segmenta en oraciones respetando abreviaturas, decimales, iniciales y signos `?` / `!` / `…`.
- Oración como unidad mínima indivisible:
  - No se corta una oración por la mitad.
  - Si una oración supera los límites, se marca como oversize y el documento puede fallar o descartarse.
- Preferencia por fronteras estructurales:
  - Se busca romper en el final de bloques y párrafos cuando sea posible.
- Conservación de encabezados:
  - Los encabezados no se mezclan con contenido de la sección anterior.
- Combinación de párrafos cortos:
  - Si varios párrafos pequeños caben dentro de los límites de token/palabras, se agrupan en el mismo chunk.
- División de párrafos largos:
  - Solo se divide en fronteras oracionales.

## 7. Preservación del texto

Los chunks se construyen directamente con un slice del texto original:

```python
texto_original[char_start:char_end]
```

Aclaraciones:

- `char_start` es inclusivo.
- `char_end` es exclusivo.
- Los offsets se calculan sobre el texto normalizado de Fase 2.
- No se resume.
- No se traduce.
- No se corrige.
- No se reformula.
- No se reconstruye con `join`.

Comprobación garantizada en el código:

```python
chunk["texto"] == documento["texto"][chunk["char_start"]:chunk["char_end"]]
```

## 8. Conteo de tokens y palabras

El módulo usa un `TokenCounter` abstracto para contar tokens.

- `TransformerTokenCounter`
  - Usa `transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)`.
  - Cuenta tokens con `add_special_tokens=True` y `truncation=False`.
- `FakeTokenCounter`
  - Se usa en pruebas y en validación sintética.
- `count_words`
  - Devuelve `len(text.split())`.

Valores provisionales definidos en el código:

- `max_tokens`: `450`.
- `max_words`: `250`.

`max_tokens` debe confirmarse con el encoder definitivo de la fase 4, porque el tokenizer y la codificación del modelo determinan el conteo real.

## 9. Solapamiento

El parámetro `--overlap-sentences` controla el solapamiento entre chunks.

- Solo se solapan oraciones completas.
- Se usa para conservar contexto entre chunks.
- El algoritmo evita chunks idénticos y evita ciclos.
- El solapamiento incorpora contenido nuevo en cada chunk.

Ejemplo sencillo:

- Chunk 0: oraciones `A B`
- Chunk 1: oraciones `B C`

Cada chunk avanza y añade contexto sin repetir exactamente el mismo rango de offsets.

## 10. Manejo de oraciones oversize

Una oración se considera oversize si:

- su número de tokens excede `max_tokens`, o
- su número de palabras excede `max_words`.

Políticas:

- `fail`
  - Falla con excepción y detiene el procesamiento.
- `skip-document`
  - El documento afectado se descarta y se registra en el archivo de errores.

No se hace truncamiento de oraciones porque eso rompería la unidad mínima y la coherencia del chunk.

El archivo de errores guarda registros con:

- `line_number`
- `doc_id`
- `fuente`
- `idioma`
- `char_start`
- `char_end`
- `num_tokens`
- `num_palabras`
- `max_tokens`
- `max_words`
- `motivo`

En `skip-document`, todo el documento se descarta, no solo la oración oversize.

## 11. Escritura atómica

La salida se escribe primero en archivos temporales y luego se publica con `os.replace`.

- Usa `tempfile.NamedTemporaryFile(delete=False)` para controlar los archivos intermedios.
- Al final, reemplaza los archivos de destino atómicamente.
- Si falla el proceso, elimina los temporales.
- Esto protege salidas previas válidas y evita resultados parciales.

## 12. Uso de la CLI

Comando de uso:

```bash
uv run python -m codefest_ad_astra.ingest.chunking \
  --entrada <DOCUMENTOS_JSONL> \
  --salida <CHUNKS_JSONL> \
  --errores <ERRORES_JSONL> \
  --tokenizer-model <ENCODER_MULTILINGUE_ELEGIDO> \
  --max-tokens 450 \
  --max-words 250 \
  --overlap-sentences 1 \
  --on-oversize fail
```

Argumentos:

- `--entrada`: ruta al JSONL de entrada con documentos procesados.
- `--salida`: ruta al JSONL de chunks.
- `--errores`: ruta al JSONL de errores.
- `--tokenizer-model`: modelo del tokenizer para conteo de tokens.
- `--max-tokens`: límite de tokens por chunk.
- `--max-words`: límite de palabras por chunk.
- `--overlap-sentences`: oraciones de solapamiento entre chunks.
- `--on-oversize`: `fail` o `skip-document`.

No se recomienda `all-MiniLM-L6-v2` como encoder de producción en esta documentación; el modelo definitivo debe decidirse en la fase 4.

## 13. Instalación y entorno

Se valida con `uv`:

```bash
uv lock
uv sync
```

Dependencias relevantes:

- `transformers` está en `project.dependencies`.
- `pytest` está en `dependency-groups.dev`.
- No usar `pip` ni `ensurepip` en este flujo.

## 14. Pruebas implementadas

- `tests/test_chunking.py`: `18` pruebas.
- `tests/test_chunking_cli.py`: `15` pruebas.
- `tests/test_sentence_splitter.py`: `9` pruebas.
- Total: `42` pruebas.

### Qué valida cada grupo

- `test_chunking.py`
  - creación de chunks con solapamiento.
  - reporte de oversize y `skip-document`.
  - preservación de `formato`/`idioma` no listados.
  - validación de campos obligatorios y tipos.
  - validación de `fenomeno`.
  - preservación exacta del slice de texto.
  - determinismo de `chunk_id` y `posicion`.
  - combinación de párrafos cortos.
  - preservación de encabezados, listas y viñetas.
  - oversize por tokens y por palabras en modos `fail` y `skip-document`.
  - orden de múltiples documentos.
  - determinismo byte a byte de la salida.
  - limpieza de temporales en fallos.

- `test_chunking_cli.py`
  - ayuda CLI.
  - argumentos faltantes.
  - ejecución válida y creación de archivos.
  - JSON inválido en entrada.
  - doc_id duplicado.
  - error de tokenizer.
  - `on-oversize` con `fail` y `skip-document`.
  - configuración inválida de CLI con código de salida `1`.
  - directorios de salida creados.
  - conservar salida anterior válida en fallos.

- `test_sentence_splitter.py`
  - abreviaturas.
  - decimales.
  - interrogación/exclamación en español.
  - iniciales y siglas en inglés.
  - `Dr.` en portugués.
  - prefijos de letra única.
  - elipsis ASCII y Unicode.
  - listas numeradas y viñetas.
  - encabezados separados.

## 15. Cómo comprobar que el chunking está bien

Validación manual recomendada:

- Ninguna oración debe cortarse en medio.
- `chunk["texto"]` debe coincidir con `doc["texto"][char_start:char_end]`.
- `num_tokens` no debe exceder `max_tokens`.
- `num_palabras` no debe exceder `250`.
- `posicion` debe comenzar en `0` y avanzar secuencialmente.
- `chunk_id` debe corresponder a la posición (`doc_id-chunk-000X`).
- El orden de documentos en la salida debe respetar el orden de entrada.
- Los offsets (`char_start`, `char_end`) deben ser únicos consecutivos y no generar chunks idénticos.
- Encabezados, listas y viñetas deben conservarse en el chunk.
- `fuente`, `formato`, `fenomeno` e `idioma` deben conservarse tal cual.

Ejemplo correcto:

- `doc_id`: `doc1`
- `chunk_id`: `doc1-chunk-0000`
- `posicion`: `0`
- `texto`: slice exacto de la entrada

Ejemplo incorrecto:

- `chunk_id` no refleja `posicion`
- `texto` reconstruido desde `join` y no coincide con offsets
- `formato` o `idioma` cambiados
- `num_tokens` mayor que el límite

## 16. Comandos de validación

Para validar se usan estos comandos:

```bash
uv run python -m py_compile src/codefest_ad_astra/ingest/chunking.py
uv run pytest -v tests
uv run pytest --collect-only -q tests
uv run python -m codefest_ad_astra.ingest.chunking --help
wc -l tests/fixtures/mini_corpus.jsonl
```

Resultados actuales esperados:

- `py_compile` sin errores.
- `42` pruebas recolectadas.
- `42` pruebas aprobadas.
- fixture con `3` líneas.
- ayuda CLI disponible.

## 17. Validación manual recomendada

Para datos reales, revisar `10-20` chunks de distintos:

- documentos
- formatos
- fenómenos
- idiomas

Checklist:

- `chunk_id` correcto
- `posicion` secuencial
- offsets correctos
- texto coincidente
- no se perdió contenido
- errores documentados
- embeddings pendientes

## 18. Pendientes antes de cerrar completamente la Fase 3

- Escoger encoder multilingüe.
- Confirmar el `tokenizer-model` definitivo.
- Confirmar `max_tokens` con el encoder elegido.
- Generar documentos reales con Fases 1 y 2.
- Ejecutar chunking sobre una muestra real.
- Revisar errores reales.
- Inspeccionar chunks reales.
- Registrar la configuración final.

## 19. Contrato para la Fase 4

La fase 4 debe:

- leer el JSONL de chunks en orden.
- generar un embedding por línea.
- no reordenar registros.
- conservar toda la metadata.
- guardar `metadata.jsonl` en el mismo orden de inserción en FAISS.
- usar el mismo encoder cuyo tokenizer contó los tokens.
- validar que ningún chunk exceda el límite del encoder.
- no resumir ni modificar texto.
- no cambiar `chunk_id` ni `doc_id`.

Relación forzada:

- línea 0 de `chunks` → embedding 0 → ID FAISS 0 → línea 0 de `metadata.jsonl`

## 20. Riesgos y limitaciones

- Segmentación basada en reglas y heurísticas.
- Se necesita prueba con corpus real.
- Documentos sin puntuación pueden segmentarse mal.
- Tablas o listas muy largas pueden generar chunks menos naturales.
- Oraciones oversize individuales pueden descartar documentos enteros.
- Depende de elegir correctamente el tokenizer para la fase 4.
- El módulo puede necesitar dividirse en componentes si crece.

## 21. Checklist para iniciar la Fase 4

- [ ] Encoder seleccionado
- [ ] Tokenizer confirmado
- [ ] Documentos reales generados
- [ ] Chunks reales generados
- [ ] Archivo de errores revisado
- [ ] Pruebas aprobadas
- [ ] Muestra manual aprobada
- [ ] Ruta final de chunks confirmada
- [ ] Límites documentados
- [ ] Orden de metadata protegido

## 22. Historial de validación

- Python `3.14.6`
- pytest `9.1.1`
- `42` pruebas
- `42` aprobadas
- Fecha de documentación: 2026-08-02
- Estado: aprobado con datos sintéticos, pendiente integración real

### Marcadores pendientes

- `<ENCODER_MULTILINGUE_PENDIENTE>`
- `<RUTA_DOCUMENTOS_REALES_PENDIENTE>`
- `<RUTA_CHUNKS_REALES_PENDIENTE>`
