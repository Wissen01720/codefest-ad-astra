# Fase 4 — Embeddings e índice FAISS

Convierte el `chunks.jsonl` de la Fase 3 en la base vectorial entregable:
`index.faiss` + `metadata.jsonl` alineados, más un manifiesto con las decisiones
de diseño.

Paquete: `src/codefest_ad_astra/indexing/`

---

## 1. Uso rápido

### Construir el índice

```bash
uv run python -m codefest_ad_astra.indexing.build_index \
  --entrada data/processed/chunks.jsonl \
  --salida entrega/base_vectorial \
  --modelo BAAI/bge-m3
```

Escribe en `entrega/base_vectorial/encoder_bge-m3/`:

| Archivo | Qué es |
|---|---|
| `index.faiss` | Índice `IndexFlatIP`, serializado con `faiss.write_index()` |
| `metadata.jsonl` | Un objeto JSON por línea; la línea *i* describe el vector *i* del índice |
| `manifest.json` | Modelo, dimensión, prefijos, tipo de índice, sha256 de la entrada, versiones |
| `descartados.jsonl` | Fragmentos omitidos por texto vacío (solo si hubo alguno) |

### Verificar la base antes de entregar

```bash
uv run python -m codefest_ad_astra.indexing.verificar \
  --base entrega/base_vectorial/encoder_bge-m3
```

Falla con código 1 si el índice y la metadata están desalineados, si hay
vectores sin normalizar, si falta un campo de la Tabla 1 o si el índice se
construyó con `--limite` (parcial). Además reporta cobertura por fenómeno,
idioma y formato — números directamente reutilizables en el informe técnico.

### Consultar (verificación manual)

```bash
uv run python -m codefest_ad_astra.indexing.search \
  --base entrega/base_vectorial/encoder_bge-m3 \
  --consulta "riesgos de colisión en la órbita baja terrestre" -k 10
```

Desde Python:

```python
from codefest_ad_astra.indexing.search import Buscador

buscador = Buscador("entrega/base_vectorial/encoder_bge-m3")
for r in buscador.buscar("adopción de IA en el sector defensa", k=10):
    print(r.rank, round(r.score, 4), r.doc_id, r.texto[:80])
```

### Opciones útiles de `build_index`

| Flag | Para qué |
|---|---|
| `--limite N` | Indexa solo los primeros N fragmentos. Marca el índice como parcial en el manifiesto (y `verificar` lo rechaza) |
| `--device cuda` | Fuerza GPU. Por defecto, automático |
| `--batch-size` | Lote del encoder (32 por defecto; en GPU sube a 64–128) |
| `--tamano-bloque` | Fragmentos por checkpoint en disco (2048 por defecto) |
| `--reanudar` | Continúa una corrida interrumpida desde el último bloque |
| `--conservar-parciales` | No borra `_parciales/` al terminar |
| `--modelo fake` | Encoder determinístico sin descargar pesos, para probar el flujo |

---

## 2. Decisiones de diseño (material para el informe técnico)

### Encoder: `BAAI/bge-m3`

| Criterio (Sección 4.3 de la especificación) | Cómo lo cumple |
|---|---|
| Soporte multilingüe | Entrenado nativamente en ES/EN/PT y ~100 idiomas más. El corpus mezcla los tres y una consulta en español debe recuperar documentos en inglés |
| Dimensionalidad | 1024 — suficiente expresividad sin inflar el índice |
| Longitud máxima de entrada | 8192 tokens; los chunks de la Fase 3 llegan acotados a 450, así que ninguno se trunca |
| Rendimiento en benchmarks | Fuerte en recuperación densa multilingüe (MTEB/MIRACL) |
| Licencia | MIT |
| Eficiencia | ~2.2 GB, corre en CPU pero pide GPU para el corpus completo (ver Sección 4) |

**Sin prefijos de instrucción.** bge-m3 no los usa. Sí los necesitan los modelos
de la familia E5 (`query: ` / `passage: `), y omitirlos degrada la recuperación
de forma notable. Por eso `encoders.prefijos_para_modelo()` centraliza esa
decisión y el prefijo efectivo **se guarda en `manifest.json`**: al consultar,
`Buscador` lee el manifiesto y usa exactamente el mismo, en lugar de confiar en
que quien busca recuerde configurarlo.

### Índice: `IndexFlatIP` con vectores normalizados

Plano y exacto, no aproximado:

- Con norma unitaria, el producto interno **es** la similitud coseno (ecuación 4
  de la especificación), que es la métrica que pide el reto.
- El corpus (decenas de miles de fragmentos) no justifica IVF/HNSW.
- Un índice aproximado introduce variabilidad entre corridas y el entregable 4
  exige que `generador.py` **reproduzca** los resultados. Si no se reproducen,
  la entrega se excluye de la evaluación.

La normalización se aplica dos veces a propósito: el encoder devuelve vectores
ya normalizados y `normalizar()` los reajusta antes de indexar (idempotente,
protege contra acumulación numérica en fp16). `construir_indice` **rechaza**
vectores cuya norma se desvíe más de 1e-3 de 1.0 en lugar de indexarlos en
silencio: un índice con vectores sin normalizar devuelve rankings mal ordenados
y nada en la evaluación lo delataría.

### La invariante que sostiene toda la entrega

> La línea *i* de `metadata.jsonl` describe el vector con identificador interno
> *i* en `index.faiss`.

Si se rompe, el sistema devuelve textos que no corresponden a los vectores
recuperados: NDCG y F1 se desploman sin ningún error visible. El código la
protege en cuatro puntos:

1. `validar_alineacion()` se ejecuta **antes** de escribir y **al** cargar.
2. `guardar_base_vectorial()` escribe la metadata primero y el índice después;
   si la metadata falla, no queda un índice nuevo junto a metadata vieja.
3. La metadata se escribe de forma atómica (temporal + `os.replace`), así que
   una corrida interrumpida no deja un archivo a medias.
4. El filtro de fragmentos vacíos es determinístico sobre el archivo de entrada,
   de modo que reanudar una corrida reproduce el mismo orden exacto.

### Metadata

Se escriben los ocho campos obligatorios de la Tabla 1 (`doc_id`, `chunk_id`,
`fuente`, `formato`, `fenomeno`, `posicion`, `num_tokens`, `texto`) **primero y
en ese orden**, y a continuación los campos extra que trae la Fase 3 (`idioma`,
`num_palabras`, `char_start`, `char_end`). La especificación permite añadir
campos; los extra dan trazabilidad al texto original y sirven de post-filtro en
la Fase 6.

Si a un fragmento le falta un campo obligatorio, la construcción **aborta** con
el número de línea. Es preferible a descubrirlo cuando el evaluador rechace la
entrega.

### Detección de truncamiento silencioso

Un encoder que recibe más tokens de los que admite **no falla: trunca**. El
vector representaría solo el inicio del fragmento mientras `metadata.jsonl`
sigue guardando el texto completo — un fallo invisible que degrada el ranking.

El riesgo es concreto: la Fase 3 cuenta tokens con el tokenizer que se le pase
por `--tokenizer-model`, y el ejemplo de `README_CHUNKING.md` usa
`all-MiniLM-L6-v2`, **que no es el tokenizer de bge-m3**. Dos tokenizadores
distintos dan conteos distintos sobre el mismo texto.

Antes de codificar, `build_index` tokeniza los 200 fragmentos más largos con el
tokenizer *del encoder de indexación* y avisa si alguno excede
`max_seq_length`, indicando con qué `--tokenizer-model` habría que rehacer la
Fase 3. El conteo queda en el manifiesto como
`fragmentos_truncados_detectados`.

> **Al correr la Fase 3 para producción, usar `--tokenizer-model BAAI/bge-m3`.**

### Checkpointing

Codificar el corpus completo con bge-m3 puede tomar horas. La codificación va
por bloques de 2048 fragmentos y cada bloque se guarda en `_parciales/`. Si la
corrida se cae, `--reanudar` continúa desde el último bloque.

El checkpoint guarda una firma (sha256 de la entrada + modelo + dimensión +
número de fragmentos) y **se niega a reanudar** si algo de eso cambió. Sin esa
verificación, reanudar sobre un `chunks.jsonl` distinto mezclaría vectores de
dos corpus y produciría exactamente la desalineación silenciosa descrita arriba.

---

## 3. Qué recibe la Fase 6 (recuperación)

`Buscador.buscar(consulta, k)` devuelve una lista de `Resultado` ordenada de
mayor a menor score, donde `score` es la similitud coseno y `metadata` es el
registro completo del fragmento.

```python
Resultado(rank=1, score=0.8123, metadata={"doc_id": ..., "chunk_id": ..., "texto": ...})
```

También existe `buscar_lote(consultas, k)`, que codifica todas las consultas en
una sola pasada del encoder — útil para las 50 consultas de la evaluación.

**Lo que la Fase 4 deliberadamente NO hace** (es trabajo de las Fases 6 y 7):

- Agregación de fragmentos a nivel de documento (max pooling / suma) y top-3.
- Fusión de varios encoders (RRF, CombSUM, CombMNZ).
- Post-filtros por metadata o por umbral de similitud.
- Subdividir fragmentos a ≤250 palabras y escribir `resultados.jsonl`.
- `generador.py`.

`verificar` sí informa cuántos fragmentos superan las 250 palabras, porque esos
son los que la Fase 7 tendrá que subdividir respetando oraciones completas.

---

## 4. Dónde correr la generación de embeddings

**Recomendación: Kaggle con GPU.** No es una preferencia, es una medición hecha
en esta máquina (sin GPU NVIDIA, `torch` es la build `+cpu`):

| Medición | Valor |
|---|---|
| bge-m3, fragmentos de ~512 tokens, CPU, batch 8 | **0.83 fragmentos/s** |
| Extrapolación a 50.000 fragmentos | **~17 horas** |

Con GPU el ritmo sube uno o dos órdenes de magnitud, así que la misma corrida
baja a menos de una hora. El módulo es portable: la misma CLI corre en ambos
lados, solo cambia `--device` y `--batch-size`.

Si aun así toca hacerlo en CPU, el checkpointing está pensado exactamente para
eso: se puede cortar la corrida y retomarla con `--reanudar`.

Para el desarrollo local, `--limite` y `--modelo fake` permiten ejercitar el
flujo completo en segundos.

Receta en un notebook de Kaggle (GPU activada, internet activado):

```python
!pip -q install faiss-cpu sentence-transformers

# El repo y el chunks.jsonl pueden subirse como Dataset de Kaggle
import sys; sys.path.insert(0, "/kaggle/input/codefest-repo/src")

!python -m codefest_ad_astra.indexing.build_index \
  --entrada /kaggle/input/codefest-chunks/chunks.jsonl \
  --salida /kaggle/working/base_vectorial \
  --modelo BAAI/bge-m3 --device cuda --batch-size 128
```

Luego se descarga `/kaggle/working/base_vectorial/` y se coloca bajo
`entrega/base_vectorial/`. `faiss-cpu` sirve perfectamente para *construir* el
índice aunque los embeddings se generen en GPU: `IndexFlatIP` no requiere GPU.

---

## 5. Tests

```bash
uv run python -m pytest tests/test_indexing_*.py -q
```

82 tests propios de esta fase (75 unitarios + 7 de integración), repartidos así:

| Archivo | Cubre |
|---|---|
| `test_indexing_encoders.py` | Slugs de modelo, prefijos por familia, normalización (incluye vector nulo), determinismo del encoder de pruebas |
| `test_indexing_faiss_store.py` | Construcción del índice, rechazo de vectores sin normalizar, **alineación id FAISS ↔ línea de metadata**, escritura atómica, UTF-8 sin escapes, lectura con `faiss.read_index()` estándar |
| `test_indexing_build.py` | Orden preservado, descarte de textos vacíos, campos obligatorios, manifiesto, `--limite`, detección de truncamiento, checkpoint/reanudación (incluye rechazo de checkpoint ajeno), CLI como módulo |
| `test_indexing_search.py` | Top-k, orden de scores, recuperación exacta del fragmento idéntico, k mayor que el índice, coherencia lote vs. individual, CLI |
| `test_indexing_verificar.py` | Detección de desalineación, índice parcial, chunk_id duplicados, fragmentos >250 palabras |
| `test_indexing_integracion_encoder_real.py` | Encoder real de `sentence-transformers`: normalización, reproducibilidad, recuperación semántica de punta a punta |

Los tests unitarios usan un `FakeEncoder` determinístico (hash estable con
`blake2b`, no `hash()`, que está aleatorizado por proceso) — corren en segundos
y sin descargar pesos. Los de integración usan `paraphrase-MiniLM-L3-v2` (68 MB)
y se saltan solos si no hay modelo en cache ni red.

El test del encoder de producción está detrás de una variable de entorno porque
descarga 2.2 GB:

```bash
CODEFEST_TEST_BGE_M3=1 uv run python -m pytest tests/test_indexing_integracion_encoder_real.py -q
```

Ese test ya se ejecutó con los pesos reales: bge-m3 carga, produce 1024
dimensiones y las paráfrasis ES/EN/PT de la misma frase quedan a similitud
>0.7 entre sí — el cruce de idiomas que el reto necesita.

### Prueba de humo sin corpus real

Con cualquier `chunks.jsonl` de juguete (mismo esquema que la Fase 3) se puede
ejercitar la cadena completa en segundos:

```bash
uv run python -m codefest_ad_astra.indexing.build_index \
  --entrada mis_chunks.jsonl --salida /tmp/bv --modelo fake
uv run python -m codefest_ad_astra.indexing.verificar --base /tmp/bv/encoder_encoder-test
uv run python -m codefest_ad_astra.indexing.search \
  --base /tmp/bv/encoder_encoder-test --consulta "prueba" -k 3
```

Ya se hizo con un corpus sintético de 9 fragmentos (3 fenómenos × 3 idiomas, en
pdf/html/json/csv) usando encoders reales: una consulta en español recuperó
primero el fragmento correcto en español y, segundo, su equivalente en
portugués.

---

## 6. Cambios al entorno del repo

- **`.venv` recreado con Python 3.12.** Estaba apuntando a un intérprete 3.14.6
  inexistente y ningún comando corría. Además `torch` no publica wheels para
  3.14. `.python-version` sigue diciendo `3.14`: **conviene bajarlo a `3.12`**
  para que `uv sync` no vuelva a romper el entorno de quien clone el repo. No lo
  cambié porque afecta a todo el equipo.
- **`pyproject.toml`**: se registró el marker `integracion` de pytest. Sin eso,
  pytest emite warnings por marker desconocido.
- **`.gitignore` ya excluye `*.faiss`**, así que el índice **no viaja por git**.
  Hay que moverlo a mano (o por Git LFS) hacia quien arme el paquete final de
  entrega. `metadata.jsonl` y `manifest.json` sí se versionan, y como el
  manifiesto lleva el `sha256` del `chunks.jsonl` de origen, siempre se puede
  comprobar si un índice suelto corresponde a los chunks vigentes.

Versiones con las que se validó: Python 3.12.3, `faiss-cpu` 1.14.3, `numpy`
2.5.1, `torch` 2.13.0+cpu, `sentence-transformers` 5.6.1.

---

## 7. Estado y pendientes

### Hecho y verificado

- Wrapper de encoders con normalización y prefijos por familia de modelo.
- Construcción, persistencia y carga de la base vectorial con la invariante de
  alineación protegida y verificable.
- CLI de construcción con checkpointing y reanudación.
- CLI de verificación de la base entregable.
- Búsqueda mínima (top-k por similitud coseno), individual y por lote.
- Suite de tests propia, verde.

### Pendiente / bloqueado

1. **No se ha construido el índice sobre el corpus real.** En esta máquina no
   existen `data/raw` ni `data/processed/chunks.jsonl` (están en `.gitignore`).
   El módulo está probado de punta a punta con fixtures y con un encoder real,
   pero necesita el `chunks.jsonl` de la Fase 3 para producir la base definitiva.
   **Es lo primero que hay que correr cuando llegue ese archivo.**
2. **Advertencia heredada de la Fase 3** (`README_CHUNKING.md`): tras el fix de
   fallback estructural, los documentos CSV/XLSX/PBF que antes se descartaban
   ahora generan chunks. Cualquier índice construido antes de ese fix debe
   regenerarse. Como aún no se construyó ninguno, esto se cumple por defecto —
   solo hay que asegurarse de usar el `chunks.jsonl` posterior al fix (el
   `sha256_entrada` del manifiesto permite comprobar de cuál se partió).
3. **Correr la Fase 3 con el tokenizer de bge-m3.** El `chunks.jsonl` definitivo
   debe generarse con `--tokenizer-model BAAI/bge-m3` (ver la sección de
   truncamiento). Si el que llegue viene con otro tokenizer, `build_index`
   avisará, pero lo correcto es rehacer el chunking.
4. **Campo `formato`.** `pipeline.py` (Fase 1) emite `texto` e `imagen` para TXT
   e imágenes; la Tabla 1 describe el campo con los ejemplos `pdf`, `html`, `md`.
   La Fase 4 conserva el valor tal cual llega, sin reinterpretarlo. Si el equipo
   decide normalizarlo, el cambio va en la Fase 1 y hay que **reindexar**.
5. **Segundo encoder (opcional).** La estructura ya lo soporta: basta correr
   `build_index` con otro `--modelo` y aparece una segunda carpeta
   `encoder_<slug>/`. La fusión de rankings es trabajo de la Fase 6.

### Dos tests de la Fase 3 fallan en Windows (pre-existentes, no tocados)

- `tests/test_chunking_cli.py::test_cli_help_returns_zero` — invoca
  `.venv/bin/python`, que es la ruta POSIX; en Windows el intérprete está en
  `.venv/Scripts/python.exe`. El arreglo es usar `sys.executable`.
- `tests/test_chunking.py::test_process_chunking_failure_removes_temporary_files`
  — depende de la semántica de archivos temporales de POSIX.

Ninguno es de esta fase y ninguno indica un problema real del chunking; ambos
son del arnés de pruebas. Los dejo señalados para quien mantiene la Fase 3.
