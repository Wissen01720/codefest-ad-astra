# CodeFest Ad Astra 2026 — Etapa 1: Base de Conocimiento Vectorial

Proyecto para la etapa virtual del CODEFEST AD ASTRA 2026, organizado por la
Universidad de los Andes y la Fuerza Aeroespacial Colombiana. La solución
implementa un pipeline multiformato y multilingüe que extrae, limpia y fragmenta
el corpus, genera embeddings, construye una base FAISS y responde las 50
consultas del reto sin utilizar modelos generativos.

El corpus cubre tres fenómenos:

1. Inteligencia artificial e innovación en entornos militares.
2. Seguridad espacial y órbita baja terrestre (LEO).
3. Dinámicas territoriales en América Latina y el Caribe.

## Estado de la entrega

El flujo obligatorio está implementado y validado de extremo a extremo.

| Componente | Estado |
| --- | --- |
| `entrega/resultados.jsonl` | 50 consultas, reproducible byte a byte |
| `entrega/generador.py` | Autónomo respecto de `src/` |
| `entrega/informe_tecnico.pdf` | Informe técnico A4 de 7 páginas |
| Base vectorial | 202.350 vectores, 1.024 dimensiones, validada |
| Grafo de conocimiento | No implementado; componente opcional |

La base contiene fragmentos de 1.734 documentos. `resultados.jsonl` devuelve,
para cada consulta entre `q001` y `q050`, un ranking de 3 documentos y un
ranking global de 10 fragmentos de hasta 250 palabras.

## Cobertura del corpus

El pipeline de extracción procesó correctamente 1.754 de 1.835 recursos
(95,6 %). Los 81 restantes quedaron clasificados mediante diagnósticos como
manifiestos o catálogos, documentos escaneados sin texto extraíble, archivos
corruptos, imágenes sin texto o contenido genuinamente corto.

Se admiten los formatos proporcionados para el reto: PDF, HTML, JSON, CSV,
XLSX, TXT, Markdown, imágenes y PBF.

## Estructura del proyecto

```text
data/           Consultas y diagnósticos del corpus
docs/           Documentación de diseño
entrega/        Artefactos finales para CodeFest
scripts/        Validación integral de la entrega
src/            Código fuente del pipeline y la recuperación
tests/          Pruebas automatizadas
README.md       Descripción y operación del proyecto
pyproject.toml  Dependencias y configuración de Python
uv.lock         Resolución reproducible de dependencias
```

### Módulos principales

| Ruta | Responsabilidad |
| --- | --- |
| `src/codefest_ad_astra/ingest/` | Extracción, limpieza, diagnósticos y chunking |
| `src/codefest_ad_astra/indexing/` | Encoder, construcción, persistencia, búsqueda y verificación de FAISS |
| `src/codefest_ad_astra/retrieval/` | Agregación documental y generación de resultados |
| `scripts/validar_entrega.py` | Validación de estructura, resultados, PDF, generador y base vectorial |

## Contenido obligatorio de entrega

```text
entrega/
├── resultados.jsonl
├── generador.py
├── informe_tecnico.pdf
└── base_vectorial/
    └── encoder_bge-m3/
        ├── index.faiss
        ├── metadata.jsonl
        └── manifest.json
```

La carpeta `entrega/base_vectorial/` debe existir localmente y formar parte del
paquete enviado a CodeFest. La base vectorial no se sube al repositorio Git
debido a su tamaño.

El repositorio Git sí contiene `resultados.jsonl`, `generador.py`,
`informe_tecnico.pdf`, el código fuente, las pruebas y la configuración del
proyecto.

## Diseño técnico

- Extracción específica por formato y representación contextual de datos
  tabulares y geográficos.
- Limpieza Unicode/UTF-8, detección de idioma, remoción de boilerplate y
  trazabilidad mediante metadata.
- Identificadores determinísticos para documentos y fragmentos.
- Chunking estructural-oracional con un máximo de 450 tokens, 250 palabras y
  una oración de solapamiento.
- Conservación de oraciones completas en prosa; los fallbacks internos se
  reservan para datos estructurados cuando corresponde.
- Encoder multilingüe `BAAI/bge-m3`, sin decoders ni modelos generativos.
- Embeddings normalizados de 1.024 dimensiones.
- Índice exacto FAISS `IndexFlatIP`; el producto interno sobre vectores
  unitarios equivale a similitud coseno.
- Ranking de 3 documentos mediante suma de scores de sus fragmentos.
- Ranking global e independiente de 10 fragmentos.
- Grafo de conocimiento no implementado por ser opcional.

Los ocho campos obligatorios de cada registro de metadata son `doc_id`,
`chunk_id`, `fuente`, `formato`, `fenomeno`, `posicion`, `num_tokens` y
`texto`. El orden de `metadata.jsonl` coincide con los identificadores internos
del índice FAISS.

## Punto de entrada de extracción

```python
from codefest_ad_astra.ingest.extraer_y_limpiar import extraer_y_limpiar

resultado = extraer_y_limpiar(path)

resultado.texto
resultado.idioma
resultado.paginas_originales
resultado.lineas_boilerplate_removidas
```

## Instalación

Requiere Python 3.12 o posterior y `uv`.

```bash
uv sync --python 3.12
```

## Flujo de construcción

Las salidas intermedias se escriben bajo `data/processed/`, que no se versiona
en Git.

```bash
uv run python -m codefest_ad_astra.ingest.pipeline \
  --corpus data/raw \
  --salida data/processed/documentos.jsonl

uv run python -m codefest_ad_astra.ingest.chunking \
  --entrada data/processed/documentos.jsonl \
  --salida data/processed/chunks.jsonl \
  --errores data/processed/errores_chunking.jsonl \
  --tokenizer-model BAAI/bge-m3 \
  --max-tokens 450 \
  --max-words 250

uv run python -m codefest_ad_astra.indexing.build_index \
  --entrada data/processed/chunks.jsonl \
  --salida entrega/base_vectorial \
  --modelo BAAI/bge-m3
```

El tokenizer empleado durante el chunking debe corresponder al encoder usado
para construir el índice.

## Validación

```bash
uv run pytest -q

uv run python -m codefest_ad_astra.indexing.verificar \
  --base entrega/base_vectorial/encoder_bge-m3

uv run python scripts/validar_entrega.py --ejecutar-generador
```

Resultados esperados:

- 60 pruebas aprobadas.
- Base con 202.350 vectores, dimensión 1.024 y metadata alineada.
- Vectores normalizados y campos obligatorios completos.
- Reproducción de `resultados.jsonl` aprobada byte a byte.

## Entrega final

El paquete que se envía a CodeFest se genera desde la raíz del proyecto:

```bash
tar -czf codefest-ad-astra-entrega.tar.gz entrega/
```

El archivo resultante debe incluir la base vectorial completa dentro de
`entrega/base_vectorial/`.

## GitHub

- No añadir `entrega/base_vectorial/` al repositorio Git.
- No añadir archivos `.tar.gz`, `.venv/`, cachés, temporales, `__pycache__/` ni
  `.pytest_cache/`.
- Antes del commit, ejecutar `git status` y comprobar que la base vectorial
  permanece ignorada.
