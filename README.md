# CodeFest Ad Astra 2026 - Etapa 1

Pipeline multiformato y multilingüe para construir una base de conocimiento,
indexarla en FAISS y responder las 50 consultas del reto sin utilizar modelos
generativos.

## Estado de la entrega

| Componente | Estado |
| --- | --- |
| `entrega/resultados.jsonl` | 50 consultas, reproducible byte a byte |
| `entrega/generador.py` | Autónomo respecto de `src/` |
| `entrega/informe_tecnico.pdf` | Informe técnico A4 de 7 páginas |
| Base vectorial | 202.350 vectores, 1.024 dimensiones, validada |
| Grafo de conocimiento | No implementado; componente opcional |

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

- Procesamiento de fuentes PDF, HTML, JSON, CSV, XLSX, TXT, Markdown, imágenes
  y PBF.
- Limpieza, normalización y trazabilidad mediante metadata.
- Chunking con un máximo de 450 tokens y 250 palabras.
- Conservación de oraciones completas en fragmentos de prosa.
- Fallbacks de división reservados para datos estructurados cuando corresponde.
- Encoder multilingüe `BAAI/bge-m3`.
- Embeddings de 1.024 dimensiones.
- Índice FAISS `IndexFlatIP`.
- Vectores normalizados para que el producto interno equivalga a similitud
  coseno.
- Ranking de 3 documentos mediante agregación de scores de fragmentos.
- Ranking global de 10 fragmentos.
- Recuperación sin modelos generativos.
- Grafo de conocimiento no implementado por ser opcional.

## Instalación

Requiere Python 3.12 o posterior y `uv`.

```bash
uv sync --python 3.12
```

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
