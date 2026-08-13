# CodeFest Ad Astra 2026 — Etapa 1

## Base de conocimiento vectorial multilingüe

Este repositorio contiene la solución para la etapa virtual de **CodeFest Ad Astra 2026**, organizada por la Universidad de los Andes y la Fuerza Aeroespacial Colombiana.

La solución construye y consulta una base de conocimiento vectorial a partir de un corpus multiformato sobre tres fenómenos:

1. Inteligencia artificial e innovación en entornos militares.
2. Seguridad espacial y órbita baja terrestre (LEO).
3. Dinámicas territoriales en América Latina y el Caribe.

El sistema procesa documentos, genera fragmentos trazables, calcula embeddings multilingües, construye un índice FAISS y responde las 50 consultas definidas por el reto. No utiliza modelos generativos para producir las respuestas.

## Resultado de la entrega

La entrega implementa el flujo completo solicitado para recuperación vectorial:

| Componente            | Resultado                                     |
| --------------------- | --------------------------------------------- |
| Consultas             | 50 consultas, de `q001` a `q050`              |
| Resultados            | 3 documentos y 10 fragmentos por consulta     |
| Base vectorial        | 202.350 vectores de 1.024 dimensiones         |
| Encoder               | `BAAI/bge-m3`                                 |
| Índice                | FAISS `IndexFlatIP` con vectores normalizados |
| Reproducibilidad      | `resultados.jsonl` se reproduce byte a byte   |
| Informe técnico       | PDF A4 de 7 páginas                           |
| Grafo de conocimiento | No implementado; componente opcional          |

Cada fragmento entregado tiene un máximo de 250 palabras y conserva trazabilidad hacia su documento de origen mediante la metadata del índice.

## Arquitectura

```text
Corpus multiformato
        │
        ▼
Extracción y limpieza
        │
        ▼
Chunking estructural y oracional
        │
        ▼
Embeddings con BAAI/bge-m3
        │
        ▼
Índice FAISS + metadata trazable
        │
        ▼
Recuperación de documentos y fragmentos
        │
        ▼
resultados.jsonl
```

El pipeline admite documentos PDF, HTML, JSON, CSV, XLSX, TXT, Markdown, imágenes con OCR y archivos PBF.

## Estructura del repositorio

```text
data/
├── consultas.jsonl              # Las 50 consultas del reto
└── diagnostics/                 # Diagnósticos y evidencia del procesamiento

docs/                            # Notas técnicas y de diseño

entrega/
├── resultados.jsonl             # Resultado final de las 50 consultas
├── generador.py                 # Generador autónomo y reproducible
├── informe_tecnico.pdf          # Informe técnico de la solución
└── base_vectorial/              # Base requerida para ejecutar la entrega
    └── encoder_bge-m3/
        ├── index.faiss
        ├── metadata.jsonl
        └── manifest.json

scripts/
└── validar_entrega.py           # Validación integral de la entrega

src/codefest_ad_astra/
├── ingest/                      # Extracción, limpieza, diagnóstico y chunking
├── indexing/                    # Embeddings, FAISS, búsqueda y verificación
└── retrieval/                   # Recuperación y generación de resultados

tests/                           # Pruebas automatizadas
pyproject.toml                   # Dependencias y configuración del proyecto
uv.lock                          # Versiones reproducibles de dependencias
```

## Diseño técnico

La solución se apoya en las siguientes decisiones:

* Extracción específica según el formato de cada recurso.
* Limpieza de texto, normalización Unicode y detección de idioma.
* Identificadores determinísticos para documentos y fragmentos.
* Chunking estructural y por oraciones, con límite de 450 tokens y 250 palabras.
* Preservación de oraciones completas en texto narrativo.
* Embeddings multilingües con `BAAI/bge-m3`.
* Vectores normalizados de 1.024 dimensiones.
* Índice FAISS `IndexFlatIP`, equivalente a similitud coseno sobre vectores unitarios.
* Ranking independiente de documentos y fragmentos:

  * top-3 documentos, agregando los puntajes de sus fragmentos;
  * top-10 fragmentos globales por consulta.
* Metadata alineada con el índice para mantener la trazabilidad de cada resultado.

La metadata conserva, entre otros, los campos obligatorios `doc_id`, `chunk_id`, `fuente`, `formato`, `fenomeno`, `posicion`, `num_tokens` y `texto`.

## Requisitos

* Python 3.12 o superior.
* [`uv`](https://docs.astral.sh/uv/).
* La carpeta `entrega/base_vectorial/encoder_bge-m3/` para ejecutar la validación completa.
* Acceso a Hugging Face o una copia local en caché de `BAAI/bge-m3` la primera vez que se ejecuta el generador.

## Instalación

Desde la raíz del repositorio:

```bash
uv sync --python 3.12
```

## Ejecución

El generador autónomo reconstruye el archivo de resultados a partir de la base vectorial y las consultas:

```bash
uv run python entrega/generador.py \
  --base entrega/base_vectorial/encoder_bge-m3 \
  --consultas data/consultas.jsonl \
  --salida /tmp/resultados.jsonl
```

Para comprobar que el resultado generado coincide exactamente con el entregado:

```bash
sha256sum /tmp/resultados.jsonl entrega/resultados.jsonl
```

Ambos hashes deben ser iguales.

## Validación

Ejecutar la suite de pruebas:

```bash
uv run pytest -q
```

Validar la estructura y los artefactos de entrega:

```bash
uv run python scripts/validar_entrega.py
```

Ejecutar además el generador y comprobar la reproducción byte a byte:

```bash
uv run python scripts/validar_entrega.py --ejecutar-generador
```

Esta última validación comprueba, entre otros aspectos:

* estructura de `entrega/`;
* formato y cobertura de `resultados.jsonl`;
* autonomía y sintaxis de `generador.py`;
* presencia y secciones del informe técnico;
* dimensiones, tipo y consistencia de la base FAISS;
* alineación entre índice y metadata;
* trazabilidad de los fragmentos recuperados;
* reproducción exacta de los resultados.

## Contenido de `entrega/`

La carpeta `entrega/` es el artefacto preparado para la evaluación de CodeFest:

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

La base vectorial ocupa aproximadamente 1,1 GB y no se versiona en Git para evitar hacer el repositorio innecesariamente pesado. Sin embargo, debe incluirse dentro del paquete final enviado a CodeFest, ya que es necesaria para reproducir los resultados.

## Uso en GitHub

El repositorio conserva el código fuente, las pruebas, el generador, los resultados y el informe técnico. No deben añadirse al control de versiones los siguientes elementos:

* `entrega/base_vectorial/`;
* archivos comprimidos de entrega;
* entornos virtuales como `.venv/`;
* cachés como `.pytest_cache/` y `__pycache__/`;
* archivos temporales o resultados intermedios no requeridos.

Antes de crear una entrega o un commit, es recomendable ejecutar:

```bash
uv run pytest -q
uv run python scripts/validar_entrega.py --ejecutar-generador
git diff --check
git status
```
