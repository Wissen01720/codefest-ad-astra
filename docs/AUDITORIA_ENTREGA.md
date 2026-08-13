# Auditoría de preparación de entrega CodeFest Ad Astra 2026

Fecha de revisión: 13 de agosto de 2026  
Rama de trabajo: `agent/preparar-entrega-codefest`

## Alcance

Se contrastaron el código, la base vectorial y los artefactos del repositorio con la especificación oficial de la Etapa 1 y la documentación técnica del equipo. No se eliminó ningún archivo; los documentos que describen una etapa anterior se conservaron con una advertencia de vigencia.

## Estado antes y después

| Componente exigido | Estado inicial | Estado auditado |
|---|---|---|
| `resultados.jsonl` | Presente | 50 consultas, orden `q001`-`q050`, 3 documentos y 10 fragmentos por consulta |
| `generador.py` | Solo implementación interna bajo `src/` | Versión autónoma en `entrega/generador.py` |
| `informe_tecnico.pdf` | Ausente | A4, 7 páginas y secciones técnicas exigidas |
| `base_vectorial/encoder_<nombre>/` | Fuera del repositorio | Incorporada desde el `.tar.gz`; índice y metadata validados |
| Grafo de conocimiento | No implementado | Se mantiene sin implementar; componente opcional |

## Comparación de requisitos con el código

| Requisito | Hallazgo | Ajuste aplicado | Estado |
|---|---|---|---|
| Fragmentos con oraciones completas | El fallback por espacios podía dividir prosa sobredimensionada | Se restringió a `csv`, `xlsx` y `pbf`; la prosa usa `fail` o `skip-document` | Conforme en código y pruebas |
| Máximo de 250 palabras | El generador podía cortar por palabras una primera oración larga | El candidato se descarta y la búsqueda amplía `k` | Conforme |
| Ranking de 10 fragmentos | Los fragmentos se limitaban a los tres documentos agregados | El ranking global de FAISS se conserva por separado | Conforme |
| Exactamente 3 documentos | Agregación por suma de scores de chunks | Se conservó y reforzó la validación | Conforme |
| Similitud coseno | `IndexFlatIP` sobre vectores normalizados | Verificación real de dimensión, tipo y normas | Conforme |
| Retiro de URL antes del encoder | Estaba documentado, pero no versionado | Se añadió `texto_para_encoder`; metadata conserva el original | Conforme en código; la base recibida registra una re-encoderización previa de 14.210 vectores |
| Metadata alineada con FAISS | Existía validación en la implementación | Se abrió el binario y se recorrieron los registros | Conforme: 202.350 vectores y 202.350 líneas |
| Campos obligatorios de metadata | Requeridos por la Tabla 1 | Se verificaron los ocho campos en cada registro | Conforme |
| Prohibición de modelos generativos | No se encontraron decoders en indexación/recuperación | El validador inspecciona imports del generador | Conforme |
| Generador reproducible | Faltaba ejecutable dentro de `entrega/` | Se añadió carga de índice, metadata y manifest, consulta normalizada y escritura atómica | Conforme estáticamente; ejecución completa requiere los pesos locales del encoder |

## Evidencia de la base vectorial

| Métrica | Valor validado |
|---|---:|
| Tipo de índice | `IndexFlatIP` |
| Dimensión | 1.024 |
| Vectores / registros de metadata | 202.350 / 202.350 |
| Documentos y fuentes únicas | 1.734 |
| Tokens promedio / máximo | 347,5 / 450 |
| Fenómeno 1 | 134.224 fragmentos |
| Fenómeno 2 | 36.644 fragmentos |
| Fenómeno 3 | 31.482 fragmentos |

La base se recibió como `base_vectorial_bge-m3.tar.gz` (839.738.557 bytes), SHA-256 `8b01f1523d96b9547b68a91f1de2966c8f25de910c4416e5da063e54de58fb9a`. El archivo gzip y las rutas internas fueron validados antes de la extracción.

## Organización aplicada

- `README.md` refleja el estado, la estructura y los comandos vigentes.
- `docs/FASE_3_CHUNKING.md` se identifica como documento histórico.
- `entrega/` contiene los cuatro componentes obligatorios.
- `scripts/validar_entrega.py` concentra los controles del paquete.
- `tests/test_delivery_compliance.py` cubre los riesgos descubiertos durante la auditoría.

## Verificaciones

| Verificación | Resultado |
|---|---|
| Suite de pruebas | 63 pasaron; 1 omitida |
| Ruff crítico (`E9`, `F63`, `F7`, `F82`) | 0 errores |
| Esquema de `resultados.jsonl` | Conforme |
| Base FAISS y metadata | Conforme |
| Autonomía y restricciones de `entrega/generador.py` | Conforme |
| Informe técnico | 7 páginas A4; conforme con el máximo de 8 |
| Reproducción byte a byte de las 50 consultas | Pendiente si el entorno no tiene disponibles los pesos de `BAAI/bge-m3` |

## Cierre recomendado

1. Mantener el `.tar.gz` como respaldo y distribuirlo por un canal apto para archivos grandes.
2. Ejecutar `uv run python scripts/validar_entrega.py` tras cualquier movimiento del paquete.
3. Cuando el encoder esté disponible localmente, ejecutar `uv run python scripts/validar_entrega.py --ejecutar-generador` y exigir coincidencia byte a byte antes del envío definitivo.
