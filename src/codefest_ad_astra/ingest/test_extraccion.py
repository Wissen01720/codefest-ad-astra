"""Prueba extraer_y_limpiar() contra una muestra de documentos reales,
cubriendo los formatos y casos límite del corpus, antes de entregarlo
al equipo de chunking.

Uso:
    uv run python -m codefest_ad_astra.ingest.test_extraccion
"""

from pathlib import Path

from .extraer_y_limpiar import extraer_y_limpiar
from .quality_metrics import assess_text_quality

# Ajusta estas rutas si alguna no existe exactamente así en tu disco --
# se tomaron del árbol de data/raw_muestra que ya compartiste.
MUESTRA = [
    "F1_IA_y_Capacidades_Estrategicas/AI_Index_Stanford/pdfs/AIINDEX_ai-index-report-2024.pdf",
    "F2_Seguridad_Entorno_Espacial/SWF_Counterspace/pdfs/Counterspace_Reports/SWF_global-counterspace-capabilities-2026-hr.pdf",
    "F2_Seguridad_Entorno_Espacial/ESA_Space_Debris/articulos/ESA_about-space-debris.json",
    "F3_Dinamicas_Territoriales/Alertas_Tempranas/alertas/ALERTAS_001-17-91689.json",
    "F1_IA_y_Capacidades_Estrategicas/AI_Index_Stanford/recursos/Healthcare_Medicine/datasets/AIINDEX_clinicaltrials-artificial-intelligence-csv.csv",
]


def main() -> None:
    corpus = Path("data/raw_muestra")

    for relativa in MUESTRA:
        path = corpus / relativa
        print("=" * 80)
        print(f"Archivo: {relativa}")

        if not path.exists():
            print("  [OMITIDO] no existe esa ruta exacta -- ajústala en MUESTRA")
            continue

        try:
            resultado = extraer_y_limpiar(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {type(exc).__name__}: {exc}")
            continue

        print(f"  idioma: {resultado.idioma}")
        print(f"  páginas originales: {resultado.paginas_originales}")
        print(f"  boilerplate removido: {resultado.lineas_boilerplate_removidas}")
        print(f"  caracteres totales: {len(resultado.texto)}")

        calidad = assess_text_quality(resultado.texto)
        print(f"  calidad: {calidad.quality_label} (score={calidad.quality_score:.2f}), "
              f"líneas útiles={calidad.useful_lines}/{calidad.lines}, "
              f"repetición={calidad.repetition_ratio:.2f}")

        print(f"  primeros 300 caracteres:")
        print(f"  {resultado.texto[:300]!r}")


if __name__ == "__main__":
    main()