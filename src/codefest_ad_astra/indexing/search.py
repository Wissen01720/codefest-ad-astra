"""Búsqueda mínima sobre la base vectorial — verificación de la Fase 4.

Alcance deliberadamente corto: consulta → vector → top-k fragmentos. Sirve para
comprobar que el índice quedó bien construido y alineado.

La Fase 6 (módulo de recuperación) es la que agrega los fragmentos a nivel de
documento, fusiona varios encoders (RRF/CombSUM) y aplica post-filtros; nada de
eso vive aquí.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .encoders import MAX_SEQ_LENGTH_POR_DEFECTO, crear_encoder
from .faiss_store import cargar_base_vectorial


@dataclass(slots=True)
class Resultado:
    rank: int
    score: float
    metadata: dict[str, Any]

    @property
    def chunk_id(self) -> str:
        return self.metadata["chunk_id"]

    @property
    def doc_id(self) -> str:
        return self.metadata["doc_id"]

    @property
    def texto(self) -> str:
        return self.metadata["texto"]


class Buscador:
    """Carga una base vectorial y responde consultas en lenguaje natural.

    Usa obligatoriamente el mismo encoder y los mismos prefijos con que se
    construyó el índice (Sección 8.1): el manifiesto es la fuente de verdad, no
    lo que el usuario pase por línea de comandos.
    """

    def __init__(
        self,
        directorio: Path,
        *,
        device: str | None = None,
        encoder=None,
        mostrar_progreso: bool = False,
    ) -> None:
        self.directorio = Path(directorio)
        self.indice, self.metadata, self.manifiesto = cargar_base_vectorial(self.directorio)

        if encoder is None:
            nombre_modelo = self.manifiesto.get("modelo")
            if not nombre_modelo:
                raise ValueError(
                    f"{self.directorio / 'manifest.json'} no indica con qué modelo se construyó "
                    "el índice; no se puede consultar sin esa garantía."
                )
            encoder = crear_encoder(
                nombre_modelo,
                device=device,
                max_seq_length=self.manifiesto.get("max_seq_length", MAX_SEQ_LENGTH_POR_DEFECTO),
                mostrar_progreso=mostrar_progreso,
                dimension=self.manifiesto.get("dimension"),
            )
            # El prefijo guardado manda sobre el inferido: si el índice se
            # construyó con uno concreto, la consulta debe usar ese mismo.
            if "prefijo_consulta" in self.manifiesto:
                encoder.prefijo_consulta = self.manifiesto["prefijo_consulta"]

        self.encoder = encoder
        if self.encoder.dimension != self.indice.d:
            raise ValueError(
                f"El encoder produce vectores de dimensión {self.encoder.dimension} pero el índice "
                f"es de dimensión {self.indice.d}. ¿Modelo distinto al que construyó el índice?"
            )

    def buscar(self, consulta: str, k: int = 10) -> list[Resultado]:
        """Devuelve los k fragmentos más similares, de mayor a menor score.

        El score es el producto interno = similitud coseno, porque tanto los
        vectores del índice como el de la consulta están normalizados.
        """
        if k <= 0:
            raise ValueError("k debe ser mayor que cero")
        k_efectivo = min(k, self.indice.ntotal)

        vector = self.encoder.codificar_consultas([consulta])
        scores, ids = self.indice.search(vector, k_efectivo)

        resultados = []
        for posicion, (score, id_interno) in enumerate(zip(scores[0], ids[0]), start=1):
            if id_interno < 0:  # FAISS rellena con -1 si hay menos vecinos que k
                continue
            resultados.append(
                Resultado(rank=posicion, score=float(score), metadata=self.metadata[int(id_interno)])
            )
        return resultados

    def buscar_lote(self, consultas: Sequence[str], k: int = 10) -> list[list[Resultado]]:
        """Igual que `buscar` pero para varias consultas (una sola pasada del encoder)."""
        if not consultas:
            return []
        k_efectivo = min(k, self.indice.ntotal)
        vectores = self.encoder.codificar_consultas(list(consultas))
        scores, ids = self.indice.search(vectores, k_efectivo)

        salida = []
        for fila_scores, fila_ids in zip(scores, ids):
            resultados = []
            for posicion, (score, id_interno) in enumerate(zip(fila_scores, fila_ids), start=1):
                if id_interno < 0:
                    continue
                resultados.append(
                    Resultado(rank=posicion, score=float(score), metadata=self.metadata[int(id_interno)])
                )
            salida.append(resultados)
        return salida


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consulta de verificación contra la base vectorial")
    parser.add_argument("--base", type=Path, required=True, help="Directorio encoder_<slug>/")
    parser.add_argument("--consulta", type=str, required=True, help="Consulta en lenguaje natural")
    parser.add_argument("-k", type=int, default=10, help="Número de fragmentos a devolver")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--json", action="store_true", help="Salida JSON en vez de texto legible")
    args = parser.parse_args(argv)

    try:
        buscador = Buscador(args.base, device=args.device)
        resultados = buscador.buscar(args.consulta, k=args.k)
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1

    if args.json:
        print(
            json.dumps(
                [{"rank": r.rank, "score": r.score, **r.metadata} for r in resultados],
                ensure_ascii=False,
            )
        )
        return 0

    print(f'Consulta: "{args.consulta}"  ({buscador.indice.ntotal} fragmentos indexados)\n')
    for r in resultados:
        vista = " ".join(r.texto.split())[:220]
        print(f"[{r.rank:2d}] score={r.score:.4f}  doc={r.doc_id}  chunk={r.chunk_id}")
        print(f"     fuente={r.metadata.get('fuente')}  fenomeno={r.metadata.get('fenomeno')}  "
              f"idioma={r.metadata.get('idioma')}")
        print(f"     {vista}...\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
