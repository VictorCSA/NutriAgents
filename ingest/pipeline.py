import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INGEST_DIR = PROJECT_ROOT / "ingest"
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Adiciona ingest/ ao path para importar os módulos irmãos
sys.path.insert(0, str(INGEST_DIR))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "ingest_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Modelo de resultado por etapa
@dataclass
class StepResult:
    name: str
    success: bool
    duration_sec: float
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# Definição das etapas
STEPS = ["extract", "clean", "chunk", "embed_and_index"]


def run_extract() -> StepResult:
    """
    Etapa 1 — Extração de texto dos PDFs.
    Erros por documento são alertados; erros de código param o pipeline.
    """
    from extract import load_sources, extract_all

    start = time.time()
    warnings = []

    sources = load_sources()  # FileNotFoundError aqui → para imediatamente
    summary = extract_all(sources)  # erros por PDF já são logados internamente

    for source_id, count in summary.items():
        if count == 0:
            warnings.append(f"{source_id}: 0 páginas extraídas — verifique o PDF")

    return StepResult(
        name="extract",
        success=True,
        duration_sec=round(time.time() - start, 2),
        warnings=warnings,
        details={"pages_per_doc": summary},
    )


def run_clean() -> StepResult:
    """
    Etapa 2 — Limpeza do texto extraído.
    Documentos sem saída útil geram alerta; pipeline continua.
    """
    from clean import clean_all_pipeline

    start = time.time()
    warnings = []

    summary = clean_all_pipeline()  # retorna dict {source_id: metrics}

    for source_id, m in summary.items():
        if m["pages_processed"] == 0:
            warnings.append(
                f"{source_id}: 0 páginas após limpeza — "
                "verifique HEADER_FOOTER_PATTERNS"
            )

    return StepResult(
        name="clean",
        success=True,
        duration_sec=round(time.time() - start, 2),
        warnings=warnings,
        details={"metrics_per_doc": summary},
    )


def run_chunk() -> StepResult:
    """
    Etapa 3 — Chunking e enriquecimento de metadados.
    """
    from chunk import chunk_all_pipeline

    start = time.time()
    warnings = []

    summary = chunk_all_pipeline()  # retorna {source_id: n_chunks}

    for source_id, n in summary.items():
        if n == 0:
            warnings.append(f"{source_id}: 0 chunks gerados")

    total_chunks = sum(summary.values())

    return StepResult(
        name="chunk",
        success=True,
        duration_sec=round(time.time() - start, 2),
        warnings=warnings,
        details={"chunks_per_doc": summary, "total_chunks": total_chunks},
    )


def run_embed_and_index() -> StepResult:
    """
    Etapa 4 — Geração de embeddings e indexação no FAISS.
    """
    from embed_and_index import embed_and_index_pipeline

    start = time.time()

    result = embed_and_index_pipeline()  # retorna {"total_vectors": int, "index_path": str}

    return StepResult(
        name="embed_and_index",
        success=True,
        duration_sec=round(time.time() - start, 2),
        details=result,
    )


# Mapa de etapas
STEP_RUNNERS = {
    "extract": run_extract,
    "clean": run_clean,
    "chunk": run_chunk,
    "embed_and_index": run_embed_and_index,
}


# Orquestrador principal
def resolve_steps(from_step: str | None, only_step: str | None) -> list[str]:
    """Retorna a lista de etapas a executar com base nos argumentos CLI."""
    if only_step:
        if only_step not in STEPS:
            raise ValueError(f"Etapa desconhecida: '{only_step}'. Opções: {STEPS}")
        return [only_step]

    if from_step:
        if from_step not in STEPS:
            raise ValueError(f"Etapa desconhecida: '{from_step}'. Opções: {STEPS}")
        return STEPS[STEPS.index(from_step):]

    return STEPS  # padrão: todas


def run_pipeline(steps: list[str]) -> None:
    results: list[StepResult] = []
    pipeline_start = time.time()

    logger.info("=" * 60)
    logger.info("NutriGuia — pipeline.py iniciado")
    logger.info(f"Etapas: {' → '.join(steps)}")
    logger.info("=" * 60)

    for step_name in steps:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"  ETAPA: {step_name.upper()}")
        logger.info(f"{'─' * 60}")

        runner = STEP_RUNNERS[step_name]

        try:
            result = runner()
            results.append(result)

        except Exception:
            # Erro de código — para imediatamente com traceback completo
            logger.error(
                f"\n[FATAL] Erro inesperado na etapa '{step_name}'. "
                "Pipeline interrompido.\n"
            )
            logger.error(traceback.format_exc())
            _print_summary(results, failed_step=step_name)
            sys.exit(1)

        # Warnings de documento — loga e continua
        for w in result.warnings:
            logger.warning(f"  ⚠  {w}")

        logger.info(
            f"  ✓ '{step_name}' concluída em {result.duration_sec}s"
        )

    total_time = round(time.time() - pipeline_start, 2)
    _print_summary(results, total_time=total_time)


def _print_summary(
    results: list[StepResult],
    failed_step: str | None = None,
    total_time: float | None = None,
) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("RESUMO DO PIPELINE")
    logger.info("=" * 60)

    for r in results:
        status = "✓" if r.success else "✗"
        warn_str = f" | {len(r.warnings)} aviso(s)" if r.warnings else ""
        logger.info(f"  {status} {r.name:<20} {r.duration_sec:>6.1f}s{warn_str}")

        # Detalhes relevantes por etapa
        if r.name == "extract" and "pages_per_doc" in r.details:
            total = sum(r.details["pages_per_doc"].values())
            logger.info(f"      → {total} página(s) extraída(s)")

        if r.name == "chunk" and "total_chunks" in r.details:
            logger.info(f"      → {r.details['total_chunks']} chunks gerados")

        if r.name == "embed_and_index" and "total_vectors" in r.details:
            logger.info(f"      → {r.details['total_vectors']} vetores indexados")
            logger.info(f"      → índice em: {r.details.get('index_path', '—')}")

    if failed_step:
        logger.info(f"\n  ✗ FALHOU em: {failed_step}")

    if total_time:
        logger.info(f"\n  Tempo total: {total_time}s")

    logger.info("=" * 60)

    # Salva resumo em JSON para auditoria
    summary_path = LOGS_DIR / "ingest_summary.json"
    summary_data = {
        "steps": [
            {
                "name": r.name,
                "success": r.success,
                "duration_sec": r.duration_sec,
                "warnings": r.warnings,
                "details": r.details,
            }
            for r in results
        ],
        "failed_step": failed_step,
        "total_time_sec": total_time,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    logger.info(f"  Resumo salvo em: {summary_path}")


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NutriAgents — pipeline de ingest",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from",
        dest="from_step",
        metavar="STEP",
        choices=STEPS,
        help=(
            "Retoma o pipeline a partir de uma etapa específica.\n"
            f"Opções: {', '.join(STEPS)}"
        ),
    )
    group.add_argument(
        "--only",
        dest="only_step",
        metavar="STEP",
        choices=STEPS,
        help="Executa somente uma etapa específica.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        steps = resolve_steps(
            from_step=args.from_step,
            only_step=args.only_step,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    run_pipeline(steps)


if __name__ == "__main__":
    main()