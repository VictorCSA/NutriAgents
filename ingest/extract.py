import json
import logging
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDFS_DIR = DATA_DIR / "raw" / "pdfs"
SOURCES_FILE = DATA_DIR / "sources.json"
OUTPUT_DIR = DATA_DIR / "processed" / "extracted"
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "ingest_extract.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# Threshold mínimo de caracteres para considerar uma página válida
MIN_PAGE_CHARS = 50



# Funções auxiliares
def load_sources() -> list[dict]:
    """Carrega e retorna o conteúdo de sources.json."""
    if not SOURCES_FILE.exists():
        raise FileNotFoundError(f"sources.json não encontrado em {SOURCES_FILE}")

    with open(SOURCES_FILE, encoding="utf-8") as f:
        sources = json.load(f)

    logger.info(f"{len(sources)} fonte(s) carregada(s) do sources.json")
    return sources


def parse_page_range(pages_used: str, total_pages: int) -> range:
    """
    Interpreta o campo pages_used do sources.json.

    Formatos aceitos:
        "all"     → todas as páginas
        "10-45"   → páginas 10 a 45 (1-indexed, inclusive)
        "5"       → só a página 5

    Retorna um range de índices 0-indexed para uso com PyMuPDF.
    """
    pages_used = str(pages_used).strip().lower()

    if pages_used == "all":
        return range(total_pages)

    # Range "10-45"
    range_match = re.fullmatch(r"(\d+)-(\d+)", pages_used)
    if range_match:
        start = int(range_match.group(1)) - 1  # converte para 0-indexed
        end = int(range_match.group(2))         # range() é exclusivo no fim
        start = max(0, start)
        end = min(total_pages, end)
        if start >= end:
            raise ValueError(
                f"Range inválido '{pages_used}': start >= end após clamp."
            )
        return range(start, end)

    # Página única "5"
    single_match = re.fullmatch(r"(\d+)", pages_used)
    if single_match:
        page = int(single_match.group(1)) - 1
        page = max(0, min(page, total_pages - 1))
        return range(page, page + 1)

    raise ValueError(
        f"Formato de pages_used não reconhecido: '{pages_used}'. "
        "Use 'all', '10-45' ou '5'."
    )


def is_valid_page(text: str) -> bool:
    """
    Retorna True se a página tem conteúdo suficiente para ser indexada.
    Filtra capas, páginas de índice e páginas essencialmente vazias.
    """
    stripped = text.strip()
    return len(stripped) >= MIN_PAGE_CHARS


def extract_pdf(source: dict) -> list[dict]:
    """
    Extrai texto de um PDF conforme a configuração em sources.json.

    Retorna lista de dicts, um por página válida:
    {
        source_id, filename, title, publisher, year,
        page (1-indexed), text
    }
    """
    source_id = source["id"]
    filename = source["filename"]
    pdf_path = RAW_PDFS_DIR / filename

    if not pdf_path.exists():
        logger.error(f"[{source_id}] PDF não encontrado: {pdf_path}")
        return []

    pages_used_raw = source.get("pages_used", "all")
    extracted_pages = []
    discarded = 0

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info(
            f"[{source_id}] Abrindo '{filename}' — {total_pages} página(s) total"
        )

        page_range = parse_page_range(pages_used_raw, total_pages)

        for page_idx in page_range:
            page = doc[page_idx]
            text = page.get_text("text")  # extração de texto nativo (não OCR)

            if not is_valid_page(text):
                discarded += 1
                logger.debug(
                    f"[{source_id}] Página {page_idx + 1} descartada "
                    f"({len(text.strip())} chars)"
                )
                continue

            extracted_pages.append(
                {
                    "source_id": source_id,
                    "filename": filename,
                    "title": source.get("title", ""),
                    "publisher": source.get("publisher", ""),
                    "year": source.get("year", ""),
                    "page": page_idx + 1,  # volta para 1-indexed na saída
                    "text": text,
                }
            )

        doc.close()

    except Exception as e:
        logger.error(f"[{source_id}] Erro ao processar '{filename}': {e}")
        return []

    total_extracted = len(extracted_pages)
    total_range = len(page_range)
    logger.info(
        f"[{source_id}] {total_extracted}/{total_range} página(s) extraída(s) "
        f"| {discarded} descartada(s)"
    )

    # Alerta se descarte for alto — pode indicar PDF problemático
    if total_range > 0 and (discarded / total_range) > 0.2:
        logger.warning(
            f"[{source_id}] ATENÇÃO: {discarded}/{total_range} páginas descartadas "
            f"({discarded/total_range:.0%}). Verifique o PDF."
        )

    return extracted_pages


def save_extracted(source_id: str, pages: list[dict]) -> Path:
    """Salva a lista de páginas extraídas em processed/extracted/{source_id}.jsonl"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{source_id}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    logger.info(f"[{source_id}] Salvo em {output_path}")
    return output_path



# Ponto de entrada
def extract_all(sources: list[dict]) -> dict[str, int]:
    """
    Itera sobre todas as fontes do sources.json e extrai os PDFs.

    Ignora fontes sem 'filename' terminando em .pdf (ex.: a TACO csv).

    Retorna dict {source_id: páginas_extraídas} para resumo final.
    """
    summary = {}

    for source in sources:
        filename = source.get("filename", "")

        if not filename.lower().endswith(".pdf"):
            logger.info(
                f"[{source['id']}] Pulando '{filename}' — não é PDF "
                "(será tratado pelo MCP)"
            )
            continue

        pages = extract_pdf(source)

        if pages:
            save_extracted(source["id"], pages)

        summary[source["id"]] = len(pages)

    return summary


def main() -> None:
    logger.info("=" * 60)
    logger.info("NutriAgents — extract.py iniciado")
    logger.info("=" * 60)

    sources = load_sources()
    summary = extract_all(sources)

    logger.info("=" * 60)
    logger.info("RESUMO DA EXTRAÇÃO")
    logger.info("=" * 60)
    total_pages = 0
    for source_id, count in summary.items():
        status = "✓" if count > 0 else "✗"
        logger.info(f"  {status} {source_id}: {count} página(s)")
        total_pages += count

    logger.info(f"\n  Total: {total_pages} página(s) extraída(s)")
    logger.info(f"  Saída: {OUTPUT_DIR}")
    logger.info("=" * 60)

    if any(count == 0 for count in summary.values()):
        logger.warning(
            "Alguns documentos retornaram 0 páginas. "
            "Verifique os logs acima e os arquivos em raw/pdfs/."
        )
        sys.exit(1)

    logger.info("Extração concluída com sucesso.")


if __name__ == "__main__":
    main()