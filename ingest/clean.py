import json
import logging
import re
import sys
from pathlib import Path

# Configuração de paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "processed" / "extracted"
OUTPUT_DIR = DATA_DIR / "processed" / "cleaned"
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "ingest_clean.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Cabeçalhos e rodapés típicos
HEADER_FOOTER_PATTERNS: list[re.Pattern] = [
    # "Ministério da Saúde" sozinho na linha
    re.compile(r"^\s*Minist[eé]rio da Sa[uú]de\s*$", re.MULTILINE | re.IGNORECASE),
    # "ANVISA" sozinho na linha
    re.compile(r"^\s*ANVISA\s*$", re.MULTILINE),
    # Número de página solto: "47" ou "- 47 -" ou "Página 47"
    re.compile(r"^\s*[-–]?\s*P[áa]gina\s+\d+\s*[-–]?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*[-–]?\s*\d{1,3}\s*[-–]?\s*$", re.MULTILINE),
    # "Guia Alimentar para a População Brasileira" como cabeçalho de página
    re.compile(r"^\s*Guia Alimentar para a Popula[çc][ãa]o Brasileira\s*$", re.MULTILINE | re.IGNORECASE),
    # Direitos reservados / copyright no rodapé
    re.compile(r"^\s*Todos os direitos reservados.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*[©Cc]opyright.*$", re.MULTILINE | re.IGNORECASE),
    # URLs soltas (rodapés com site institucional)
    re.compile(r"^\s*www\.[^\s]+\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*http[s]?://[^\s]+\s*$", re.MULTILINE | re.IGNORECASE),
    # "ISBN", "ISSN" no rodapé
    re.compile(r"^\s*IS[BS]N[\s:]+[\d\-X]+\s*$", re.MULTILINE | re.IGNORECASE),
]


# Funções de limpeza

def remove_headers_footers(text: str) -> str:
    """Remove padrões de cabeçalho e rodapé conhecidos."""
    for pattern in HEADER_FOOTER_PATTERNS:
        text = pattern.sub("", text)
    return text


def fix_hyphenation(text: str) -> str:
    """
    Reconstitui palavras quebradas por hifenização no fim de linha.

    Padrão: "ali-\nmentos" → "alimentos"
    Só une quando a quebra é claramente de hifenização (letra-\nletra minúscula).
    Não toca em hífens compostos legítimos como "guia-alimentar".
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """
    Normaliza espaços e quebras de linha:
    - Múltiplas linhas em branco → uma linha em branco (separador de parágrafo)
    - Múltiplos espaços → um espaço
    - Espaços antes de pontuação → sem espaço
    """
    # Múltiplas quebras de linha → dupla (preserva parágrafo)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Quebra de linha simples dentro de parágrafo → espaço
    # (linhas que NÃO são seguidas por linha em branco)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Múltiplos espaços → um
    text = re.sub(r" {2,}", " ", text)
    # Espaço antes de pontuação
    text = re.sub(r" ([.,;:!?])", r"\1", text)
    return text


def remove_control_characters(text: str) -> str:
    """
    Remove caracteres de controle e não-imprimíveis que PyMuPDF
    ocasionalmente extrai de PDFs com encoding ruim.
    Preserva \n (usada para estrutura de parágrafo).
    """
    # Remove tudo que não seja imprimível exceto \n e \t
    cleaned = re.sub(r"[^\x09\x0A\x20-\x7E\x80-\xFF\u0100-\uFFFF]", "", text)
    return cleaned


def remove_repeated_punctuation(text: str) -> str:
    """
    Remove sequências de pontuação repetida que indicam
    ruído de extração (ex: "........", "--------", "========").
    """
    return re.sub(r"([.\-=_*#])\1{3,}", "", text)


def clean_text(text: str) -> str:
    """
    Pipeline de limpeza completo aplicado a uma página.
    A ordem das operações importa.
    """
    text = remove_control_characters(text)
    text = remove_headers_footers(text)
    text = fix_hyphenation(text)
    text = remove_repeated_punctuation(text)
    text = normalize_whitespace(text)
    return text.strip()


# Métricas de limpeza (para log e auditoria)
def compute_reduction(original: str, cleaned: str) -> float:
    """Retorna o percentual de redução de caracteres após limpeza."""
    if len(original) == 0:
        return 0.0
    return (1 - len(cleaned) / len(original)) * 100


# Processamento por documento
def clean_document(source_id: str) -> dict:
    """
    Lê processed/extracted/{source_id}.jsonl, limpa cada página
    e salva em processed/cleaned/{source_id}.jsonl.

    Retorna métricas: {pages_processed, pages_discarded_after_clean, avg_reduction_pct}
    """
    input_path = EXTRACTED_DIR / f"{source_id}.jsonl"

    if not input_path.exists():
        logger.error(f"[{source_id}] Arquivo extraído não encontrado: {input_path}")
        return {"pages_processed": 0, "pages_discarded_after_clean": 0}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{source_id}.jsonl"

    pages_processed = 0
    pages_discarded = 0
    reductions = []

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            page_obj = json.loads(line)
            original_text = page_obj["text"]

            cleaned = clean_text(original_text)
            reduction = compute_reduction(original_text, cleaned)
            reductions.append(reduction)

            # Após limpeza, página pode ter ficado muito curta (era quase só ruído)
            if len(cleaned) < 50:
                pages_discarded += 1
                logger.debug(
                    f"[{source_id}] p.{page_obj['page']} descartada após limpeza "
                    f"({len(cleaned)} chars restantes)"
                )
                continue

            # Alerta se a redução for muito agressiva — pode indicar falso positivo
            if reduction > 60:
                logger.warning(
                    f"[{source_id}] p.{page_obj['page']} teve {reduction:.1f}% "
                    "de redução. Verifique se o padrão de limpeza é muito amplo."
                )

            page_obj["text"] = cleaned
            fout.write(json.dumps(page_obj, ensure_ascii=False) + "\n")
            pages_processed += 1

    avg_reduction = sum(reductions) / len(reductions) if reductions else 0.0
    logger.info(
        f"[{source_id}] {pages_processed} página(s) limpas | "
        f"{pages_discarded} descartada(s) | "
        f"redução média: {avg_reduction:.1f}%"
    )

    return {
        "pages_processed": pages_processed,
        "pages_discarded_after_clean": pages_discarded,
        "avg_reduction_pct": round(avg_reduction, 1),
    }


# Ponto de entrada
def clean_all() -> None:
    """Processa todos os .jsonl disponíveis em processed/extracted/."""
    extracted_files = sorted(EXTRACTED_DIR.glob("*.jsonl"))

    if not extracted_files:
        logger.error(
            f"Nenhum arquivo .jsonl encontrado em {EXTRACTED_DIR}. "
            "Rode extract.py primeiro."
        )
        sys.exit(1)

    logger.info(f"{len(extracted_files)} documento(s) para limpar.")

    summary = {}
    for path in extracted_files:
        source_id = path.stem
        metrics = clean_document(source_id)
        summary[source_id] = metrics

    # Resumo final
    logger.info("=" * 60)
    logger.info("RESUMO DA LIMPEZA")
    logger.info("=" * 60)
    total_pages = 0
    total_discarded = 0
    for source_id, m in summary.items():
        status = "✓" if m["pages_processed"] > 0 else "✗"
        logger.info(
            f"  {status} {source_id}: "
            f"{m['pages_processed']} páginas | "
            f"{m['pages_discarded_after_clean']} descartadas | "
            f"redução média {m.get('avg_reduction_pct', 0)}%"
        )
        total_pages += m["pages_processed"]
        total_discarded += m["pages_discarded_after_clean"]

    logger.info(f"\n  Total útil: {total_pages} página(s)")
    logger.info(f"  Total descartado após limpeza: {total_discarded} página(s)")
    logger.info(f"  Saída: {OUTPUT_DIR}")
    logger.info("=" * 60)

    if any(m["pages_processed"] == 0 for m in summary.values()):
        logger.warning(
            "Alguns documentos resultaram em 0 páginas após limpeza. "
            "Verifique os avisos acima."
        )
        sys.exit(1)

    logger.info("Limpeza concluída com sucesso.")


def main() -> None:
    logger.info("=" * 60)
    logger.info("NutriAgents — clean.py iniciado")
    logger.info("=" * 60)
    clean_all()


if __name__ == "__main__":
    main()