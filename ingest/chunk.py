import hashlib
import json
import logging
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuração de paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DIR = DATA_DIR / "processed" / "cleaned"
SOURCES_FILE = DATA_DIR / "sources.json"
OUTPUT_DIR = DATA_DIR / "processed" / "chunks"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Parâmetros de chunking

# Tamanho alvo de cada chunk em caracteres (~400-500 tokens para bge-m3)
# bge-m3 usa ~3.5 chars/token para português → 500 tokens ≈ 1750 chars
CHUNK_SIZE = 1500

# Overlap entre chunks adjacentes em caracteres (~50 tokens ≈ 175 chars)
CHUNK_OVERLAP = 200

# Separadores tentados em ordem pelo RecursiveCharacterTextSplitter
# Tenta quebrar em parágrafo primeiro, depois frase, depois palavra
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "ingest_chunk.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# Funções auxiliares
def load_sources() -> dict[str, dict]:
    """
    Carrega sources.json e retorna dict indexado por source_id
    para lookup O(1) durante o enriquecimento de metadados.
    """
    if not SOURCES_FILE.exists():
        raise FileNotFoundError(f"sources.json não encontrado em {SOURCES_FILE}")

    with open(SOURCES_FILE, encoding="utf-8") as f:
        sources_list = json.load(f)

    sources = {s["id"]: s for s in sources_list}
    logger.info(f"{len(sources)} fonte(s) carregada(s) do sources.json")
    return sources


def make_chunk_id(source_id: str, page: int, chunk_index: int) -> str:
    """
    Gera um chunk_id legível e único.
    Formato: {source_id}_p{page:04d}_c{chunk_index:03d}

    Exemplo: guia_alimentar_ms_p0047_c002
    """
    return f"{source_id}_p{page:04d}_c{chunk_index:03d}"


def make_chunk_hash(text: str) -> str:
    """
    Hash SHA-1 curto do texto do chunk (primeiros 8 chars).
    Útil para detectar duplicatas se documentos forem reindexados.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def build_citation(source: dict, page: int) -> str:
    """
    Monta a string de citação que o Answerer vai usar na resposta.
    Exemplo: "Guia Alimentar para a População Brasileira (MS, 2014, p. 47)"
    """
    title = source.get("title", source["id"])
    publisher = source.get("publisher", "")
    year = source.get("year", "")
    parts = [p for p in [publisher, str(year)] if p]
    meta = ", ".join(parts)
    if meta:
        return f"{title} ({meta}, p. {page})"
    return f"{title} (p. {page})"


# Chunking de um documento
def chunk_document(source_id: str, sources: dict[str, dict]) -> list[dict]:
    """
    Lê processed/cleaned/{source_id}.jsonl e produz lista de chunks
    enriquecidos com metadados.

    Estratégia:
    - Cada página é chunkada individualmente (não concatena páginas)
    - Isso preserva a rastreabilidade de página na citação
    - RecursiveCharacterTextSplitter respeita limites naturais de parágrafo/frase
    """
    input_path = CLEANED_DIR / f"{source_id}.jsonl"

    if not input_path.exists():
        logger.error(f"[{source_id}] Arquivo limpo não encontrado: {input_path}")
        return []

    source_meta = sources.get(source_id, {})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks = []
    pages_processed = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            page_obj = json.loads(line)
            text = page_obj["text"]
            page_num = page_obj["page"]

            # Divide a página em chunks
            splits = splitter.split_text(text)

            for chunk_idx, chunk_text in enumerate(splits):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                chunk_id = make_chunk_id(source_id, page_num, chunk_idx)
                citation = build_citation(source_meta, page_num)

                chunk = {
                    # Identificação
                    "chunk_id": chunk_id,
                    "chunk_hash": make_chunk_hash(chunk_text),
                    # Rastreabilidade
                    "source_id": source_id,
                    "filename": page_obj.get("filename", ""),
                    "title": page_obj.get("title", ""),
                    "publisher": page_obj.get("publisher", ""),
                    "year": page_obj.get("year", ""),
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    # Citação pronta para o Answerer
                    "citation": citation,
                    # Conteúdo
                    "text": chunk_text,
                    # Métricas do chunk
                    "char_count": len(chunk_text),
                    "topics": source_meta.get("topics", []),
                }

                all_chunks.append(chunk)

            pages_processed += 1

    total_chunks = len(all_chunks)
    avg_chars = (
        sum(c["char_count"] for c in all_chunks) / total_chunks
        if total_chunks > 0
        else 0
    )

    logger.info(
        f"[{source_id}] {pages_processed} página(s) → "
        f"{total_chunks} chunk(s) | "
        f"média {avg_chars:.0f} chars/chunk"
    )

    # Alerta se chunks estiverem muito pequenos (chunking excessivo)
    if avg_chars < 300 and total_chunks > 0:
        logger.warning(
            f"[{source_id}] Chunks com média muito baixa ({avg_chars:.0f} chars). "
            "Considere aumentar CHUNK_SIZE ou verificar a qualidade do texto limpo."
        )

    return all_chunks


# Consolidação e persistência
def save_chunks(all_chunks: list[dict]) -> None:
    """
    Salva todos os chunks em um único processed/chunks/chunks.jsonl.
    Um chunk por linha (formato JSONL para leitura incremental eficiente).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"chunks.jsonl salvo em {OUTPUT_FILE} ({len(all_chunks)} chunks)")


def check_duplicates(all_chunks: list[dict]) -> None:
    """
    Verifica e loga chunks duplicados via hash.
    Duplicatas podem ocorrer se o mesmo documento for processado duas vezes.
    """
    hashes = [c["chunk_hash"] for c in all_chunks]
    unique = set(hashes)
    duplicates = len(hashes) - len(unique)

    if duplicates > 0:
        logger.warning(
            f"{duplicates} chunk(s) duplicado(s) detectado(s) via hash. "
            "Verifique se algum documento foi indexado mais de uma vez."
        )
    else:
        logger.info("Nenhum chunk duplicado detectado.")


# Ponto de entrada
def chunk_all() -> None:
    """Processa todos os documentos limpos e consolida em chunks.jsonl."""
    cleaned_files = sorted(CLEANED_DIR.glob("*.jsonl"))

    if not cleaned_files:
        logger.error(
            f"Nenhum arquivo .jsonl encontrado em {CLEANED_DIR}. "
            "Rode clean.py primeiro."
        )
        sys.exit(1)

    logger.info(f"{len(cleaned_files)} documento(s) para chunkar.")

    sources = load_sources()
    all_chunks = []

    for path in cleaned_files:
        source_id = path.stem
        chunks = chunk_document(source_id, sources)
        all_chunks.extend(chunks)

    check_duplicates(all_chunks)
    save_chunks(all_chunks)

    # Resumo final
    logger.info("=" * 60)
    logger.info("RESUMO DO CHUNKING")
    logger.info("=" * 60)

    # Agrupa por source_id para exibir
    by_source: dict[str, int] = {}
    for chunk in all_chunks:
        sid = chunk["source_id"]
        by_source[sid] = by_source.get(sid, 0) + 1

    for source_id, count in by_source.items():
        logger.info(f"  ✓ {source_id}: {count} chunk(s)")

    logger.info(f"\n  Total: {len(all_chunks)} chunk(s)")
    logger.info(f"  Saída: {OUTPUT_FILE}")
    logger.info("=" * 60)
    logger.info("Chunking concluído com sucesso.")


def main() -> None:
    logger.info("=" * 60)
    logger.info("NutriAgents — chunk.py iniciado")
    logger.info("=" * 60)
    chunk_all()


if __name__ == "__main__":
    main()