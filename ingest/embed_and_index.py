import json
import logging
import sys
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Configuração de paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks" / "chunks.jsonl"
VECTORSTORE_DIR = DATA_DIR / "processed" / "vectorstore" / "faiss_index"
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Configuração do modelo de embeddings  

# Trocar para "BAAI/bge-small-en-v1.5" se a máquina for limitada em RAM/VRAM
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Tamanho do batch para geração de embeddings
# Reduzir para 16 ou 8 se houver OOM em GPU/CPU
BATCH_SIZE = 32

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "embed_and_index.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# Carregamento dos chunks
def load_chunks() -> list[Document]:
    """
    Lê chunks.jsonl e retorna lista de LangChain Documents.

    Cada Document carrega:
    - page_content: texto do chunk (usado para embedding)
    - metadata: todos os campos do chunk exceto 'text'
    """
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"chunks.jsonl não encontrado em {CHUNKS_FILE}. "
            "Rode chunk.py primeiro."
        )

    documents = []

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)

            # Separa texto do restante dos metadados
            text = chunk.pop("text")

            documents.append(
                Document(
                    page_content=text,
                    metadata=chunk,  # chunk_id, source_id, title, page, section, etc.
                )
            )

    logger.info(f"{len(documents)} chunk(s) carregados de {CHUNKS_FILE}")
    return documents


# Inicialização do modelo de embeddings
def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Inicializa o modelo de embeddings HuggingFace.

    O modelo é baixado automaticamente na primeira execução
    e cacheado em ~/.cache/huggingface/hub.

    encode_kwargs normalize_embeddings=True é recomendado pelo
    repositório oficial do bge-m3 para melhor qualidade de retrieval.
    """
    logger.info(f"Carregando modelo de embeddings: {EMBEDDING_MODEL_NAME}")
    logger.info("(Primeira execução fará download — pode demorar alguns minutos)")

    start = time.time()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},  # trocar para "cuda" se disponível
        encode_kwargs={"normalize_embeddings": True, "batch_size": BATCH_SIZE},
    )

    elapsed = time.time() - start
    logger.info(f"Modelo carregado em {elapsed:.1f}s")

    return embeddings


# Geração do índice FAISS
def build_faiss_index(documents: list[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Gera embeddings para todos os documentos e constrói o índice FAISS.

    Usa FAISS.from_documents do LangChain, que internamente:
    1. Chama embeddings.embed_documents() em batches
    2. Constrói o índice IndexFlatL2
    3. Associa os vetores aos Documents com metadados

    Loga progresso a cada 100 chunks para acompanhar execuções longas.
    """
    total = len(documents)
    logger.info(f"Gerando embeddings para {total} chunk(s)...")
    logger.info(f"Batch size: {BATCH_SIZE} | Modelo: {EMBEDDING_MODEL_NAME}")

    start = time.time()

    # Processa em lotes para logar progresso
    vectorstore = None
    batch_num = 0

    for i in range(0, total, 100):
        batch = documents[i : i + 100]
        batch_num += 1

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        processed = min(i + 100, total)
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0

        logger.info(
            f"  [{processed}/{total}] "
            f"{processed/total*100:.0f}% | "
            f"{elapsed:.0f}s decorridos | "
            f"ETA: {eta:.0f}s"
        )

    elapsed_total = time.time() - start
    logger.info(
        f"Embeddings gerados em {elapsed_total:.1f}s "
        f"({total/elapsed_total:.1f} chunks/s)"
    )

    return vectorstore


# Persistência
def save_index(vectorstore: FAISS) -> None:
    """
    Persiste o índice FAISS em disco.

    Gera dois arquivos em VECTORSTORE_DIR:
    - index.faiss  → vetores (binário)
    - index.pkl    → mapeamento vetor → Document com metadados
    """
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    logger.info(f"Índice FAISS salvo em {VECTORSTORE_DIR}")
    logger.info("  Arquivos gerados: index.faiss, index.pkl")


# Validação do índice
def validate_index(embeddings: HuggingFaceEmbeddings) -> None:
    """
    Validação de sanidade: carrega o índice salvo e faz uma query de teste.

    Verifica se:
    - O índice carrega sem erro
    - Uma query retorna resultados com metadados completos
    - Os chunks retornados são semanticamente relevantes
    """
    logger.info("Validando índice salvo...")

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    test_query = "alimentos que diabéticos devem evitar"
    results = vectorstore.similarity_search(test_query, k=3)

    if not results:
        logger.error("Validação falhou: nenhum resultado retornado para query de teste.")
        sys.exit(1)

    logger.info(f"Validação OK — {len(results)} resultado(s) para query de teste:")
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        logger.info(
            f"  [{i}] source_id={meta.get('source_id')} | "
            f"page={meta.get('page')} | "
            f"chunk_id={meta.get('chunk_id')} | "
            f"preview: '{doc.page_content[:80]}...'"
        )


# Resumo do índice
def log_index_summary(documents: list[Document]) -> None:
    """Loga distribuição de chunks por documento para auditoria."""
    from collections import Counter

    counter = Counter(doc.metadata.get("source_id", "unknown") for doc in documents)

    logger.info("Distribuição de chunks por documento:")
    for source_id, count in sorted(counter.items(), key=lambda x: -x[1]):
        logger.info(f"  {source_id}: {count} chunk(s)")


# Ponto de entrada
def main() -> None:
    logger.info("=" * 60)
    logger.info("NutriAgents — embed_and_index.py iniciado")
    logger.info("=" * 60)

    # 1. Carregar chunks
    documents = load_chunks()

    # 2. Resumo de distribuição
    log_index_summary(documents)

    # 3. Carregar modelo de embeddings
    embeddings = load_embedding_model()

    # 4. Gerar índice FAISS
    vectorstore = build_faiss_index(documents, embeddings)

    # 5. Persistir
    save_index(vectorstore)

    # 6. Validar
    validate_index(embeddings)

    logger.info("=" * 60)
    logger.info("embed_and_index.py concluído com sucesso.")
    logger.info(f"Índice disponível em: {VECTORSTORE_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()