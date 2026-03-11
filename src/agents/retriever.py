import logging
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Configuração de paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = DATA_DIR / "processed" / "vectorstore" / "faiss_index"


# Logging
logger = logging.getLogger(__name__)

# Configuração
TOP_K = 5
EMBEDDING_MODEL = "BAAI/bge-m3"

# Inicialização lazy dos recursos pesados
_embeddings = None
_vectorstore = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Carrega o modelo de embeddings (singleton — carrega uma vez)."""
    global _embeddings
    if _embeddings is None:
        logger.info(f"Carregando modelo de embeddings: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embeddings carregados.")
    return _embeddings


def get_vectorstore() -> FAISS:
    """Carrega o índice FAISS (singleton — carrega uma vez)."""
    global _vectorstore
    if _vectorstore is None:
        if not VECTORSTORE_DIR.exists():
            raise FileNotFoundError(
                f"Índice FAISS não encontrado em {VECTORSTORE_DIR}. "
                "Rode ingest/pipeline.py primeiro."
            )
        logger.info(f"Carregando índice FAISS de {VECTORSTORE_DIR}")
        _vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        logger.info("Índice FAISS carregado.")
    return _vectorstore


# Busca no vectorstore
def search(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """
    Executa a busca densa no FAISS e retorna os top-k chunks com metadados.

    Retorna lista de dicts no formato:
    {
        "chunk_id":   str,
        "source_id":  str,
        "title":      str,
        "publisher":  str,
        "year":       str | int,
        "page":       int,
        "section":    str,
        "text":       str,
        "score":      float,   # distância coseno (menor = mais relevante)
    }
    """
    vs = get_vectorstore()

    # similarity_search_with_score retorna (Document, score)
    results = vs.similarity_search_with_score(query, k=top_k)

    chunks = []
    for doc, score in results:
        meta = doc.metadata
        chunks.append(
            {
                "chunk_id": meta.get("chunk_id", ""),
                "source_id": meta.get("source_id", ""),
                "title": meta.get("title", ""),
                "publisher": meta.get("publisher", ""),
                "year": meta.get("year", ""),
                "page": meta.get("page", 0),
                "section": meta.get("section", ""),
                "text": doc.page_content,
                "score": round(float(score), 4),
            }
        )

    logger.info(
        f"Busca retornou {len(chunks)} chunk(s). "
        f"Scores: {[c['score'] for c in chunks]}"
    )
    return chunks


# Interface principal do agente
def retrieve(query: str) -> dict[str, Any]:
    """
    Ponto de entrada do Retriever Agent.

    Recebe uma query e retorna os chunks relevantes do vectorstore.

    Retorna:
    {
        "query":   str,
        "chunks":  list[dict],   # top-k chunks com metadados
        "status":  "ok" | "empty" | "error",
        "message": str,          # descrição do status
    }
    """
    if not query or not query.strip():
        return {
            "query": query,
            "chunks": [],
            "status": "error",
            "message": "Query vazia recebida pelo Retriever.",
        }

    try:
        chunks = search(query)

        if not chunks:
            logger.warning(f"Nenhum chunk encontrado para query: '{query}'")
            return {
                "query": query,
                "chunks": [],
                "status": "empty",
                "message": "Nenhuma evidência encontrada no corpus para esta pergunta.",
            }

        return {
            "query": query,
            "chunks": chunks,
            "status": "ok",
            "message": f"{len(chunks)} chunk(s) recuperado(s).",
        }

    except FileNotFoundError as e:
        logger.error(f"Vectorstore não encontrado: {e}")
        return {
            "query": query,
            "chunks": [],
            "status": "error",
            "message": str(e),
        }

    except Exception as e:
        logger.error(f"Erro inesperado no Retriever: {e}")
        return {
            "query": query,
            "chunks": [],
            "status": "error",
            "message": f"Erro interno no Retriever: {e}",
        }


# Utilitário — formata chunks para log/debug
def format_chunks_for_log(chunks: list[dict]) -> str:
    """Formata os chunks recuperados de forma legível para debug."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"  [{i}] {chunk['title']} | p.{chunk['page']} | "
            f"score={chunk['score']} | {chunk['text'][:80]}..."
        )
    return "\n".join(lines)


# Execução direta — teste rápido
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_queries = [
        "quais alimentos um diabético deve evitar?",
        "o que celíacos não podem comer?",
        "alimentos com alto teor de sódio para hipertensos",
    ]

    for q in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {q}")
        print("=" * 60)
        result = retrieve(q)
        print(f"Status: {result['status']}")
        if result["chunks"]:
            print(format_chunks_for_log(result["chunks"]))
        else:
            print(f"  → {result['message']}")