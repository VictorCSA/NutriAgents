import logging
import os
from pathlib import Path
from typing import Any

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Logging
logger = logging.getLogger(__name__)

# Configuração
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Singleton LLM
_llm = None


def get_llm() -> OllamaLLM:
    """Carrega o LLM Ollama (singleton — carrega uma vez)."""
    global _llm
    if _llm is None:
        logger.info(f"Conectando ao Ollama: {OLLAMA_MODEL} em {OLLAMA_BASE_URL}")
        _llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,  # leve criatividade para síntese, mas ainda controlado
            num_predict=1024,
        )
    return _llm


# Formatação do contexto para o prompt
def format_context(chunks: list[dict]) -> str:
    """
    Formata os chunks recuperados em bloco de contexto numerado para o prompt.
    Cada chunk recebe um índice [N] que será usado nas citações inline.

    Exemplo de saída:
        [1] Fonte: Guia Alimentar para a População Brasileira, p. 47
        Trecho: "Prefira alimentos in natura ou minimamente processados..."

        [2] Fonte: Diretrizes SBD — Nutrição, p. 12
        Trecho: "O índice glicêmico dos alimentos deve ser considerado..."
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        fonte = f"{chunk.get('title', 'Documento')} — {chunk.get('publisher', '')}"
        if chunk.get("year"):
            fonte += f", {chunk['year']}"
        pagina = chunk.get("page", "?")
        trecho = chunk.get("text", "").strip()

        lines.append(f"[{i}] Fonte: {fonte}, p. {pagina}")
        lines.append(f"Trecho: {trecho}")
        lines.append("")

    return "\n".join(lines)


def format_references(chunks: list[dict]) -> str:
    """
    Gera a seção de referências ao final da resposta.

    Exemplo:
        ## Referências
        [1] Guia Alimentar para a População Brasileira — Ministério da Saúde, 2014. p. 47.
        [2] Diretrizes SBD — Nutrição. p. 12.
    """
    lines = ["## Referências"]
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "Documento")
        publisher = chunk.get("publisher", "")
        year = chunk.get("year", "")
        page = chunk.get("page", "?")
        section = chunk.get("section", "")

        ref = f"[{i}] {title}"
        if publisher:
            ref += f" — {publisher}"
        if year:
            ref += f", {year}"
        ref += f". p. {page}."
        if section:
            ref += f" ({section})"

        lines.append(ref)

    return "\n".join(lines)


# Prompt do Answerer
ANSWERER_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template="""Você é um assistente especializado em nutrição e alimentação saudável, \
baseado em documentos públicos brasileiros de saúde.

## Instruções

Responda à pergunta do usuário usando EXCLUSIVAMENTE as informações dos trechos \
fornecidos abaixo. Não use conhecimento externo.

Formato obrigatório da resposta:
- Use **Markdown**: negrito para termos importantes, listas quando listar itens
- Cite as fontes inline usando o número entre colchetes: [1], [2], etc.
- Seja claro e objetivo
- Se os trechos não forem suficientes para responder, diga explicitamente que \
não encontrou informação suficiente no corpus

## Trechos recuperados

{context}

## Pergunta do usuário

{query}

## Resposta (em Markdown, com citações inline):"""
)


# Interface principal do agente
def answer(query: str, chunks: list[dict]) -> dict[str, Any]:
    """
    Ponto de entrada do Answerer/Writer Agent.

    Recebe a query e os chunks do Retriever e gera a resposta formatada.

    Retorna:
    {
        "draft":      str,   # resposta com citações inline + seção de referências
        "references": str,   # só a seção de referências (para o Self-Check)
        "chunks":     list,  # chunks usados (repassados ao Self-Check)
        "status":     "ok" | "no_evidence" | "error",
        "message":    str,
    }
    """
    if not chunks:
        logger.warning("Answerer recebeu lista de chunks vazia.")
        return {
            "draft": (
                "Não encontrei informações suficientes no corpus para responder "
                "a esta pergunta com segurança."
            ),
            "references": "",
            "chunks": [],
            "status": "no_evidence",
            "message": "Nenhum chunk disponível para síntese.",
        }

    try:
        context = format_context(chunks)
        references = format_references(chunks)

        llm = get_llm()
        chain = ANSWERER_PROMPT | llm
        raw_response = chain.invoke({"query": query, "context": context})
        raw_response = raw_response.strip()

        if not raw_response:
            logger.warning("LLM retornou resposta vazia.")
            return {
                "draft": "Não foi possível gerar uma resposta para esta pergunta.",
                "references": references,
                "chunks": chunks,
                "status": "error",
                "message": "LLM retornou resposta vazia.",
            }

        # Monta o draft completo: corpo + referências
        draft = f"{raw_response}\n\n{references}"

        logger.info(
            f"Answerer gerou resposta: {len(raw_response)} chars | "
            f"{len(chunks)} chunk(s) usado(s)"
        )

        return {
            "draft": draft,
            "references": references,
            "chunks": chunks,
            "status": "ok",
            "message": "Resposta gerada com sucesso.",
        }

    except Exception as e:
        logger.error(f"Erro inesperado no Answerer: {e}")
        return {
            "draft": "Ocorreu um erro ao gerar a resposta. Tente novamente.",
            "references": "",
            "chunks": chunks,
            "status": "error",
            "message": f"Erro interno no Answerer: {e}",
        }


# Execução direta — teste rápido
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Chunks simulados para teste sem precisar do vectorstore
    chunks_mock = [
        {
            "chunk_id": "guia_alimentar_ms_p047_c001",
            "source_id": "guia_alimentar_ms",
            "title": "Guia Alimentar para a População Brasileira",
            "publisher": "Ministério da Saúde",
            "year": 2014,
            "page": 47,
            "section": "Capítulo 3",
            "text": (
                "Prefira sempre alimentos in natura ou minimamente processados. "
                "Óleos, gorduras, sal e açúcar devem ser usados em pequenas quantidades. "
                "Evite alimentos ultraprocessados."
            ),
            "score": 0.12,
        },
        {
            "chunk_id": "diretrizes_sbd_p012_c002",
            "source_id": "diretrizes_sbd",
            "title": "Diretrizes da Sociedade Brasileira de Diabetes — Nutrição",
            "publisher": "SBD",
            "year": 2023,
            "page": 12,
            "section": "Terapia Nutricional",
            "text": (
                "Pessoas com diabetes tipo 2 devem priorizar alimentos com baixo "
                "índice glicêmico, como legumes, verduras e grãos integrais. "
                "O consumo de açúcar simples e bebidas açucaradas deve ser evitado."
            ),
            "score": 0.18,
        },
    ]

    query = "Quais alimentos uma pessoa com diabetes deve evitar?"

    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print("=" * 60)

    resultado = answer(query, chunks_mock)

    print(f"Status: {resultado['status']}")
    print(f"Message: {resultado['message']}")
    print(f"\n--- DRAFT ---\n{resultado['draft']}")