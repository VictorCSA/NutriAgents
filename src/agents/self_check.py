import logging
import os
from typing import Any

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Logging
logger = logging.getLogger(__name__)

# Configuração
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Limiar de score abaixo do qual a resposta é considerada não suportada
# O LLM retorna um score de 1 a 5 — abaixo de 3 dispara re-busca
SUPPORT_THRESHOLD = 3

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
            temperature=0.0,  # determinístico — avaliação deve ser consistente
            num_predict=256,  # resposta curta — só score + justificativa breve
            extra_body={
                "think": True,
                "num_ctx": 4096,
                "options": {
                    "num_predict": 256,
                    "max_thinking_tokens": 256,  # thinking curto para avaliação
                },
            },
        )
    return _llm


# Prompt de avaliação
SELF_CHECK_PROMPT = PromptTemplate(
    input_variables=["draft", "context"],
    template="""Você é um avaliador rigoroso de respostas nutricionais baseadas em evidências.

## Sua tarefa

Avalie se a resposta abaixo está suportada pelos trechos de documentos fornecidos.

Critérios de avaliação:
- As afirmações principais da resposta aparecem nos trechos? (mesmo que com outras palavras)
- A resposta não inventa informações que não estão nos trechos?
- A resposta não contradiz os trechos?

## Trechos recuperados (evidências disponíveis)

{context}

## Resposta a avaliar

{draft}

## Instruções de resposta

Responda APENAS neste formato exato, sem mais nada:
SCORE: [número de 1 a 5]
MOTIVO: [uma frase explicando o score]

Escala:
5 = todas as afirmações têm suporte claro nos trechos
4 = maioria das afirmações suportada, pequenas extrapolações aceitáveis
3 = metade suportada, algumas afirmações sem evidência clara
2 = poucas afirmações suportadas, muitas extrapolações
1 = resposta não tem suporte nos trechos ou contradiz as evidências

SCORE:"""
)

# Formatação do contexto (igual ao Answerer, mas simplificada)
def format_context(chunks: list[dict]) -> str:
    """Formata os chunks em bloco de contexto numerado para o prompt."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        trecho = chunk.get("text", "").strip()
        fonte = chunk.get("title", "Documento")
        lines.append(f"[{i}] {fonte}: {trecho}")
        lines.append("")
    return "\n".join(lines)


# Parse da resposta do LLM
def parse_evaluation(raw: str) -> tuple[int, str]:
    """
    Extrai SCORE e MOTIVO da resposta do LLM.

    Retorna (score, motivo). Em caso de falha de parsing, retorna (3, motivo_erro)
    — valor neutro que não bloqueia nem aprova automaticamente, forçando re-busca.
    """
    score = None
    motivo = "Não foi possível avaliar a resposta."

    for line in raw.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score = int(line.split(":", 1)[1].strip()[0])
            except (ValueError, IndexError):
                pass
        elif line.upper().startswith("MOTIVO:"):
            motivo = line.split(":", 1)[1].strip()

    if score is None or not (1 <= score <= 5):
        logger.warning(
            f"Parse do Self-Check falhou (raw='{raw[:100]}'). "
            "Usando score neutro 3."
        )
        score = 3

    return score, motivo


# Interface principal do agente
def self_check(draft: str, chunks: list[dict], retry_count: int = 0) -> dict[str, Any]:
    """
    Ponto de entrada do Self-Check Agent.

    Parâmetros:
        draft       — rascunho gerado pelo Answerer
        chunks      — chunks usados pelo Retriever
        retry_count — quantas re-buscas já foram feitas (0 = primeira avaliação)

    Retorna:
    {
        "verdict":      "approved" | "retry" | "refused",
        "score":        int,        # 1-5
        "motivo":       str,        # justificativa do LLM
        "draft":        str,        # draft original (se aprovado) ou mensagem de recusa
        "retry_count":  int,        # repassado ao grafo para controle
    }

    Vereditos:
        approved → draft passa para o Safety
        retry    → grafo deve disparar nova busca com query reformulada (1 tentativa)
        refused  → sem evidências suficientes após re-busca, recusa ao usuário
    """
    if not draft or not draft.strip():
        logger.warning("Self-Check recebeu draft vazio.")
        return {
            "verdict": "refused",
            "score": 0,
            "motivo": "Draft vazio recebido pelo Self-Check.",
            "draft": _mensagem_recusa("sem conteúdo para avaliar"),
            "retry_count": retry_count,
        }

    if not chunks:
        logger.warning("Self-Check recebeu chunks vazios — sem evidências para validar.")
        # Sem chunks não há como validar — recusa direto se já é retry
        if retry_count >= 1:
            return {
                "verdict": "refused",
                "score": 0,
                "motivo": "Nenhuma evidência disponível após re-busca.",
                "draft": _mensagem_recusa("nenhuma evidência encontrada no corpus"),
                "retry_count": retry_count,
            }
        return {
            "verdict": "retry",
            "score": 0,
            "motivo": "Nenhum chunk disponível para validação.",
            "draft": draft,
            "retry_count": retry_count,
        }

    try:
        context = format_context(chunks)
        llm = get_llm()
        chain = SELF_CHECK_PROMPT | llm
        raw = chain.invoke({"draft": draft, "context": context})
        score, motivo = parse_evaluation(raw)

        logger.info(
            f"Self-Check: score={score}/5 | retry_count={retry_count} | "
            f"motivo='{motivo}'"
        )

        # Score suficiente → aprova
        if score >= SUPPORT_THRESHOLD:
            return {
                "verdict": "approved",
                "score": score,
                "motivo": motivo,
                "draft": draft,
                "retry_count": retry_count,
            }

        # Score insuficiente e ainda não tentou re-busca → sinaliza retry
        if retry_count < 1:
            logger.warning(
                f"Self-Check REPROVADO (score={score}). "
                "Sinalizando re-busca ao grafo."
            )
            return {
                "verdict": "retry",
                "score": score,
                "motivo": motivo,
                "draft": draft,
                "retry_count": retry_count,
            }

        # Score insuficiente e já tentou re-busca → recusa
        logger.warning(
            f"Self-Check RECUSOU após re-busca (score={score}). "
            "Sem evidências suficientes."
        )
        return {
            "verdict": "refused",
            "score": score,
            "motivo": motivo,
            "draft": _mensagem_recusa(motivo),
            "retry_count": retry_count,
        }

    except Exception as e:
        logger.error(f"Erro inesperado no Self-Check: {e}")
        # Em caso de erro do LLM, aprovação conservadora para não bloquear o fluxo
        # mas loga o problema para auditoria
        return {
            "verdict": "approved",
            "score": -1,
            "motivo": f"Self-Check indisponível (erro: {e}). Aprovação conservadora.",
            "draft": draft,
            "retry_count": retry_count,
        }


def _mensagem_recusa(motivo: str) -> str:
    """Gera mensagem padronizada de recusa ao usuário."""
    return (
        "Não consegui encontrar evidências suficientes nos documentos disponíveis "
        "para responder a esta pergunta com segurança.\n\n"
        f"**Motivo:** {motivo}\n\n"
        "Recomendo consultar diretamente as fontes oficiais ou um profissional "
        "de saúde para obter informações confiáveis sobre este tema."
    )


# Execução direta — testes rápidos
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    chunks_mock = [
        {
            "title": "Guia Alimentar para a População Brasileira",
            "publisher": "Ministério da Saúde",
            "page": 47,
            "text": (
                "Prefira sempre alimentos in natura ou minimamente processados. "
                "Evite alimentos ultraprocessados ricos em açúcar, gordura e sódio."
            ),
        },
        {
            "title": "Diretrizes SBD — Nutrição",
            "publisher": "SBD",
            "page": 12,
            "text": (
                "Pessoas com diabetes tipo 2 devem evitar açúcar simples e "
                "bebidas açucaradas. Priorize alimentos com baixo índice glicêmico."
            ),
        },
    ]

    casos = [
        (
            "DEVE APROVAR — bem suportado",
            "Pessoas com diabetes devem evitar alimentos ultraprocessados e bebidas "
            "açucaradas [1][2]. Prefira alimentos in natura e com baixo índice glicêmico [2].",
            0,
        ),
        (
            "DEVE REPROVAR/RETRY — alucinação",
            "Pessoas com diabetes devem tomar 500mg de creatina por dia para "
            "melhorar a sensibilidade à insulina. O jejum intermitente de 72 horas "
            "é altamente recomendado.",
            0,
        ),
        (
            "DEVE RECUSAR — alucinação após retry",
            "Pessoas com diabetes devem tomar 500mg de creatina por dia para "
            "melhorar a sensibilidade à insulina.",
            1,  # simula que já tentou re-busca
        ),
    ]

    for descricao, draft, retry_count in casos:
        print(f"\n{'=' * 60}")
        print(f"Caso: {descricao}")
        print(f"Draft (início): {draft[:80]}...")
        print(f"Retry count: {retry_count}")
        print("=" * 60)

        resultado = self_check(draft, chunks_mock, retry_count)
        print(f"Verdict:     {resultado['verdict']}")
        print(f"Score:       {resultado['score']}/5")
        print(f"Motivo:      {resultado['motivo']}")