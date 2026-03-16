"""
src/graph/graph.py
------------------
Grafo LangGraph — orquestra todos os agentes do NutriAgents.

Nós:
    supervisor   → classifica a intenção (qa / automation / refuse)
    retriever    → busca chunks no FAISS
    answerer     → gera resposta com citações (rota qa)
    self_check   → valida suporte das afirmações (re-busca se falhar)
    automation   → gera plano alimentar com RAG + MCP (rota automation)
    safety       → adiciona disclaimer / bloqueia conteúdo perigoso

Fluxo QA:
    supervisor → retriever → answerer → self_check → safety → END

Fluxo Automation:
    supervisor → automation → safety → END

Fluxo Refuse:
    supervisor → END
"""

import logging
import re
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from agents.supervisor import classify
from agents.retriever import retrieve
from agents.answerer import answer
from agents.self_check import self_check as run_self_check
from agents.safety import check as safety_check
from agents.automation import generate_meal_plan

logger = logging.getLogger(__name__)


# ── Estado do grafo ───────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # Input
    message: str

    # Supervisor
    intent: str
    motivo: str

    # Retriever
    chunks: list
    retriever_status: str

    # Answerer
    draft: str
    references: str
    answerer_status: str

    # Self-Check
    self_check_score: int
    self_check_motivo: str
    self_check_verdict: str   # "approved" | "retry" | "refused"
    retry_count: int

    # Automation
    restrictions: list
    mcp_foods: dict
    automation_status: str

    # Safety
    safe_response: str
    safety_status: str
    safety_reason: str

    # Output final
    final_response: str


# ── Nós do grafo ──────────────────────────────────────────────────────────────

def node_supervisor(state: AgentState) -> AgentState:
    logger.info(">>> Nó: supervisor")
    result = classify(state["message"])
    return {
        **state,
        "intent":         result["intent"],
        "motivo":         result["motivo"],
        "final_response": result.get("response", ""),
    }


def node_retriever(state: AgentState) -> AgentState:
    logger.info(">>> Nó: retriever")
    result = retrieve(state["message"])
    return {
        **state,
        "chunks":           result["chunks"],
        "retriever_status": result["status"],
    }


def node_answerer(state: AgentState) -> AgentState:
    logger.info(">>> Nó: answerer")
    result = answer(state["message"], state.get("chunks", []))
    return {
        **state,
        "draft":           result["draft"],
        "references":      result["references"],
        "answerer_status": result["status"],
    }


def node_self_check(state: AgentState) -> AgentState:
    logger.info(">>> Nó: self_check")

    draft       = state.get("draft", "")
    chunks      = state.get("chunks", [])
    retry_count = state.get("retry_count", 0)

    # Chama a interface real do self_check.py (retorna verdict/score/motivo/draft)
    result = run_self_check(draft, chunks, retry_count)

    verdict = result.get("verdict", "approved")
    score   = result.get("score", -1)
    motivo  = result.get("motivo", "")

    logger.info(f"Self-check resultado: verdict={verdict} score={score}")

    return {
        **state,
        "self_check_score":   score,
        "self_check_motivo":  motivo,
        "self_check_verdict": verdict,
        "retry_count":        result.get("retry_count", retry_count),
        # Se recusado, já atualiza o draft com a mensagem de recusa
        "draft": result.get("draft", draft),
    }


def node_automation(state: AgentState) -> AgentState:
    logger.info(">>> Nó: automation")
    result = generate_meal_plan(state["message"])
    return {
        **state,
        "draft":             result["draft"],
        "chunks":            result.get("chunks", []),
        "restrictions":      result.get("restrictions", []),
        "mcp_foods":         result.get("mcp_foods", {}),
        "automation_status": result["status"],
    }


def node_safety(state: AgentState) -> AgentState:
    logger.info(">>> Nó: safety")

    draft = state.get("draft") or state.get("final_response") or ""

    DISCLAIMER = (
        "\n\n---\n"
        "> ⚠️ **Aviso importante:** As informações acima têm caráter "
        "**exclusivamente informativo** e são baseadas em documentos públicos de saúde. "
        "**Não substituem consulta com nutricionista, médico ou outro profissional de saúde.**"
    )

    safe = None
    status = "approved_with_disclaimer"
    reason = ""

    # Tenta usar o safety_check importado (aceita diferentes nomes de chave)
    try:
        result = safety_check(draft)
        safe   = result.get("safe_response") or result.get("response")
        status = result.get("safety_status") or result.get("status", "approved_with_disclaimer")
        reason = result.get("safety_reason") or result.get("reason", "")
    except Exception as e:
        logger.warning(f"safety_check indisponível, usando fallback inline: {e}")

    # Fallback inline se safety_check falhou ou retornou vazio
    if not safe:
        if not draft.strip():
            safe, status, reason = "Não foi possível gerar uma resposta.", "blocked", "Draft vazio."
        else:
            safe   = draft + DISCLAIMER
            status = "approved_with_disclaimer"
            reason = "Fallback inline com disclaimer."

    logger.info(f"Safety: status={status}")
    return {
        **state,
        "safe_response":  safe,
        "safety_status":  status,
        "safety_reason":  reason,
        "final_response": safe,
    }


def route_after_supervisor(state: AgentState) -> str:
    intent = state.get("intent", "refuse")
    logger.info(f"Router após supervisor: intent='{intent}'")
    if intent == "qa":
        return "retriever"
    if intent == "automation":
        return "automation"
    return END


def route_after_self_check(state: AgentState) -> str:
    verdict = state.get("self_check_verdict", "approved")
    retry   = state.get("retry_count", 0)

    if verdict == "retry" and retry <= 1:
        # Incrementa retry_count para evitar loop infinito
        logger.info("Self-check → retry no retriever")
        return "retriever"

    # "approved" ou "refused" (draft já substituído pela mensagem de recusa)
    return "safety"


# ── Construção do grafo ───────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("supervisor",  node_supervisor)
    g.add_node("retriever",   node_retriever)
    g.add_node("answerer",    node_answerer)
    g.add_node("self_check",  node_self_check)
    g.add_node("automation",  node_automation)
    g.add_node("safety",      node_safety)

    g.set_entry_point("supervisor")

    g.add_conditional_edges("supervisor", route_after_supervisor, {
        "retriever":  "retriever",
        "automation": "automation",
        END:          END,
    })

    g.add_edge("retriever",  "answerer")
    g.add_edge("answerer",   "self_check")

    g.add_conditional_edges("self_check", route_after_self_check, {
        "safety":    "safety",
        "retriever": "retriever",
    })

    g.add_edge("automation", "safety")
    g.add_edge("safety",     END)

    return g.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Ponto de entrada público ──────────────────────────────────────────────────

def run(message: str) -> dict[str, Any]:
    """
    Executa o pipeline completo para uma mensagem do usuário.

    Retorna:
    {
        "final_response"  : str,
        "intent"          : str,
        "self_check_score": int | None,
        "safety_status"   : str | None,
        "state"           : dict,
    }
    """
    logger.info(f"run() → message='{message[:80]}'")

    graph = get_graph()
    initial_state: AgentState = {
        "message":     message,
        "retry_count": 0,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"Erro no grafo LangGraph: {e}")
        return {
            "final_response": (
                "⚠️ Ocorreu um erro interno. Verifique se o Ollama está rodando "
                f"e tente novamente.\n\nDetalhe: `{e}`"
            ),
            "intent":           "error",
            "self_check_score": None,
            "safety_status":    None,
            "state":            {},
        }

    return {
        "final_response":   final_state.get("final_response", "Sem resposta."),
        "intent":           final_state.get("intent", "—"),
        "self_check_score": final_state.get("self_check_score"),
        "safety_status":    final_state.get("safety_status"),
        "state":            dict(final_state),
    }


# ── Teste direto ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    tests = [
        "Quais alimentos um diabético deve evitar?",
        "Gera um plano alimentar semanal para celíaco com hipertensão",
        "Qual é a capital da França?",
    ]

    for msg in tests:
        print(f"\n{'='*60}")
        print(f"Input: {msg}")
        print("=" * 60)
        result = run(msg)
        print(f"Intent:      {result['intent']}")
        print(f"Self-check:  {result['self_check_score']}")
        print(f"Safety:      {result['safety_status']}")
        print(f"\nResposta:\n{result['final_response'][:400]}...")