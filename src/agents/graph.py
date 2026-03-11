import logging
import os
import json
from typing import Any, Literal

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from supervisor import classify
from retriever import retrieve
from answerer import answer
from self_check import self_check
from safety import check as safety_check

# Logging

logger = logging.getLogger(__name__)

# Estado compartilhado do grafo
class GraphState(TypedDict):
    # Entrada
    message: str                        # mensagem original do usuário

    # Supervisor
    intent: str                         # "qa" | "automation" | "refuse"
    intent_motivo: str                  # justificativa da classificação

    # Retriever
    chunks: list[dict]                  # chunks recuperados do vectorstore
    retriever_status: str               # "ok" | "empty" | "error"

    # Answerer
    draft: str                          # rascunho gerado pelo Answerer
    references: str                     # seção de referências
    answerer_status: str                # "ok" | "no_evidence" | "error"

    # Self-Check
    self_check_verdict: str             # "approved" | "retry" | "refused"
    self_check_score: int               # 1-5
    self_check_motivo: str              # justificativa do Self-Check
    retry_count: int                    # número de re-buscas realizadas

    # Safety
    safety_status: str                  # "approved" | "approved_with_disclaimer" | "blocked"
    safety_reasons: list[str]           # gatilhos detectados

    # Saída final
    final_response: str                 # resposta entregue ao usuário


# Utilitário de log de estado
def log_state(node_name: str, state: GraphState) -> None:
    """Loga o estado completo após cada transição de nó."""
    logger.info(f"{'═' * 60}")
    logger.info(f"  NÓ CONCLUÍDO: {node_name.upper()}")
    logger.info(f"{'═' * 60}")

    # Campos relevantes por nó
    campos = {
        "supervisor": ["message", "intent", "intent_motivo"],
        "retriever": ["retriever_status", "retry_count",
                      "_chunks_count"],
        "answerer": ["answerer_status", "_draft_len"],
        "self_check": ["self_check_verdict", "self_check_score",
                       "self_check_motivo", "retry_count"],
        "safety": ["safety_status", "safety_reasons",
                   "_final_response_len"],
    }

    # Campos calculados
    state["_chunks_count"] = len(state.get("chunks") or [])
    state["_draft_len"] = len(state.get("draft") or "")
    state["_final_response_len"] = len(state.get("final_response") or "")

    for campo in campos.get(node_name, list(state.keys())):
        if campo.startswith("_"):
            valor = state.get(campo, "N/A")
        else:
            valor = state.get(campo, "N/A")
            if isinstance(valor, str) and len(valor) > 120:
                valor = valor[:120] + "..."
            elif isinstance(valor, list) and len(valor) > 3:
                valor = valor[:3] + [f"... +{len(valor)-3} itens"]
        logger.info(f"  {campo:<28} = {valor}")

    # Limpa campos temporários
    for tmp in ["_chunks_count", "_draft_len", "_final_response_len"]:
        state.pop(tmp, None)


# Nós do grafo
def node_supervisor(state: GraphState) -> GraphState:
    """Classifica a intenção da mensagem do usuário."""
    logger.info(f"{'─' * 60}")
    logger.info("  → ENTRANDO: supervisor")

    result = classify(state["message"])

    state["intent"] = result["intent"]
    state["intent_motivo"] = result["motivo"]

    # Se for recusa, já define a resposta final
    if result["intent"] == "refuse":
        state["final_response"] = result["response"]

    log_state("supervisor", state)
    return state


def node_retriever(state: GraphState) -> GraphState:
    """Busca chunks relevantes no vectorstore."""
    logger.info(f"{'─' * 60}")
    logger.info("  → ENTRANDO: retriever")

    result = retrieve(state["message"])

    state["chunks"] = result.get("chunks", [])
    state["retriever_status"] = result.get("status", "error")

    log_state("retriever", state)
    return state


def node_answerer(state: GraphState) -> GraphState:
    """Gera o rascunho de resposta com citações."""
    logger.info(f"{'─' * 60}")
    logger.info("  → ENTRANDO: answerer")

    result = answer(state["message"], state["chunks"])

    state["draft"] = result.get("draft", "")
    state["references"] = result.get("references", "")
    state["answerer_status"] = result.get("status", "error")

    log_state("answerer", state)
    return state


def node_self_check(state: GraphState) -> GraphState:
    """Valida se o draft está suportado pelos chunks."""
    logger.info(f"{'─' * 60}")
    logger.info("  → ENTRANDO: self_check")

    result = self_check(
        draft=state["draft"],
        chunks=state["chunks"],
        retry_count=state.get("retry_count", 0),
    )

    state["self_check_verdict"] = result["verdict"]
    state["self_check_score"] = result["score"]
    state["self_check_motivo"] = result["motivo"]
    state["retry_count"] = result["retry_count"]

    # Se recusado, define resposta final
    if result["verdict"] == "refused":
        state["final_response"] = result["draft"]

    log_state("self_check", state)
    return state


def node_safety(state: GraphState) -> GraphState:
    """Aplica política de segurança e adiciona disclaimers."""
    logger.info(f"{'─' * 60}")
    logger.info("  → ENTRANDO: safety")

    result = safety_check(state["draft"])

    state["safety_status"] = result["status"]
    state["safety_reasons"] = result["reasons"]
    state["final_response"] = result["response"]

    log_state("safety", state)
    return state


# Arestas condicionais
def route_supervisor(state: GraphState) -> Literal["retriever", "__end__"]:
    """
    Após o supervisor:
        qa         → retriever
        refuse     → END (resposta já definida)
        automation → END temporário (rota não implementada ainda)
    """
    intent = state.get("intent", "refuse")

    if intent == "qa":
        logger.info("  ↳ Rota: supervisor → retriever")
        return "retriever"

    if intent == "automation":
        logger.info("  ↳ Rota: supervisor → END (automation não implementado)")
        state["final_response"] = (
            "A geração de planos alimentares ainda está sendo implementada. "
            "Por enquanto, posso responder perguntas sobre nutrição e restrições alimentares."
        )
        return "__end__"

    logger.info("  ↳ Rota: supervisor → END (refuse)")
    return "__end__"


def route_self_check(state: GraphState) -> Literal["safety", "retriever", "__end__"]:
    """
    Após o self_check:
        approved → safety
        retry    → retriever (re-busca, incrementa retry_count)
        refused  → END (resposta de recusa já definida)
    """
    verdict = state.get("self_check_verdict", "refused")

    if verdict == "approved":
        logger.info("  ↳ Rota: self_check → safety")
        return "safety"

    if verdict == "retry":
        state["retry_count"] = state.get("retry_count", 0) + 1
        logger.info(
            f"  ↳ Rota: self_check → retriever "
            f"(retry #{state['retry_count']})"
        )
        return "retriever"

    logger.info("  ↳ Rota: self_check → END (refused)")
    return "__end__"


# Estado inicial padrão
def initial_state(message: str) -> GraphState:
    """Cria o estado inicial com todos os campos em seus valores padrão."""
    return GraphState(
        message=message,
        intent="",
        intent_motivo="",
        chunks=[],
        retriever_status="",
        draft="",
        references="",
        answerer_status="",
        self_check_verdict="",
        self_check_score=0,
        self_check_motivo="",
        retry_count=0,
        safety_status="",
        safety_reasons=[],
        final_response="",
    )


# Construção do grafo
def build_graph() -> StateGraph:
    """Constrói e compila o grafo LangGraph."""
    graph = StateGraph(GraphState)

    # Registra os nós
    graph.add_node("supervisor", node_supervisor)
    graph.add_node("retriever", node_retriever)
    graph.add_node("answerer", node_answerer)
    graph.add_node("self_check", node_self_check)
    graph.add_node("safety", node_safety)

    # Ponto de entrada
    graph.set_entry_point("supervisor")

    # Arestas fixas
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", "self_check")
    graph.add_edge("safety", END)

    # Arestas condicionais
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "retriever": "retriever",
            "__end__": END,
        },
    )
    graph.add_conditional_edges(
        "self_check",
        route_self_check,
        {
            "safety": "safety",
            "retriever": "retriever",
            "__end__": END,
        },
    )

    return graph.compile()


# Interface pública
_graph = None


def get_graph():
    """Retorna o grafo compilado (singleton)."""
    global _graph
    if _graph is None:
        logger.info("Compilando grafo LangGraph...")
        _graph = build_graph()
        logger.info("Grafo compilado.")
    return _graph


def run(message: str) -> dict[str, Any]:
    """
    Ponto de entrada público do grafo.

    Recebe a mensagem do usuário e retorna o estado final com a resposta.

    Retorna:
    {
        "final_response": str,   # resposta ao usuário
        "intent":         str,   # intenção classificada
        "safety_status":  str,   # status do safety
        "self_check_score": int, # score do self-check
        "state":          dict,  # estado completo (para debug)
    }
    """
    graph = get_graph()
    state = initial_state(message)

    logger.info(f"\n{'█' * 60}")
    logger.info(f"  NOVA EXECUÇÃO DO GRAFO")
    logger.info(f"  Mensagem: '{message[:80]}'")
    logger.info(f"{'█' * 60}")

    final_state = graph.invoke(state)

    logger.info(f"\n{'█' * 60}")
    logger.info(f"  EXECUÇÃO CONCLUÍDA")
    logger.info(f"  Intent:        {final_state.get('intent')}")
    logger.info(f"  Self-Check:    {final_state.get('self_check_verdict')} "
                f"(score={final_state.get('self_check_score')})")
    logger.info(f"  Safety:        {final_state.get('safety_status')}")
    logger.info(f"{'█' * 60}\n")

    return {
        "final_response": final_state.get("final_response", ""),
        "intent": final_state.get("intent", ""),
        "safety_status": final_state.get("safety_status", ""),
        "self_check_score": final_state.get("self_check_score", 0),
        "state": final_state,
    }


# Execução direta — teste end-to-end
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    casos = [
        "Quais alimentos um diabético deve evitar?",
        "Me gera um plano alimentar semanal para celíaco com hipertensão",
        "Qual é a capital da França?",
    ]

    for mensagem in casos:
        print(f"\n{'=' * 60}")
        print(f"INPUT: {mensagem}")
        print("=" * 60)
        resultado = run(mensagem)
        print(f"INTENT:        {resultado['intent']}")
        print(f"SAFETY:        {resultado['safety_status']}")
        print(f"SELF-CHECK:    score={resultado['self_check_score']}")
        print(f"\nRESPOSTA:\n{resultado['final_response'][:400]}")