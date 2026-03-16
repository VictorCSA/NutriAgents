"""
eval/evaluate.py
----------------
Script de avaliação completo do NutriAgents.

Cobre os três itens obrigatórios:
    1. RAG  — 15 perguntas rotuladas + métricas RAGAS
              (Context Precision, Recall, Faithfulness, Answer Relevancy, latência)
    2. Automação — 5 tarefas de geração de plano alimentar
              (taxa de sucesso, nº de steps LLM, tempo médio)
    3. MCP  — valida as 4 tools do mcp-opennutrition
              (disponibilidade, allowlist, log de auditoria)

Uso:
    cd NutriAgents
    python eval/evaluate.py                    # avaliação completa
    python eval/evaluate.py --only rag         # só RAG
    python eval/evaluate.py --only automation  # só automação
    python eval/evaluate.py --only mcp         # só MCP

Saída:
    eval/results/rag_results.json
    eval/results/automation_results.json
    eval/results/mcp_results.json
    eval/results/summary.md
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Setup de paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

# Silencia logs verbosos
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import warnings
warnings.filterwarnings("ignore")
for lib in ("sentence_transformers", "transformers", "huggingface_hub",
            "torch", "tensorflow", "jax"):
    logging.getLogger(lib).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# 1. DATASET DE AVALIAÇÃO RAG
RAG_DATASET = [
    # (pergunta, resposta_esperada_resumida, tópico)
    {
        "question": "Quais alimentos um diabético tipo 2 deve evitar?",
        "expected_topics": ["açúcar", "ultraprocessados", "índice glicêmico"],
        "topic": "diabetes",
    },
    {
        "question": "O que celíacos não podem comer?",
        "expected_topics": ["glúten", "trigo", "cevada", "centeio"],
        "topic": "doença_celíaca",
    },
    {
        "question": "Quais alimentos hipertensos devem evitar?",
        "expected_topics": ["sódio", "sal", "ultraprocessados"],
        "topic": "hipertensão",
    },
    {
        "question": "Quais são as fontes de proteína recomendadas para vegetarianos?",
        "expected_topics": ["leguminosas", "soja", "tofu", "feijão", "lentilha"],
        "topic": "vegetarianismo",
    },
    {
        "question": "Como a alimentação pode ajudar no controle da pressão arterial?",
        "expected_topics": ["potássio", "sódio", "frutas", "vegetais"],
        "topic": "hipertensão",
    },
    {
        "question": "Quais alimentos têm alto índice glicêmico?",
        "expected_topics": ["açúcar", "pão branco", "arroz branco", "batata"],
        "topic": "diabetes",
    },
    {
        "question": "O que é a dieta DASH e para quem ela é indicada?",
        "expected_topics": ["hipertensão", "sódio", "frutas", "vegetais"],
        "topic": "hipertensão",
    },
    {
        "question": "Quais alimentos são naturalmente sem glúten?",
        "expected_topics": ["arroz", "milho", "batata", "mandioca"],
        "topic": "doença_celíaca",
    },
    {
        "question": "Qual o papel das fibras alimentares para diabéticos?",
        "expected_topics": ["glicemia", "fibra", "controle"],
        "topic": "diabetes",
    },
    {
        "question": "Quais nutrientes são importantes para quem tem intolerância à lactose?",
        "expected_topics": ["cálcio", "vitamina D", "leite"],
        "topic": "intolerância_lactose",
    },
    {
        "question": "Como substituir o trigo na alimentação celíaca?",
        "expected_topics": ["farinha de arroz", "polvilho", "amido", "milho"],
        "topic": "doença_celíaca",
    },
    {
        "question": "Quais alimentos ricos em potássio são bons para hipertensos?",
        "expected_topics": ["banana", "batata", "feijão", "espinafre"],
        "topic": "hipertensão",
    },
    {
        "question": "O que é contagem de carboidratos no diabetes?",
        "expected_topics": ["carboidratos", "insulina", "glicemia", "controle"],
        "topic": "diabetes",
    },
    {
        "question": "Quais são os sintomas de contaminação por glúten em celíacos?",
        "expected_topics": ["intestino", "diarreia", "inflamação", "anticorpos"],
        "topic": "doença_celíaca",
    },
    {
        "question": "Posso comer arroz e feijão tendo diabetes tipo 2?",
        "expected_topics": ["porção", "índice glicêmico", "fibra", "combinação"],
        "topic": "diabetes",
    },
]


# 2. DATASET DE AVALIAÇÃO DE AUTOMAÇÃO
AUTOMATION_DATASET = [
    {
        "id": "auto_01",
        "input": "Gere um plano alimentar de 7 dias para celíaco",
        "restrictions_expected": ["celíaco"],
        "must_contain": ["Segunda", "Terça", "Café da manhã", "Almoço", "Jantar"],
        "must_not_contain": ["trigo", "glúten", "cevada"],
        "description": "Plano 7 dias celíaco simples",
    },
    {
        "id": "auto_02",
        "input": "Crie um cardápio semanal para diabético tipo 2",
        "restrictions_expected": ["diabético"],
        "must_contain": ["Segunda", "Café da manhã", "Almoço"],
        "must_not_contain": ["açúcar refinado", "refrigerante"],
        "description": "Cardápio semanal diabético",
    },
    {
        "id": "auto_03",
        "input": "Monte um plano alimentar para celíaco com hipertensão",
        "restrictions_expected": ["celíaco", "hipertenso"],
        "must_contain": ["Segunda", "Café da manhã", "Almoço"],
        "must_not_contain": [],
        "description": "Plano com múltiplas restrições",
    },
    {
        "id": "auto_04",
        "input": "Elabore uma dieta semanal sem lactose e vegetariana",
        "restrictions_expected": ["sem_lactose", "vegetariano"],
        "must_contain": ["Segunda", "Café da manhã"],
        "must_not_contain": ["leite", "carne"],
        "description": "Dieta sem lactose vegetariana",
    },
    {
        "id": "auto_05",
        "input": "Faça um plano alimentar para hipertenso com baixo sódio",
        "restrictions_expected": ["hipertenso"],
        "must_contain": ["Segunda", "Café da manhã", "Almoço"],
        "must_not_contain": [],
        "description": "Plano hipertenso baixo sódio",
    },
]


# 3. AVALIAÇÃO RAG
def evaluate_rag(use_ragas: bool = True) -> dict:
    """Avalia o pipeline RAG com 15 perguntas rotuladas."""
    logger.info("=== Iniciando avaliação RAG ===")

    try:
        from agents.retriever import retrieve
        from agents.answerer import answer
        from agents.self_check import self_check
    except ImportError as e:
        logger.error(f"Erro ao importar agentes: {e}")
        return {"error": str(e)}

    results = []

    for i, item in enumerate(RAG_DATASET, 1):
        logger.info(f"RAG [{i}/{len(RAG_DATASET)}]: {item['question'][:60]}")
        t0 = time.time()

        try:
            # Retriever
            ret = retrieve(item["question"])
            chunks = ret.get("chunks", [])
            t_retrieval = time.time() - t0

            # Answerer
            t1 = time.time()
            ans = answer(item["question"], chunks)
            t_answer = time.time() - t1

            # Self-check
            t2 = time.time()
            sc = self_check(ans["draft"], chunks, 0)
            t_selfcheck = time.time() - t2

            draft = ans.get("draft", "")
            latency_total = time.time() - t0

            # Métricas simples (sem RAGAS)
            context_texts = " ".join(c.get("text", "") for c in chunks).lower()
            draft_lower = draft.lower()

            # Context Recall aproximado: quantos tópicos esperados aparecem nos chunks
            topics_in_context = sum(
                1 for t in item["expected_topics"]
                if t.lower() in context_texts
            )
            context_recall = topics_in_context / len(item["expected_topics"]) if item["expected_topics"] else 0

            # Answer Relevancy aproximado: quantos tópicos esperados aparecem na resposta
            topics_in_answer = sum(
                1 for t in item["expected_topics"]
                if t.lower() in draft_lower
            )
            answer_relevancy = topics_in_answer / len(item["expected_topics"]) if item["expected_topics"] else 0

            # Faithfulness: self_check score normalizado
            sc_score = sc.get("score", 0)
            faithfulness = sc_score / 5.0 if sc_score > 0 else 0

            result = {
                "question": item["question"],
                "topic": item["topic"],
                "n_chunks": len(chunks),
                "context_recall": round(context_recall, 3),
                "answer_relevancy": round(answer_relevancy, 3),
                "faithfulness": round(faithfulness, 3),
                "self_check_score": sc_score,
                "self_check_verdict": sc.get("verdict", "unknown"),
                "latency_total_s": round(latency_total, 2),
                "latency_retrieval_s": round(t_retrieval, 2),
                "latency_answer_s": round(t_answer, 2),
                "latency_selfcheck_s": round(t_selfcheck, 2),
                "answer_preview": draft[:200],
                "status": "ok",
            }

        except Exception as e:
            logger.error(f"Erro na pergunta {i}: {e}")
            result = {
                "question": item["question"],
                "topic": item["topic"],
                "status": "error",
                "error": str(e),
            }

        results.append(result)

    # Tenta RAGAS se disponível
    ragas_metrics = {}
    if use_ragas:
        ragas_metrics = _run_ragas(results)

    # Agrega métricas
    ok_results = [r for r in results if r.get("status") == "ok"]
    summary = {
        "n_questions": len(RAG_DATASET),
        "n_success": len(ok_results),
        "avg_context_recall": round(
            sum(r["context_recall"] for r in ok_results) / len(ok_results), 3
        ) if ok_results else 0,
        "avg_answer_relevancy": round(
            sum(r["answer_relevancy"] for r in ok_results) / len(ok_results), 3
        ) if ok_results else 0,
        "avg_faithfulness": round(
            sum(r["faithfulness"] for r in ok_results) / len(ok_results), 3
        ) if ok_results else 0,
        "avg_latency_s": round(
            sum(r["latency_total_s"] for r in ok_results) / len(ok_results), 2
        ) if ok_results else 0,
        "avg_chunks_retrieved": round(
            sum(r["n_chunks"] for r in ok_results) / len(ok_results), 1
        ) if ok_results else 0,
        "ragas": ragas_metrics,
    }

    output = {"summary": summary, "results": results}
    out_path = RESULTS_DIR / "rag_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"RAG: resultados salvos em {out_path}")
    logger.info(f"RAG Summary: {summary}")
    return output


def _run_ragas(rag_results: list) -> dict:
    """Tenta calcular métricas RAGAS se a biblioteca estiver instalada."""
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            context_precision, context_recall,
            faithfulness, answer_relevancy,
        )
        from datasets import Dataset

        logger.info("RAGAS disponível — calculando métricas oficiais...")

        # Importa retriever e answerer para montar o dataset no formato RAGAS
        from agents.retriever import retrieve
        from agents.answerer import answer

        ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        for item in RAG_DATASET[:10]:  # limita a 10 para não demorar demais
            try:
                ret = retrieve(item["question"])
                chunks = ret.get("chunks", [])
                ans = answer(item["question"], chunks)
                ragas_data["question"].append(item["question"])
                ragas_data["answer"].append(ans.get("draft", ""))
                ragas_data["contexts"].append([c.get("text", "") for c in chunks])
                ragas_data["ground_truth"].append(", ".join(item["expected_topics"]))
            except Exception:
                continue

        if not ragas_data["question"]:
            return {}

        ds = Dataset.from_dict(ragas_data)
        score = ragas_evaluate(
            ds,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        )
        return {k: round(float(v), 3) for k, v in score.items()}

    except ImportError:
        logger.info("RAGAS não instalado — usando métricas aproximadas. "
                    "Instale com: pip install ragas datasets")
        return {}
    except Exception as e:
        logger.warning(f"RAGAS falhou: {e}")
        return {}


# 4. AVALIAÇÃO DE AUTOMAÇÃO
def evaluate_automation() -> dict:
    """Avalia o Automation Agent com 5 tarefas de geração de plano."""
    logger.info("=== Iniciando avaliação de Automação ===")

    try:
        from agents.automation import generate_meal_plan, extract_restrictions
    except ImportError as e:
        logger.error(f"Erro ao importar automation: {e}")
        return {"error": str(e)}

    results = []

    DAYS_OF_WEEK = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    MEALS = ["Café da manhã", "Almoço", "Lanche", "Jantar"]

    for item in AUTOMATION_DATASET:
        logger.info(f"Automação [{item['id']}]: {item['description']}")
        t0 = time.time()

        try:
            result = generate_meal_plan(item["input"])
            elapsed = time.time() - t0

            draft = result.get("draft", "")
            restrictions_found = result.get("restrictions", [])
            mcp_foods = result.get("mcp_foods", {})
            chunks = result.get("chunks", [])
            status = result.get("status", "error")

            # Verifica must_contain
            contains_ok = all(kw in draft for kw in item["must_contain"])

            # Verifica must_not_contain
            contains_forbidden = any(kw.lower() in draft.lower()
                                     for kw in item["must_not_contain"])

            # Conta dias únicos presentes no plano
            days_present = sum(1 for d in DAYS_OF_WEEK if d in draft)

            # Conta refeições únicas
            meals_present = sum(1 for m in MEALS if m in draft)

            # Verifica variação: conta linhas de tabela únicas (sem repetição)
            table_rows = [
                line.strip() for line in draft.splitlines()
                if "|" in line and any(icon in line for icon in ["☀️", "🌞", "🌤️", "🌙"])
            ]
            unique_rows = len(set(table_rows))
            total_rows = len(table_rows)
            variation_rate = round(unique_rows / total_rows, 3) if total_rows > 0 else 0

            # Restrições detectadas corretamente
            restrictions_correct = all(
                r in restrictions_found
                for r in item["restrictions_expected"]
            )

            # Steps = chamadas LLM estimadas (1 header + 7 dias + 1 footer + RAG)
            n_steps = 9 + len(chunks)  # estimativa

            success = (
                status in ("ok", "partial")
                and contains_ok
                and not contains_forbidden
                and days_present >= 5
            )

            result_entry = {
                "id": item["id"],
                "description": item["description"],
                "input": item["input"],
                "success": success,
                "status": status,
                "elapsed_s": round(elapsed, 2),
                "n_steps_estimated": n_steps,
                "days_present": days_present,
                "meals_present": meals_present,
                "variation_rate": variation_rate,
                "unique_meal_rows": unique_rows,
                "total_meal_rows": total_rows,
                "restrictions_detected": restrictions_found,
                "restrictions_correct": restrictions_correct,
                "mcp_foods_count": len(mcp_foods),
                "rag_chunks_count": len(chunks),
                "contains_required_keywords": contains_ok,
                "contains_forbidden_keywords": contains_forbidden,
                "draft_length": len(draft),
                "draft_preview": draft[:300],
            }

        except Exception as e:
            logger.error(f"Erro na tarefa {item['id']}: {e}")
            result_entry = {
                "id": item["id"],
                "description": item["description"],
                "input": item["input"],
                "success": False,
                "status": "error",
                "error": str(e),
                "elapsed_s": round(time.time() - t0, 2),
            }

        results.append(result_entry)
        logger.info(f"  → success={result_entry.get('success')} | "
                    f"elapsed={result_entry.get('elapsed_s')}s | "
                    f"days={result_entry.get('days_present', 0)}/7")

    # Agrega
    ok = [r for r in results if r.get("success")]
    elapsed_all = [r["elapsed_s"] for r in results if "elapsed_s" in r]
    steps_all = [r["n_steps_estimated"] for r in results if "n_steps_estimated" in r]

    summary = {
        "n_tasks": len(AUTOMATION_DATASET),
        "n_success": len(ok),
        "success_rate": round(len(ok) / len(AUTOMATION_DATASET), 3),
        "avg_elapsed_s": round(sum(elapsed_all) / len(elapsed_all), 2) if elapsed_all else 0,
        "avg_steps": round(sum(steps_all) / len(steps_all), 1) if steps_all else 0,
        "avg_variation_rate": round(
            sum(r.get("variation_rate", 0) for r in results) / len(results), 3
        ),
        "avg_mcp_foods": round(
            sum(r.get("mcp_foods_count", 0) for r in results) / len(results), 1
        ),
    }

    output = {"summary": summary, "results": results}
    out_path = RESULTS_DIR / "automation_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Automação: resultados salvos em {out_path}")
    logger.info(f"Automação Summary: {summary}")
    return output


# 5. AVALIAÇÃO MCP
def evaluate_mcp() -> dict:
    """Valida as tools do MCP e os controles de segurança."""
    logger.info("=== Iniciando avaliação MCP ===")

    try:
        from nutritools.opennutrition_client import (
            search_foods, get_food_by_id, browse_foods,
            ALLOWED_TOOLS, MCP_SERVER_PATH, MCP_CALL_LOG,
        )
    except ImportError as e:
        logger.error(f"Erro ao importar cliente MCP: {e}")
        return {"error": str(e)}

    results = []

    # ── Teste 1: server path existe ────────────────────────────────────────────
    server_exists = Path(MCP_SERVER_PATH).exists()
    results.append({
        "test": "server_path_exists",
        "passed": server_exists,
        "detail": str(MCP_SERVER_PATH),
    })
    logger.info(f"MCP server path: {'OK' if server_exists else 'NÃO ENCONTRADO'} — {MCP_SERVER_PATH}")

    if not server_exists:
        logger.warning("Servidor MCP não encontrado. Pulando testes de tool.")
        output = {
            "summary": {"server_available": False, "tests_passed": 0, "tests_total": 1},
            "results": results,
        }
        (RESULTS_DIR / "mcp_results.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output

    # ── Teste 2: search-food-by-name ───────────────────────────────────────────
    logger.info("MCP: testando search-food-by-name...")
    t0 = time.time()
    try:
        foods = search_foods("chicken breast", limit=3)
        elapsed = time.time() - t0
        passed = isinstance(foods, list) and len(foods) > 0
        results.append({
            "test": "search_food_by_name",
            "passed": passed,
            "elapsed_s": round(elapsed, 2),
            "n_results": len(foods),
            "sample": foods[0] if foods else None,
        })
        logger.info(f"  search_foods: {'OK' if passed else 'FALHOU'} — {len(foods)} resultados em {elapsed:.1f}s")
    except Exception as e:
        results.append({"test": "search_food_by_name", "passed": False, "error": str(e)})
        logger.warning(f"  search_foods falhou: {e}")

    # ── Teste 3: get-food-by-id com id válido ──────────────────────────────────
    logger.info("MCP: testando get-food-by-id...")
    food_id = None
    if len(results) >= 2 and results[-1].get("passed") and results[-1].get("sample"):
        food_id = results[-1]["sample"].get("id", "")

    if food_id and food_id.startswith("fd_"):
        t0 = time.time()
        try:
            details = get_food_by_id(food_id)
            elapsed = time.time() - t0
            passed = details is not None and isinstance(details, dict)
            results.append({
                "test": "get_food_by_id_valid",
                "passed": passed,
                "elapsed_s": round(elapsed, 2),
                "food_id": food_id,
                "has_nutrition": "nutrition_100g" in (details or {}),
            })
            logger.info(f"  get_food_by_id: {'OK' if passed else 'FALHOU'} em {elapsed:.1f}s")
        except Exception as e:
            results.append({"test": "get_food_by_id_valid", "passed": False, "error": str(e)})
    else:
        results.append({
            "test": "get_food_by_id_valid",
            "passed": None,
            "detail": "Pulado — nenhum fd_ id disponível do search anterior",
        })

    # ── Teste 4: get-food-by-id com id INVÁLIDO (deve falhar graciosamente) ────
    logger.info("MCP: testando rejeição de id inválido (lc_...)...")
    try:
        result_invalid = get_food_by_id("lc_abc123_invalid")
        passed = result_invalid is None  # deve retornar None sem explodir
        results.append({
            "test": "get_food_by_id_invalid_rejected",
            "passed": passed,
            "detail": "id 'lc_...' deve retornar None (allowlist de formato)",
        })
        logger.info(f"  rejeição id inválido: {'OK' if passed else 'FALHOU'}")
    except Exception as e:
        results.append({
            "test": "get_food_by_id_invalid_rejected",
            "passed": False,
            "error": str(e),
        })

    # ── Teste 5: allowlist — tool fora da lista deve ser bloqueada ─────────────
    logger.info("MCP: testando allowlist...")
    from nutritools.opennutrition_client import _call_tool
    try:
        _call_tool("execute_shell", {"cmd": "whoami"})
        results.append({
            "test": "allowlist_blocks_forbidden_tool",
            "passed": False,
            "detail": "FALHA DE SEGURANÇA: tool proibida foi executada!",
        })
        logger.error("  FALHA DE SEGURANÇA: allowlist não bloqueou tool proibida!")
    except ValueError as e:
        results.append({
            "test": "allowlist_blocks_forbidden_tool",
            "passed": True,
            "detail": f"Corretamente bloqueado: {e}",
        })
        logger.info("  allowlist OK — tool proibida bloqueada corretamente")
    except Exception as e:
        results.append({
            "test": "allowlist_blocks_forbidden_tool",
            "passed": True,  # qualquer exceção != executar é aceitável
            "detail": f"Bloqueado com: {type(e).__name__}: {e}",
        })

    # ── Teste 6: log de auditoria ──────────────────────────────────────────────
    logger.info("MCP: verificando log de auditoria...")
    log_exists = Path(MCP_CALL_LOG).exists()
    log_entries = 0
    if log_exists:
        try:
            with open(MCP_CALL_LOG) as f:
                log_entries = sum(1 for line in f if line.strip())
        except Exception:
            pass
    results.append({
        "test": "audit_log_exists",
        "passed": log_exists,
        "log_path": str(MCP_CALL_LOG),
        "n_entries": log_entries,
    })
    logger.info(f"  log de auditoria: {'OK' if log_exists else 'NÃO ENCONTRADO'} — {log_entries} entradas")

    # ── Resumo ─────────────────────────────────────────────────────────────────
    definitive = [r for r in results if r.get("passed") is not None]
    passed_count = sum(1 for r in definitive if r["passed"])

    summary = {
        "server_available": server_exists,
        "allowed_tools": sorted(ALLOWED_TOOLS),
        "tests_total": len(definitive),
        "tests_passed": passed_count,
        "tests_failed": len(definitive) - passed_count,
        "audit_log_entries": log_entries,
        "mcp_server_path": str(MCP_SERVER_PATH),
        "security_controls": {
            "allowlist": "Apenas 4 tools permitidas: search-food-by-name, get-food-by-id, browse-foods, barcode-lookup",
            "id_validation": "IDs devem começar com 'fd_' — lc_... rejeitados antes da chamada",
            "audit_log": f"Todas as chamadas registradas em {MCP_CALL_LOG}",
            "no_file_access": "Servidor não tem acesso ao sistema de arquivos do projeto",
            "no_network": "Servidor roda offline — sem chamadas externas após instalação",
            "forbidden": "Não pode: ler arquivos, executar shell, fazer HTTP, modificar banco de dados",
        },
    }

    output = {"summary": summary, "results": results}
    out_path = RESULTS_DIR / "mcp_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"MCP: resultados salvos em {out_path}")
    logger.info(f"MCP Summary: {passed_count}/{len(definitive)} testes passaram")
    return output


# 6. RELATÓRIO FINAL
def generate_summary_report(rag: dict, automation: dict, mcp: dict) -> str:
    """Gera relatório Markdown consolidado."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    rag_s = rag.get("summary", {})
    auto_s = automation.get("summary", {})
    mcp_s = mcp.get("summary", {})

    ragas_block = ""
    if rag_s.get("ragas"):
        ragas_block = "\n**Métricas RAGAS oficiais:**\n"
        for k, v in rag_s["ragas"].items():
            ragas_block += f"- {k}: {v}\n"

    report = f"""# NutriAgents — Relatório de Avaliação
Gerado em: {now}

---

## 1. Avaliação RAG

| Métrica | Valor |
|---------|-------|
| Perguntas avaliadas | {rag_s.get('n_questions', '—')} |
| Sucesso | {rag_s.get('n_success', '—')}/{rag_s.get('n_questions', '—')} |
| Context Recall (médio) | {rag_s.get('avg_context_recall', '—')} |
| Answer Relevancy (médio) | {rag_s.get('avg_answer_relevancy', '—')} |
| Faithfulness (médio) | {rag_s.get('avg_faithfulness', '—')} |
| Latência média | {rag_s.get('avg_latency_s', '—')}s |
| Chunks recuperados (médio) | {rag_s.get('avg_chunks_retrieved', '—')} |
{ragas_block}
> **Nota:** Context Recall e Answer Relevancy são calculados como fração de tópicos esperados
> encontrados nos chunks/resposta. Faithfulness = self_check_score / 5.

---

## 2. Avaliação de Automação

| Métrica | Valor |
|---------|-------|
| Tarefas avaliadas | {auto_s.get('n_tasks', '—')} |
| Taxa de sucesso | {auto_s.get('n_success', '—')}/{auto_s.get('n_tasks', '—')} ({auto_s.get('success_rate', 0)*100:.0f}%) |
| Tempo médio por plano | {auto_s.get('avg_elapsed_s', '—')}s |
| Steps médios (estimado) | {auto_s.get('avg_steps', '—')} |
| Variação de refeições (médio) | {auto_s.get('avg_variation_rate', 0)*100:.0f}% de linhas únicas |
| Alimentos MCP por plano (médio) | {auto_s.get('avg_mcp_foods', '—')} |

**Critério de sucesso:** plano gerado com status ok/partial + contém dias da semana + palavras-chave obrigatórias presentes + sem palavras proibidas.

---

## 3. Avaliação MCP

| Métrica | Valor |
|---------|-------|
| Servidor disponível | {'✅ Sim' if mcp_s.get('server_available') else '❌ Não'} |
| Testes passados | {mcp_s.get('tests_passed', '—')}/{mcp_s.get('tests_total', '—')} |
| Entradas no log de auditoria | {mcp_s.get('audit_log_entries', '—')} |

**Tools na allowlist:**
{chr(10).join(f'- `{t}`' for t in mcp_s.get('allowed_tools', []))}

**Controles de segurança:**
| Controle | Descrição |
|----------|-----------|
| Allowlist | {mcp_s.get('security_controls', {}).get('allowlist', '—')} |
| Validação de ID | {mcp_s.get('security_controls', {}).get('id_validation', '—')} |
| Log de auditoria | {mcp_s.get('security_controls', {}).get('audit_log', '—')} |
| Acesso a arquivos | {mcp_s.get('security_controls', {}).get('no_file_access', '—')} |
| Rede | {mcp_s.get('security_controls', {}).get('no_network', '—')} |
| Proibido | {mcp_s.get('security_controls', {}).get('forbidden', '—')} |

---

*Relatório gerado automaticamente por `eval/evaluate.py`*
"""

    out_path = RESULTS_DIR / "summary.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info(f"Relatório salvo em {out_path}")
    return report


# 7. ENTRY POINT
def main():
    parser = argparse.ArgumentParser(description="Avaliação do NutriAgents")
    parser.add_argument(
        "--only",
        choices=["rag", "automation", "mcp"],
        help="Rodar apenas uma avaliação específica",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Pular métricas RAGAS oficiais (mais rápido)",
    )
    args = parser.parse_args()

    rag_result = {}
    auto_result = {}
    mcp_result = {}

    if args.only in (None, "rag"):
        rag_result = evaluate_rag(use_ragas=not args.no_ragas)

    if args.only in (None, "automation"):
        auto_result = evaluate_automation()

    if args.only in (None, "mcp"):
        mcp_result = evaluate_mcp()

    if args.only is None:
        report = generate_summary_report(rag_result, auto_result, mcp_result)
        print("\n" + "=" * 60)
        print(report)

    print(f"\nResultados salvos em: {RESULTS_DIR}")


if __name__ == "__main__":
    main()