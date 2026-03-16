"""
src/agents/automation.py
------------------------
Automation Agent — gera planos alimentares semanais personalizados.

Fluxo:
    1. Extrai restrições alimentares da mensagem do usuário (supervisor já
       classificou como "automation").
    2. Consulta o RAG (retriever) para obter diretrizes nutricionais
       baseadas em documentos públicos brasileiros.
    3. Chama o MCP mcp-opennutrition para buscar alimentos adequados
       para cada restrição identificada.
    4. Usa o LLM para compor o plano alimentar de 7 dias com:
       - Café da manhã, almoço, lanche e jantar
       - Dados nutricionais reais dos alimentos (via MCP)
       - Citações das fontes RAG
    5. Retorna o plano estruturado em Markdown.

Restrições suportadas (detectadas automaticamente):
    celíaco / sem glúten, diabético / diabetes, hipertenso / hipertensão,
    sem lactose / intolerância à lactose, alergia a amendoim, vegetariano,
    vegano, baixo sódio, baixo açúcar.
"""

import logging
import os
import re
from typing import Any

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from agents.retriever import retrieve
from nutritools.opennutrition_client import search_foods, get_food_by_id, get_nutrition_summary

logger = logging.getLogger(__name__)

# ── Configuração ──────────────────────────────────────────────────────────────

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_llm = None

def get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            num_predict=2048,
        )
    return _llm


# ── Mapeamento de restrições ──────────────────────────────────────────────────

RESTRICTION_PATTERNS = {
    "celíaco":       [r"cel[íi]ac[ao]", r"sem\s+gl[úu]ten", r"gl[úu]ten"],
    "diabético":     [r"diab[eé]tic[ao]", r"diabetes", r"glicemia", r"insulina"],
    "hipertenso":    [r"hipertens[ãa]o", r"hipertenso", r"press[ãa]o\s+alta", r"baixo\s+s[óo]dio"],
    "sem_lactose":   [r"sem\s+lactose", r"intoler[aâ]ncia\s+[aà]\s+lactose", r"lactose"],
    "vegetariano":   [r"vegetarian[ao]"],
    "vegano":        [r"vegan[ao]"],
}

# Alimentos-chave para buscar no MCP por restrição
RESTRICTION_FOODS = {
    "celíaco":     ["arroz integral", "quinoa", "batata doce", "frango", "ovo",
                    "mandioca", "milho", "amendoim", "abacate", "laranja"],
    "diabético":   ["lentilha", "brócolis", "salmão", "tofu", "pepino",
                    "alface", "abobrinha", "atum", "queijo cottage", "iogurte natural"],
    "hipertenso":  ["banana", "espinafre", "batata doce", "feijão", "salmão",
                    "cenoura", "beterraba", "laranja", "frango", "aveia"],
    "sem_lactose": ["tofu", "sardinha", "couve", "amendoim", "quinoa",
                    "frango", "atum", "batata doce", "arroz integral", "manga"],
    "vegetariano": ["tofu", "lentilha", "grão de bico", "quinoa", "ovo",
                    "espinafre", "brócolis", "batata doce", "iogurte natural", "queijo"],
    "vegano":      ["tofu", "lentilha", "grão de bico", "quinoa", "amendoim",
                    "espinafre", "brócolis", "batata doce", "arroz integral", "manga"],
    "geral":       ["arroz integral", "feijão", "frango", "brócolis", "banana",
                    "ovo", "batata doce", "cenoura", "laranja", "atum"],
}

DEFAULT_FOODS = ["arroz integral", "feijão", "frango", "brócolis", "banana",
                 "ovo", "batata doce", "cenoura", "laranja", "atum"]


# ── Extração de restrições ────────────────────────────────────────────────────

def extract_restrictions(message: str) -> list[str]:
    """Identifica restrições alimentares mencionadas na mensagem."""
    found = []
    msg_lower = message.lower()
    for restriction, patterns in RESTRICTION_PATTERNS.items():
        if any(re.search(p, msg_lower) for p in patterns):
            found.append(restriction)
    return found if found else ["geral"]


# ── Busca MCP de alimentos ────────────────────────────────────────────────────

def _ask_llm_for_foods(restrictions: list[str], message: str) -> list[str]:
    """
    Pede ao LLM uma lista de alimentos adequados para as restrições,
    em inglês (para busca no dataset OpenNutrition).
    Retorna lista de termos de busca.
    """
    import os
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate(
        input_variables=["restrictions", "message"],
        template="""You are a nutrition expert. List exactly 15 suitable foods for someone with these dietary restrictions: {restrictions}

Context: {message}

Rules:
- Use simple English food names suitable for database search (e.g. "brown rice", "chicken breast", "lentils")
- Each food on a separate line, no numbering, no explanations
- Only foods SAFE for the restrictions above
- Vary between proteins, carbs, vegetables, fruits

List 15 foods:"""
    )

    try:
        llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.4,
            num_predict=256,
        )
        raw = (prompt | llm).invoke({
            "restrictions": ", ".join(restrictions),
            "message": message[:200],
        })
        foods = [
            line.strip().lower().strip("-•*").strip()
            for line in raw.strip().splitlines()
            if line.strip() and len(line.strip()) > 2
        ]
        # Limpa numeração residual (ex: "1. brown rice" → "brown rice")
        import re
        foods = [re.sub(r"^[0-9]+[.)\s]+", "", f).strip() for f in foods]
        foods = [f for f in foods if f and not f[0].isdigit()][:20]
        logger.info(f"LLM sugeriu {len(foods)} alimentos para busca MCP: {foods}")
        return foods
    except Exception as e:
        logger.warning(f"Falha ao consultar LLM para lista de alimentos: {e}")
        # Fallback para lista hardcoded por restrição
        food_names = set()
        for r in restrictions:
            food_names.update(RESTRICTION_FOODS.get(r, DEFAULT_FOODS))
        return list(food_names)


def _fetch_food_data(food_name_en: str) -> tuple[str, str] | None:
    """
    Busca dados nutricionais de um alimento no MCP.
    Retorna (food_name_en, summary) ou None se não encontrado.
    O nome em inglês é preservado como chave — a tradução acontece em get_nutrition_summary.
    """
    try:
        results = search_foods(food_name_en, limit=1)
        if not results:
            return None

        food = results[0]
        food_id = food.get("id", "")
        # Usa o nome do dataset se disponível, senão o termo de busca
        name_from_dataset = food.get("name") or food_name_en

        # Tenta detalhes completos via get-food-by-id
        if food_id and str(food_id).startswith("fd_"):
            details = get_food_by_id(str(food_id))
            if details:
                # Garante que o name está presente antes de passar para summary
                if not details.get("name"):
                    details["name"] = name_from_dataset
                summary = get_nutrition_summary(details)
                if summary and "não identificado" not in summary:
                    return (food_name_en, summary)

        # Usa nutrition_100g que já veio no resultado do search
        n100 = food.get("nutrition_100g", {})
        if n100:
            # Injeta o nome explicitamente para evitar "Alimento não identificado"
            summary = get_nutrition_summary({**n100, "name": name_from_dataset})
            return (food_name_en, summary)
        else:
            return (food_name_en, f"**{name_from_dataset.title()}** — encontrado no OpenNutrition.")

    except Exception as e:
        logger.warning(f"MCP falhou para '{food_name_en}': {e}")
    return None


def fetch_foods_for_restrictions(restrictions: list[str], message: str = "") -> dict[str, str]:
    """
    Usa o LLM para decidir quais alimentos buscar, depois consulta
    o MCP OpenNutrition para obter os dados nutricionais reais.

    Retorna dict: { nome_en → summary_nutricional }
    """
    # LLM decide os alimentos em inglês
    food_names_en = _ask_llm_for_foods(restrictions, message)

    nutrition_data = {}
    mcp_available = True

    for food_name in food_names_en:
        result = _fetch_food_data(food_name)
        if result:
            key, summary = result
            nutrition_data[key] = summary
        else:
            logger.debug(f"'{food_name}' não encontrado no OpenNutrition")
            mcp_available = False

    logger.info(f"MCP retornou dados para {len(nutrition_data)}/{len(food_names_en)} alimentos")
    if not mcp_available:
        logger.info("Alguns alimentos não encontrados no MCP — plano gerado com dados parciais.")

    return nutrition_data


def format_nutrition_context(nutrition_data: dict) -> str:
    """Formata os dados nutricionais do MCP para o prompt."""
    if not nutrition_data:
        return "Dados nutricionais do MCP não disponíveis para esta sessão."
    lines = ["### Dados nutricionais dos alimentos (fonte: OpenNutrition via MCP)\n"]
    for food_name, summary in nutrition_data.items():
        lines.append(summary)
    return "\n".join(lines)


# ── Prompts do Automation Agent ──────────────────────────────────────────────

# Prompt para gerar UM dia do plano
DAY_PROMPT = PromptTemplate(
    input_variables=["day_num", "day_name", "restrictions", "nutrition_context",
                     "used_meals", "rag_context"],
    template="""Você é um assistente nutricional. Crie as refeições do {day_name} de um plano alimentar.

Restrições: {restrictions}

Alimentos disponíveis com dados nutricionais (OpenNutrition via MCP):
{nutrition_context}

Diretrizes nutricionais:
{rag_context}

Refeições JÁ USADAS nos dias anteriores (NÃO repita):
{used_meals}

Crie 4 refeições DIFERENTES das anteriores para {day_name}:

## {day_name}
| Refeição | Opção sugerida | Observação |
|----------|---------------|------------|
| ☀️ Café da manhã | [alimento diferente dos anteriores] | [dica] |
| 🌞 Almoço | [alimento diferente dos anteriores] | [dica] |
| 🌤️ Lanche | [alimento diferente dos anteriores] | [dica] |
| 🌙 Jantar | [alimento diferente dos anteriores] | [dica] |
"""
)

# Prompt para o cabeçalho e observações finais
HEADER_PROMPT = PromptTemplate(
    input_variables=["restrictions", "message"],
    template="""Você é um assistente nutricional. Escreva uma introdução curta para um plano alimentar.

Restrições: {restrictions}
Solicitação: {message}

Escreva APENAS:
1. Um título: # 🥗 Plano Alimentar Semanal — [restrições]
2. Uma seção "## Como usar este plano" com 2 frases práticas.

Seja breve e direto:"""
)

FOOTER_PROMPT = PromptTemplate(
    input_variables=["restrictions", "rag_context"],
    template="""Com base nas diretrizes abaixo, escreva 3 observações nutricionais importantes para quem tem: {restrictions}

Diretrizes:
{rag_context}

Escreva APENAS a seção:
## Observações nutricionais importantes
[3 bullet points com citações [N] quando aplicável]

## Referências
[lista das fontes]
"""
)


# ── Interface principal ───────────────────────────────────────────────────────

def generate_meal_plan(message: str) -> dict[str, Any]:
    """
    Ponto de entrada do Automation Agent.

    Parâmetros:
        message : mensagem original do usuário (ex: "gera um plano para celíaco")

    Retorna:
    {
        "draft"           : str,   # plano alimentar em Markdown com citações
        "restrictions"    : list,  # restrições identificadas
        "mcp_foods"       : dict,  # dados nutricionais obtidos via MCP
        "chunks"          : list,  # chunks RAG usados
        "status"          : "ok" | "partial" | "error",
        "message"         : str,
    }
    """
    logger.info(f"Automation Agent iniciado. Message: '{message[:80]}'")

    # 1. Extrai restrições
    restrictions = extract_restrictions(message)
    logger.info(f"Restrições detectadas: {restrictions}")

    # 2. Busca diretrizes no RAG
    rag_query = f"plano alimentar restrição {' '.join(restrictions)} alimentos permitidos proibidos"
    rag_result = retrieve(rag_query)
    chunks = rag_result.get("chunks", [])

    rag_context = ""
    if chunks:
        from agents.answerer import format_references
        # Trunca cada chunk a 400 chars para evitar que listas longas dominem o prompt
        lines = []
        for i, chunk in enumerate(chunks, 1):
            fonte = f"{chunk.get('title', 'Documento')} — {chunk.get('publisher', '')}"
            if chunk.get("year"):
                fonte += f", {chunk['year']}"
            trecho = chunk.get("text", "").strip()[:400]
            lines.append(f"[{i}] Fonte: {fonte}, p. {chunk.get('page', '?')}")
            lines.append(f"Trecho: {trecho}")
            lines.append("")
        rag_context = "\n".join(lines) + "\n\n" + format_references(chunks)
    else:
        rag_context = "Nenhuma diretriz específica encontrada no corpus para estas restrições."
        logger.warning("RAG não retornou chunks para o plano alimentar.")

    # 3. Busca dados nutricionais via MCP
    mcp_foods = fetch_foods_for_restrictions(restrictions, message)
    nutrition_context = format_nutrition_context(mcp_foods)

    # 4. Gera o plano dia por dia para evitar repetição
    DAYS = [
        ("Dia 1 — Segunda-feira",  "Segunda-feira"),
        ("Dia 2 — Terça-feira",    "Terça-feira"),
        ("Dia 3 — Quarta-feira",   "Quarta-feira"),
        ("Dia 4 — Quinta-feira",   "Quinta-feira"),
        ("Dia 5 — Sexta-feira",    "Sexta-feira"),
        ("Dia 6 — Sábado",         "Sábado"),
        ("Dia 7 — Domingo",        "Domingo"),
    ]

    restrictions_str = ", ".join(r.replace("_", " ").title() for r in restrictions)

    try:
        llm = get_llm()

        # Cabeçalho
        header = (HEADER_PROMPT | llm).invoke({
            "restrictions": restrictions_str,
            "message": message,
        }).strip()

        # Gera cada dia passando as refeições já usadas
        days_text = []
        used_meals: list[str] = []

        for day_label, day_name in DAYS:
            used_str = "\n".join(used_meals[-12:]) if used_meals else "Nenhuma ainda."
            day_raw = (DAY_PROMPT | llm).invoke({
                "day_num":          day_label,
                "day_name":         day_label,
                "restrictions":     restrictions_str,
                "nutrition_context": nutrition_context[:800],  # trunca para caber no contexto
                "used_meals":       used_str,
                "rag_context":      rag_context[:400],
            }).strip()

            days_text.append(day_raw)

            # Extrai as refeições geradas para passar ao próximo dia
            for line in day_raw.splitlines():
                if "|" in line and any(icon in line for icon in ["☀️", "🌞", "🌤️", "🌙"]):
                    cols = [c.strip() for c in line.split("|") if c.strip()]
                    if len(cols) >= 2:
                        used_meals.append(cols[1])  # coluna "Opção sugerida"

            logger.info(f"Gerado: {day_label}")

        # Rodapé com observações
        footer = (FOOTER_PROMPT | llm).invoke({
            "restrictions": restrictions_str,
            "rag_context":  rag_context[:600],
        }).strip()

        draft = header + "\n\n---\n\n" + "\n\n".join(days_text) + "\n\n---\n\n" + footer

        status = "ok" if chunks and mcp_foods else "partial"
        logger.info(f"Automation Agent concluído. Status: {status}")

        return {
            "draft": draft,
            "restrictions": restrictions,
            "mcp_foods": mcp_foods,
            "chunks": chunks,
            "status": status,
            "message": f"Plano gerado com {len(chunks)} fonte(s) RAG e {len(mcp_foods)} alimento(s) via MCP.",
        }

    except Exception as e:
        logger.error(f"Erro no Automation Agent: {e}")
        return {
            "draft": "Ocorreu um erro ao gerar o plano alimentar. Tente novamente.",
            "restrictions": restrictions,
            "mcp_foods": mcp_foods,
            "chunks": chunks,
            "status": "error",
            "message": f"Erro interno: {e}",
        }


# ── Teste direto ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    test_cases = [
        "Gera um plano alimentar semanal para celíaco com hipertensão",
        "Quero um cardápio de 7 dias para diabético tipo 2",
        "Cria uma dieta semanal sem lactose e vegetariana",
    ]

    for msg in test_cases:
        print(f"\n{'='*60}")
        print(f"Input: {msg}")
        print("=" * 60)
        result = generate_meal_plan(msg)
        print(f"Status: {result['status']}")
        print(f"Restrições: {result['restrictions']}")
        print(f"Alimentos MCP: {list(result['mcp_foods'].keys())}")
        print(f"\nPlano (primeiros 500 chars):\n{result['draft'][:500]}...")