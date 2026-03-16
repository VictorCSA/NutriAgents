"""
src/mcp/opennutrition_client.py
--------------------------------
Cliente MCP para o servidor mcp-opennutrition (Node.js local).

Expõe as 4 tools do servidor como funções Python simples,
com logging de todas as chamadas (auditoria) e allowlist de operações.

Tools disponíveis (allowlist completa):
    - search_foods      : busca por nome/marca (→ search_by_name do MCP)
    - get_food_by_id    : detalhes nutricionais por ID (→ get_by_id do MCP)
    - browse_foods      : listagem paginada (→ browse_foods do MCP)
    - lookup_barcode    : busca por código de barras EAN-13 (→ barcode_lookup do MCP)

Segurança:
    - Allowlist explícita: apenas as 4 tools acima são permitidas.
    - Todas as chamadas são registradas em logs/mcp_calls.jsonl.
    - Sem acesso a sistema de arquivos, rede ou outros recursos.
    - Timeout de 15s por chamada (evita travamento do pipeline).
    - Parâmetros são validados antes de passar ao servidor MCP.

Configuração:
    MCP_NODE_BIN    : caminho do binário node (default: "node")
    MCP_SERVER_PATH : caminho absoluto para mcp-opennutrition/build/index.js

Uso:
    from src.mcp.opennutrition_client import search_foods, get_food_by_id
    results = search_foods("arroz integral", limit=5)
"""

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuração ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MCP_CALL_LOG = LOGS_DIR / "mcp_calls.jsonl"

MCP_NODE_BIN = os.getenv("MCP_NODE_BIN", "node")
MCP_SERVER_PATH = os.getenv(
    "MCP_SERVER_PATH",
    str(PROJECT_ROOT / "mcp-opennutrition" / "build" / "index.js"),
)

# Allowlist explícita de tools permitidas
ALLOWED_TOOLS = {"search-food-by-name", "get-food-by-id", "browse-foods", "barcode-lookup"}

# ── Logging de auditoria ──────────────────────────────────────────────────────

def _log_call(tool: str, params: dict, result: Any, error: str | None = None) -> None:
    """Registra cada chamada MCP em logs/mcp_calls.jsonl (auditoria)."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "params": params,
        "success": error is None,
        "error": error,
        "result_preview": str(result)[:200] if result else None,
    }
    try:
        with open(MCP_CALL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Falha ao registrar chamada MCP no log: {e}")


# ── Cliente MCP via langchain-mcp-adapters ────────────────────────────────────

def _get_mcp_tools() -> dict:
    """
    Inicializa a conexão com o servidor MCP e retorna as tools disponíveis.
    Usa MultiServerMCPClient do langchain-mcp-adapters (stdio transport).
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        raise ImportError(
            "langchain-mcp-adapters não instalado. "
            "Execute: pip install langchain-mcp-adapters"
        )

    server_path = Path(MCP_SERVER_PATH)
    if not server_path.exists():
        raise FileNotFoundError(
            f"Servidor MCP não encontrado em: {MCP_SERVER_PATH}\n"
            "Clone e compile o mcp-opennutrition:\n"
            "  git clone https://github.com/deadletterq/mcp-opennutrition\n"
            "  cd mcp-opennutrition && npm install && npm run build"
        )

    client = MultiServerMCPClient(
        {
            "opennutrition": {
                "command": MCP_NODE_BIN,
                "args": [str(server_path)],
                "transport": "stdio",
            }
        }
    )
    return client


def _call_tool(tool_name: str, params: dict, timeout: int = 15) -> Any:
    """
    Chama uma tool do servidor MCP com validação de allowlist e logging.

    Raises:
        ValueError  : se tool_name não estiver na allowlist.
        RuntimeError: se o servidor MCP não estiver disponível.
    """
    # Verificação de allowlist
    if tool_name not in ALLOWED_TOOLS:
        msg = (
            f"Tool '{tool_name}' não está na allowlist. "
            f"Permitidas: {sorted(ALLOWED_TOOLS)}"
        )
        logger.error(msg)
        _log_call(tool_name, params, None, error=msg)
        raise ValueError(msg)

    logger.info(f"MCP call → tool='{tool_name}' params={params}")

    try:
        import asyncio
        import subprocess, json as _json, sys as _sys

        async def _invoke():
            from langchain_mcp_adapters.client import MultiServerMCPClient

            server_path = Path(MCP_SERVER_PATH)
            if not server_path.exists():
                raise FileNotFoundError(
                    f"Servidor MCP não encontrado em: {MCP_SERVER_PATH}"
                )

            # API 0.1.0: sem context manager
            client = MultiServerMCPClient(
                {
                    "opennutrition": {
                        "command": MCP_NODE_BIN,
                        "args": [str(server_path)],
                        "transport": "stdio",
                    }
                }
            )
            tools = await client.get_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool is None:
                raise RuntimeError(
                    f"Tool '{tool_name}' não encontrada. "
                    f"Disponíveis: {[t.name for t in tools]}"
                )
            return await tool.ainvoke(params)

        # Streamlit já roda um event loop — usa nest_asyncio para permitir
        # asyncio.run() aninhado, ou cria um loop novo em thread separada
        try:
            import nest_asyncio
            nest_asyncio.apply()
            result = asyncio.run(_invoke())
        except ImportError:
            # nest_asyncio não instalado — roda em thread com loop próprio
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _invoke())
                result = future.result(timeout=30)

        _log_call(tool_name, params, result)
        return result

    except Exception as e:
        # Desempacota ExceptionGroup (Python 3.11) ou BaseExceptionGroup (3.10)
        inner = e
        if hasattr(e, 'exceptions') and e.exceptions:
            inner = e.exceptions[0]
            # Log de todas as sub-exceções para diagnóstico
            for i, sub in enumerate(e.exceptions):
                logger.error(f"  Sub-exceção [{i}]: {type(sub).__name__}: {sub}")
        error_msg = f"{type(inner).__name__}: {inner}"
        logger.error(f"Erro na chamada MCP '{tool_name}': {error_msg}", exc_info=True)
        _log_call(tool_name, params, None, error=error_msg)
        raise RuntimeError(f"Falha na tool MCP '{tool_name}': {error_msg}") from inner


# ── API pública ───────────────────────────────────────────────────────────────

# Traduções PT→EN para busca no dataset (em inglês)
_PT_TO_EN = {
    # Proteínas
    "salmão": "salmon fillet", "frango": "chicken breast", "atum": "tuna",
    "sardinha": "sardine", "ovo": "whole egg", "tofu": "tofu",
    "queijo cottage": "cottage cheese", "iogurte natural": "plain yogurt",
    "queijo": "cheese",
    # Grãos e tubérculos
    "arroz integral": "brown rice", "quinoa": "quinoa grain",
    "batata doce": "sweet potato", "mandioca": "cassava",
    "milho": "corn", "feijão": "black beans", "lentilha": "lentils",
    "grão de bico": "chickpeas", "aveia": "rolled oats",
    # Vegetais
    "brócolis": "broccoli", "espinafre": "spinach", "couve": "kale",
    "cenoura": "carrot", "beterraba": "beetroot", "alface": "lettuce",
    "abobrinha": "zucchini", "pepino": "cucumber",
    # Frutas
    "banana": "banana", "laranja": "orange", "manga": "mango",
    "abacate": "avocado",
    # Oleaginosas
    "amendoim": "peanut", "amêndoas": "almond",
    # Outros
    "gergelim": "sesame", "leite de aveia": "oat milk",
    "chia": "chia seeds",
}

# Tradução EN→PT para exibir nome amigável ao usuário
_EN_TO_PT = {v: k for k, v in _PT_TO_EN.items()}
# Mapeamentos adicionais para nomes parciais do dataset
_EN_TO_PT.update({
    "salmon": "Salmão", "chicken": "Frango", "tuna": "Atum",
    "egg": "Ovo", "rice": "Arroz", "potato": "Batata",
    "bean": "Feijão", "oat": "Aveia", "broccoli": "Brócolis",
    "spinach": "Espinafre", "carrot": "Cenoura", "banana": "Banana",
    "orange": "Laranja", "mango": "Manga", "avocado": "Abacate",
    "lentil": "Lentilha", "chickpea": "Grão-de-bico", "quinoa": "Quinoa",
    "kale": "Couve", "sardine": "Sardinha", "cheese": "Queijo",
    "yogurt": "Iogurte", "tofu": "Tofu", "almond": "Amêndoa",
    "peanut": "Amendoim", "corn": "Milho", "cassava": "Mandioca",
    "zucchini": "Abobrinha", "cucumber": "Pepino", "lettuce": "Alface",
    "beetroot": "Beterraba",
})


def _pt_name(english_name: str, fallback: str) -> str:
    """Retorna nome em português para um alimento buscado em inglês."""
    if not english_name or english_name.strip() in ("", "Alimento desconhecido", "Alimento"):
        return fallback if fallback and fallback not in ("Alimento", "") else english_name
    # Tenta match exato
    if english_name.lower() in _EN_TO_PT:
        return _EN_TO_PT[english_name.lower()].title()
    # Tenta match parcial (ex: "Brown Rice, cooked" → "Arroz integral")
    name_lower = english_name.lower()
    for en_key, pt_val in _EN_TO_PT.items():
        if en_key in name_lower:
            return pt_val.title()
    # Sem tradução — retorna o nome original do dataset (nunca "Alimento")
    return english_name.title()


def _parse_mcp_result(raw) -> list[dict]:
    """Extrai lista de alimentos do resultado bruto do MCP."""
    # O MCP retorna lista de content blocks: [{'type':'text','text':'[...]','id':'lc_...'}]
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("type") == "text":
                try:
                    parsed = json.loads(item["text"])
                    if isinstance(parsed, list):
                        return parsed
                    if isinstance(parsed, dict) and "foods" in parsed:
                        return parsed["foods"]
                except (json.JSONDecodeError, KeyError):
                    pass
        # Pode ser lista direta de alimentos
        if raw and isinstance(raw[0], dict) and "id" in raw[0]:
            return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def search_foods(query: str, limit: int = 5) -> list[dict]:
    """
    Busca alimentos por nome, traduzindo PT→EN automaticamente.

    Retorna lista de dicts com id (fd_...), name e informações básicas.
    """
    if not query or not query.strip():
        return []

    limit = max(1, min(20, limit))

    # Traduz para inglês se houver mapeamento
    query_en = _PT_TO_EN.get(query.strip().lower(), query.strip())

    try:
        raw = _call_tool("search-food-by-name", {"query": query_en, "limit": limit})
        return _parse_mcp_result(raw)
    except RuntimeError as e:
        logger.warning(f"search_foods falhou: {e}")
        return []


def get_food_by_id(food_id: str) -> dict | None:
    """
    Retorna o perfil nutricional completo de um alimento por ID (fd_...).

    Retorna dict com macronutrientes, vitaminas e minerais,
    ou None se não encontrado ou id inválido.
    """
    if not food_id:
        return None

    # O servidor exige id começando com "fd_" — rejeita ids do LangChain (lc_...)
    food_id = str(food_id)
    if not food_id.startswith("fd_"):
        logger.warning(f"get_food_by_id ignorado: id inválido '{food_id}' (deve começar com fd_)")
        return None

    try:
        raw = _call_tool("get-food-by-id", {"id": food_id})
        items = _parse_mcp_result(raw)
        if items:
            return items[0] if isinstance(items, list) else items
        # Tenta parsear direto
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return raw if isinstance(raw, dict) else None
    except RuntimeError as e:
        logger.warning(f"get_food_by_id falhou para id='{food_id}': {e}")
        return None


def browse_foods(page: int = 1, page_size: int = 10) -> list[dict]:
    """
    Lista alimentos de forma paginada.

    Parâmetros:
        page      : número da página (começa em 1)
        page_size : itens por página (1–50)
    """
    page = max(1, page)
    page_size = max(1, min(50, page_size))

    try:
        raw = _call_tool("browse_foods", {"page": page, "pageSize": page_size})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and "foods" in raw:
            return raw["foods"]
        return []
    except RuntimeError as e:
        logger.warning(f"browse_foods falhou: {e}")
        return []


def lookup_barcode(barcode: str) -> dict | None:
    """
    Busca um alimento pelo código de barras EAN-13.

    Retorna dict com informações do produto ou None se não encontrado.
    """
    if not barcode or not barcode.strip():
        return None

    try:
        raw = _call_tool("barcode_lookup", {"barcode": barcode.strip()})
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"raw": raw}
        return raw if isinstance(raw, dict) else None
    except RuntimeError as e:
        logger.warning(f"lookup_barcode falhou para '{barcode}': {e}")
        return None


def get_nutrition_summary(food_data: dict) -> str:
    """
    Formata os dados nutricionais de um alimento em texto legível.
    Suporta o formato do OpenNutrition: { name, nutrition_100g: { energy_kcal, proteins, ... } }
    """
    if not food_data:
        return "Dados nutricionais não disponíveis."

    raw_name = food_data.get("name", "") or ""
    # Exibe nome em português quando possível; nunca usa "Alimento" como fallback
    name = _pt_name(raw_name, raw_name) if raw_name else "Alimento não identificado"

    # OpenNutrition aninha os dados em nutrition_100g
    n = food_data.get("nutrition_100g") or food_data

    # Mapeamento abrangente — cobre variações de nomes do dataset
    nutrient_map = {
        # Energia
        "energy_kcal":    ("Calorias", "kcal"),
        "energy":         ("Calorias", "kcal"),
        "calories":       ("Calorias", "kcal"),
        # Macros
        "proteins":       ("Proteína", "g"),
        "protein":        ("Proteína", "g"),
        "carbohydrates":  ("Carboidratos", "g"),
        "carbs":          ("Carboidratos", "g"),
        "fat":            ("Gordura total", "g"),
        "total_fat":      ("Gordura total", "g"),
        "fiber":          ("Fibra", "g"),
        "dietary_fiber":  ("Fibra", "g"),
        # Micro
        "sodium":         ("Sódio", "mg"),
        "sugars":         ("Açúcar", "g"),
        "sugar":          ("Açúcar", "g"),
        "saturated_fat":  ("Gordura saturada", "g"),
        "saturatedFat":   ("Gordura saturada", "g"),
        "iron":           ("Ferro", "mg"),
        "calcium":        ("Cálcio", "mg"),
    }

    seen_labels = set()
    lines = [f"**{name}** (por 100g):"]

    for key, (label, unit) in nutrient_map.items():
        if label in seen_labels:
            continue
        val = n.get(key)
        if val is None:
            val = food_data.get(key)
        if val is not None and val != 0:
            lines.append(f"  - {label}: {val} {unit}")
            seen_labels.add(label)

    if len(lines) == 1:
        available = list((n or food_data).keys())[:8]
        logger.debug(f"Nutrientes não mapeados para '{name}': {available}")
        return f"**{name}** — encontrado no OpenNutrition."

    return "\n".join(lines)