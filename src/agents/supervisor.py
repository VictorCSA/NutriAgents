import logging
import os
import re
from typing import Any

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Logging
logger = logging.getLogger(__name__)

# Configuração
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Intenções válidas
VALID_INTENTS = {"qa", "automation", "refuse"}

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
            temperature=0.0,   # classificação deve ser determinística
            num_predict=64,    # resposta curtíssima — só intent + motivo
        )
    return _llm


# Prompt de classificação
SUPERVISOR_PROMPT = PromptTemplate(
    input_variables=["message"],
    template="""Você é o roteador de um assistente nutricional brasileiro.

Classifique a mensagem do usuário em exatamente uma das três intenções:

    qa         → PERGUNTA sobre nutrição, alimentos, restrições alimentares,
                 o que comer ou evitar, composição nutricional, doenças e dieta.
                 A mensagem busca uma INFORMAÇÃO ou EXPLICAÇÃO.
                 Exemplos:
                   - "Quais alimentos um diabético deve evitar?"
                   - "O que celíacos não podem comer?"
                   - "Posso comer arroz com hipertensão?"
                   - "Quais alimentos têm muito sódio?"

    automation → PEDIDO para CRIAR, GERAR, MONTAR, ELABORAR ou FAZER
                 um plano alimentar, cardápio, dieta ou lista de refeições.
                 A mensagem pede um PRODUTO gerado, não uma explicação.
                 Palavras-chave: gera, cria, monta, faz, elabora, quero um plano,
                 me dá um cardápio, preciso de uma dieta.
                 Exemplos:
                   - "Me gera um plano alimentar semanal para celíaco"
                   - "Cria um cardápio de 7 dias sem glúten"
                   - "Quero uma dieta semanal para diabético tipo 2"
                   - "Monta um plano alimentar com restrição de sódio"

    refuse     → qualquer outro assunto não relacionado a nutrição ou alimentação.
                 Exemplos:
                   - "Qual é a capital da França?"
                   - "Me ajuda a programar em Python?"

ATENÇÃO: Se a mensagem contém verbos de ação como GERAR, CRIAR, MONTAR, ELABORAR,
FAZER, QUERO UM PLANO ou PRECISO DE UM CARDÁPIO → classifique como automation.

Responda APENAS neste formato exato, sem mais nada:
INTENT: [qa|automation|refuse]
MOTIVO: [uma frase curta explicando a classificação]

Mensagem do usuário: {message}

INTENT:"""

)

# Parse da resposta do LLM
def parse_intent(raw: str) -> tuple[str, str]:
    """
    Extrai INTENT e MOTIVO da resposta do LLM.

    Retorna (intent, motivo).
    Em caso de falha de parsing, retorna ("qa", motivo_erro) como fallback
    conservador — melhor tentar responder do que recusar por erro técnico.
    """
    intent = None
    motivo = "Classificação automática."

    for line in raw.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("INTENT:"):
            candidate = line.split(":", 1)[1].strip().lower()
            # Remove pontuação residual
            candidate = re.sub(r"[^\w]", "", candidate)
            if candidate in VALID_INTENTS:
                intent = candidate
        elif line.upper().startswith("MOTIVO:"):
            motivo = line.split(":", 1)[1].strip()

    if intent is None:
        logger.warning(
            f"Parse do Supervisor falhou (raw='{raw[:100]}'). "
            "Usando fallback 'qa'."
        )
        intent = "qa"

    return intent, motivo


# Mensagem de recusa padronizada
MENSAGEM_RECUSA = (
    "Desculpe, só consigo ajudar com temas relacionados a **nutrição e alimentação** "
    "— como restrições alimentares, alimentos recomendados para condições de saúde, "
    "composição nutricional e planos alimentares.\n\n"
    "Sua pergunta parece estar fora desse escopo. Posso ajudar com alguma dúvida "
    "sobre alimentação saudável?"
)

# Interface principal do agente
def classify(message: str) -> dict[str, Any]:
    """
    Ponto de entrada do Supervisor Agent.

    Recebe a última mensagem do usuário e retorna a intenção classificada.

    Retorna:
    {
        "intent":   "qa" | "automation" | "refuse",
        "motivo":   str,     # justificativa da classificação
        "message":  str,     # mensagem original (repassada ao próximo agente)
        "response": str,     # preenchido só se intent == "refuse"
    }
    """
    if not message or not message.strip():
        logger.warning("Supervisor recebeu mensagem vazia.")
        return {
            "intent": "refuse",
            "motivo": "Mensagem vazia.",
            "message": message,
            "response": "Por favor, envie uma mensagem para que eu possa ajudar.",
        }

    try:
        llm = get_llm()
        chain = SUPERVISOR_PROMPT | llm
        raw = chain.invoke({"message": message})
        intent, motivo = parse_intent(raw)

        logger.info(
            f"Supervisor: intent='{intent}' | motivo='{motivo}' | "
            f"message='{message[:60]}...'"
        )

        response = MENSAGEM_RECUSA if intent == "refuse" else ""

        return {
            "intent": intent,
            "motivo": motivo,
            "message": message,
            "response": response,
        }

    except Exception as e:
        logger.error(f"Erro inesperado no Supervisor: {e}")
        # Fallback conservador — tenta responder em vez de recusar
        return {
            "intent": "qa",
            "motivo": f"Erro no Supervisor ({e}). Fallback para qa.",
            "message": message,
            "response": "",
        }


# Execução direta — testes rápidos
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    casos = [
        # Q&A esperados
        ("QA", "Quais alimentos um diabético deve evitar?"),
        ("QA", "O que celíacos não podem comer?"),
        ("QA", "Quais são os alimentos com mais sódio?"),
        ("QA", "Posso comer arroz se tenho hipertensão?"),
        # Automação esperada
        ("AUTOMATION", "Me gera um plano alimentar semanal para celíaco com hipertensão"),
        ("AUTOMATION", "Quero um cardápio de 7 dias sem glúten e sem lactose"),
        ("AUTOMATION", "Cria uma dieta semanal para diabético tipo 2"),
        # Recusa esperada
        ("REFUSE", "Qual é a capital da França?"),
        ("REFUSE", "Me ajuda a programar em Python"),
        ("REFUSE", "Quem ganhou a Copa do Mundo de 2022?"),
    ]

    acertos = 0
    for esperado, mensagem in casos:
        resultado = classify(mensagem)
        intent_obtido = resultado["intent"].upper()
        ok = "✓" if intent_obtido == esperado else "✗"
        if intent_obtido == esperado:
            acertos += 1
        print(
            f"{ok} [{esperado:10s} → {intent_obtido:10s}] {mensagem[:55]}"
        )

    print(f"\nAcurácia: {acertos}/{len(casos)} ({acertos/len(casos):.0%})")