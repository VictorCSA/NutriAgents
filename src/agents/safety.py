import logging
import re
from typing import Any

# Logging
logger = logging.getLogger(__name__)

# Disclaimer padrão — obrigatório em toda resposta nutricional
DISCLAIMER_PADRAO = (
    "\n\n---\n"
    "**Aviso importante:** As informações fornecidas têm caráter educativo e "
    "informativo, baseadas em documentos públicos de saúde. "
    "**Não substituem consulta com nutricionista, médico ou outro profissional de saúde.** "
    "Sempre busque orientação profissional antes de fazer mudanças na sua alimentação, "
    "especialmente se você possui condições de saúde diagnosticadas."
)

DISCLAIMER_SUPLEMENTO = (
    "\n\n**Atenção — suplementação:** Doses, tipos e indicações de suplementos "
    "variam conforme condição clínica individual. Consulte um nutricionista ou médico "
    "antes de iniciar qualquer suplementação."
)

DISCLAIMER_PRESCRICAO = (
    "\n\n**Atenção — orientação individualizada:** A resposta acima contém "
    "recomendações de caráter geral. Um plano alimentar personalizado deve ser "
    "elaborado por nutricionista habilitado, considerando seu histórico clínico completo."
)

# Padrões de detecção por categoria

# BLOQUEIO: substituição de medicação por dieta
PADROES_SUBSTITUIR_MEDICACAO = [
    re.compile(p, re.IGNORECASE) for p in [
        r"substitu[íi]r?\s+.{0,30}(medica[çc][ãa]o|rem[ée]dio|insulina|metformina|"
        r"losartana|enalapril|estatina|antidiab[ée]tico)",
        r"(parar|suspender|deixar\s+de\s+tomar)\s+.{0,20}(medica[çc][ãa]o|rem[ée]dio|"
        r"insulina|comprimido)",
        r"dieta\s+(pode|consegue|[ée]\s+capaz\s+de)\s+(substituir|repor|dispensar)",
        # "não precisará/precisa/precisar mais (de) [medicamento ou tomar]"
        r"n[ãa]o\s+precisar[áa]?\s+mais\s+(de\s+)?(medica[çc][ãa]o|rem[ée]dio|insulina|tomar)",
        r"n[ãa]o\s+precisa\s+mais\s+(de\s+)?(medica[çc][ãa]o|rem[ée]dio|insulina|tomar)",
        # "não precisará mais tomar [qualquer coisa]" — cobre nomes de medicamentos direto
        r"n[ãa]o\s+precisar[áa]?\s+mais\s+tomar\b",
        r"n[ãa]o\s+precisa\s+mais\s+tomar\b",
        r"cura[r]?\s+(diabetes|hipertens[ãa]o|doen[çc]a)\s+(com|atrav[ée]s|pela?)\s+"
        r"(dieta|alimenta[çc][ãa]o)",
    ]
]


# BLOQUEIO: perda de peso extrema
PADROES_PERDA_PESO_EXTREMA = [
    re.compile(p, re.IGNORECASE) for p in [
        r"perder?\s+\d{2,}\s*kg\s+(em|por)\s+\d+\s*(dias?|semanas?|m[eê]s)",
        r"jejum\s+(prolongado|extendido|de\s+\d+\s*(dias?|horas?))",
        r"menos\s+de\s+[45]\d{2}\s*kcal",
        r"dieta\s+(muito\s+)?(extrema|radical|drastica|severa|agressiva)",
        r"(eliminar|cortar|zerar)\s+(completamente|totalmente)\s+(carboidratos|"
        r"gorduras|calorias)",
        r"\b([3-9]\d{2}|[1-9]\d{3})\s*calorias?\s+por\s+dia\b",  # < 1000 kcal/dia
    ]
]

# DISCLAIMER EXTRA: prescrição dietética individualizada
PADROES_PRESCRICAO = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(voc[eê]|o\s+paciente)\s+deve\s+(comer|consumir|ingerir|evitar)\s+exatamente",
        r"sua\s+(dieta|alimenta[çc][ãa]o)\s+deve\s+ser",
        r"recomendo\s+(especificamente|para\s+voc[eê])",
        r"no\s+seu\s+caso\s+(espec[íi]fico)?,?\s+(voc[eê]\s+deve|recomendo|indico)",
        r"(prescrevo|prescri[çc][ãa]o)\s+(diet[ée]tica|alimentar)",
    ]
]

# DISCLAIMER EXTRA: suplementos
PADROES_SUPLEMENTO = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(suplemento|suplementa[çc][ãa]o|suplemente)\b",
        r"\b(vitamina|mineral|prote[íi]na\s+em\s+p[óo]|whey|creatina|"
        r"[ôo]mega[-\s]?3|ferro|c[áa]lcio|magn[ée]sio)\s+(em\s+)?(c[áa]psulas?|"
        r"comprimidos?|doses?|mg|mcg)\b",
        r"\d+\s*(mg|mcg|g|ui|iu)\s+(de|do|da)\s+\w+",
    ]
]

# Funções de detecção
def detectar_substituicao_medicacao(texto: str) -> bool:
    return any(p.search(texto) for p in PADROES_SUBSTITUIR_MEDICACAO)

def detectar_perda_peso_extrema(texto: str) -> bool:
    return any(p.search(texto) for p in PADROES_PERDA_PESO_EXTREMA)


def detectar_prescricao(texto: str) -> bool:
    return any(p.search(texto) for p in PADROES_PRESCRICAO)


def detectar_suplemento(texto: str) -> bool:
    return any(p.search(texto) for p in PADROES_SUPLEMENTO)

# Interface principal do agente
def check(draft: str) -> dict[str, Any]:
    """
    Ponto de entrada do Safety/Policy Agent.

    Recebe o rascunho de resposta do Answerer e aplica a política de segurança.

    Retorna:
    {
        "status":   "approved" | "approved_with_disclaimer" | "blocked",
        "response": str,   # resposta final (com disclaimers) ou mensagem de recusa
        "reasons":  list[str],   # lista de gatilhos detectados (para log/auditoria)
        "blocked":  bool,
    }
    """
    if not draft or not draft.strip():
        return {
            "status": "blocked",
            "response": "Não foi possível gerar uma resposta para esta pergunta.",
            "reasons": ["rascunho vazio"],
            "blocked": True,
        }

    reasons = []
    extra_disclaimers = []

    # --- Verificações de BLOQUEIO ---

    if detectar_substituicao_medicacao(draft):
        reasons.append("substituição de medicação por dieta")
        logger.warning(f"Safety BLOQUEOU: {reasons[-1]}")
        return {
            "status": "blocked",
            "response": (
                "Não posso fornecer orientações sobre substituição de medicamentos "
                "por dieta. A interrupção ou alteração de medicamentos deve ser "
                "avaliada exclusivamente pelo médico responsável pelo seu tratamento.\n\n"
                "Posso ajudar com informações gerais sobre alimentação saudável "
                "dentro do contexto de condições de saúde específicas."
            ),
            "reasons": reasons,
            "blocked": True,
        }

    if detectar_perda_peso_extrema(draft):
        reasons.append("perda de peso extrema / dieta muito restritiva")
        logger.warning(f"Safety BLOQUEOU: {reasons[-1]}")
        return {
            "status": "blocked",
            "response": (
                "Não posso recomendar dietas extremamente restritivas ou estratégias "
                "de perda de peso acelerada, pois representam risco à saúde.\n\n"
                "Recomendo consultar um nutricionista para um plano alimentar seguro, "
                "personalizado e baseado em evidências científicas."
            ),
            "reasons": reasons,
            "blocked": True,
        }

    # --- Verificações de DISCLAIMER EXTRA ---

    if detectar_prescricao(draft):
        reasons.append("prescrição dietética individualizada detectada")
        extra_disclaimers.append(DISCLAIMER_PRESCRICAO)
        logger.info(f"Safety adicionou disclaimer: {reasons[-1]}")

    if detectar_suplemento(draft):
        reasons.append("menção a suplementação")
        extra_disclaimers.append(DISCLAIMER_SUPLEMENTO)
        logger.info(f"Safety adicionou disclaimer: {reasons[-1]}")

    # --- Monta resposta final com disclaimers ---

    response = draft
    for disclaimer in extra_disclaimers:
        response += disclaimer
    response += DISCLAIMER_PADRAO  # sempre ao final

    status = "approved_with_disclaimer" if reasons else "approved"
    logger.info(f"Safety: {status} | gatilhos: {reasons if reasons else 'nenhum'}")

    return {
        "status": status,
        "response": response,
        "reasons": reasons,
        "blocked": False,
    }


# Utilitário — inspeciona quais gatilhos um texto ativa (útil para testes)
def inspect(texto: str) -> dict[str, bool]:
    """
    Retorna quais categorias de risco foram detectadas no texto.
    Útil para debug e para os testes de avaliação.
    """
    return {
        "substituicao_medicacao": detectar_substituicao_medicacao(texto),
        "perda_peso_extrema": detectar_perda_peso_extrema(texto),
        "prescricao_individualizada": detectar_prescricao(texto),
        "suplemento": detectar_suplemento(texto),
    }

# Execução direta — testes rápidos
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    casos = [
        (
            "DEVE BLOQUEAR — medicação",
            "Com essa dieta você não precisará mais tomar metformina para controlar o diabetes.",
        ),
        (
            "DEVE BLOQUEAR — peso extremo",
            "Para perder 20kg em 30 dias, reduza para menos de 500 calorias por dia.",
        ),
        (
            "DEVE DISCLAIMER — suplemento",
            "O ômega-3 em cápsulas de 1000mg pode auxiliar no controle do colesterol.",
        ),
        (
            "DEVE DISCLAIMER — prescrição",
            "No seu caso específico, você deve consumir exatamente 1800 kcal por dia.",
        ),
        (
            "DEVE APROVAR — informativo geral",
            "Alimentos integrais são ricos em fibras e contribuem para o controle glicêmico.",
        ),
    ]

    for descricao, draft in casos:
        print(f"\n{'=' * 60}")
        print(f"Caso: {descricao}")
        print(f"Draft: {draft[:80]}...")
        print("=" * 60)
        resultado = check(draft)
        print(f"Status:  {resultado['status']}")
        print(f"Blocked: {resultado['blocked']}")
        print(f"Reasons: {resultado['reasons']}")
        print(f"Response (primeiros 200 chars):\n{resultado['response'][:200]}")