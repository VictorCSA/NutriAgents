import sys
import os
import time
import warnings
from pathlib import Path

# Silencia logs verbosos de bibliotecas externas
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import logging
for _lib in ("sentence_transformers", "transformers", "huggingface_hub",
             "filelock", "torch", "tensorflow", "jax", "torchao"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

# Patch para bug do torch.classes no Windows com Streamlit
# https://github.com/pytorch/pytorch/issues/127462
try:
    import torch
    torch.classes.__path__ = []
except Exception:
    pass

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))


from graph.graph import run

# Configuração da página
st.set_page_config(
    page_title="NutriAgents",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS mínimo — só layout e sidebar escura, NUNCA cor de texto
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif !important; }

/* Sidebar escura — único lugar com tema invertido */
[data-testid="stSidebar"] { background-color: #1C1C1C !important; }
[data-testid="stSidebar"] * { color: #CCCCCC !important; background-color: transparent !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #FFFFFF !important; }
[data-testid="stSidebar"] hr { border-color: #444 !important; }
[data-testid="stSidebar"] .stSpinner p { color: #CCCCCC !important; }

/* Pipeline nodes */
.pipeline-container { display:flex; flex-direction:column; gap:4px; padding:0.3rem 0; }
.pipeline-node { display:flex; align-items:center; gap:8px; padding:5px 8px; border-radius:5px; font-size:0.75rem; font-family:'IBM Plex Mono',monospace; color:#666 !important; }
.pipeline-node.active { background:rgba(255,255,255,0.08) !important; color:#7EE8A2 !important; font-weight:700; }
.pipeline-node.done   { color:#5CB85C !important; }
.pipeline-node.error  { color:#E05555 !important; }
.pipeline-node .dot   { width:7px; height:7px; border-radius:50%; background:currentColor; flex-shrink:0; }
.pipeline-node.active .dot { animation:blink 0.9s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* Badges — fundo escuro para contrastar na sidebar */
.badge { display:inline-block; padding:2px 7px; border-radius:4px; font-size:0.69rem; font-weight:700; font-family:'IBM Plex Mono',monospace; }
.badge-green  { background:#1A3A1A !important; color:#7EE8A2 !important; }
.badge-orange { background:#3A2A08 !important; color:#FFC46B !important; }
.badge-red    { background:#3A0808 !important; color:#FF8080 !important; }
.badge-gray   { background:#2A2A2A !important; color:#AAAAAA !important; }

/* Sidebar titles */
.sidebar-title { font-size:0.64rem !important; font-weight:700 !important; letter-spacing:0.12em !important; text-transform:uppercase !important; color:#777 !important; margin:0.9rem 0 0.35rem 0 !important; }

/* Mensagem do usuário — balão escuro */
.msg-user {
    background: #1A1A1A !important;
    color: #F5F5F5 !important;
    padding: 0.75rem 1rem;
    border-radius: 14px 14px 3px 14px;
    margin: 0.3rem 0 0.5rem 4rem;
    font-size: 0.9rem;
    line-height: 1.6;
}
.msg-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin: 0.6rem 0 0.15rem 0;
}
</style>
""", unsafe_allow_html=True)

# Estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = {}
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

PIPELINE_NODES_QA = [
    ("supervisor", "Supervisor"),
    ("retriever",  "Retriever"),
    ("answerer",   "Answerer"),
    ("self_check", "Self-Check"),
    ("safety",     "Safety"),
]

PIPELINE_NODES_AUTO = [
    ("supervisor",  "Supervisor"),
    ("automation",  "Automation"),
    ("mcp",         "MCP OpenNutrition"),
    ("safety",      "Safety"),
]

PIPELINE_NODES = PIPELINE_NODES_QA  # default

# Sidebar
with st.sidebar:
    st.markdown("## 🥗 NutriAgents")
    st.caption("Assistente nutricional com documentos públicos brasileiros.")
    st.divider()

    st.markdown('<p class="sidebar-title">Pipeline</p>', unsafe_allow_html=True)
    pipeline_placeholder = st.empty()

    def render_pipeline(active_node="", done_nodes=[], error_node="", nodes=None):
        if nodes is None:
            nodes = PIPELINE_NODES_QA
        html = '<div class="pipeline-container">'
        for nid, nlabel in nodes:
            if nid == error_node:
                css, icon = "pipeline-node error", "✗"
            elif nid == active_node:
                css, icon = "pipeline-node active", "●"
            elif nid in done_nodes:
                css, icon = "pipeline-node done", "✓"
            else:
                css, icon = "pipeline-node", "○"
            html += f'<div class="{css}"><span class="dot"></span>{icon} {nlabel}</div>'
        html += '</div>'
        pipeline_placeholder.markdown(html, unsafe_allow_html=True)

    render_pipeline()

    st.divider()
    st.markdown('<p class="sidebar-title">Última execução</p>', unsafe_allow_html=True)
    meta_placeholder = st.empty()

    def render_meta(state: dict):
        if not state:
            meta_placeholder.caption("Nenhuma execução ainda.")
            return
        intent = state.get("intent", "—")
        score  = state.get("self_check_score", "—")
        safety = state.get("safety_status", "—")
        bi  = {"qa":"badge-green","automation":"badge-orange","refuse":"badge-red"}.get(intent,"badge-gray")
        bs  = {"approved":"badge-green","approved_with_disclaimer":"badge-orange","blocked":"badge-red"}.get(safety,"badge-gray")
        bsc = "badge-green" if isinstance(score, int) and score >= 3 else ("badge-gray" if score == "—" else "badge-red")
        html = (
            f'<p style="font-size:0.81rem;margin:0.2rem 0;">Intent &nbsp;<span class="badge {bi}">{intent}</span></p>'
            f'<p style="font-size:0.81rem;margin:0.2rem 0;">Self-Check &nbsp;<span class="badge {bsc}">{score}/5</span></p>'
            f'<p style="font-size:0.81rem;margin:0.2rem 0;">Safety &nbsp;<span class="badge {bs}">{safety}</span></p>'
        )
        inner_state = state.get("state", {})
        restrictions = inner_state.get("restrictions", [])
        mcp_foods = inner_state.get("mcp_foods", {})
        if restrictions:
            restr_str = ", ".join(r.replace("_", " ") for r in restrictions)
            html += f'<p style="font-size:0.78rem;margin:0.3rem 0;color:#9a9;">Restrições: {restr_str}</p>'
        if mcp_foods:
            html += f'<p style="font-size:0.78rem;margin:0.2rem 0;color:#9a9;">🔌 MCP: {len(mcp_foods)} alimento(s)</p>'
        meta_placeholder.markdown(html, unsafe_allow_html=True)

    render_meta(st.session_state.pipeline_state)

    st.divider()
    st.markdown('<p class="sidebar-title">Sobre</p>', unsafe_allow_html=True)
    st.caption("Fontes: Guia Alimentar MS, Diretrizes SBD, Tabela TACO, ANVISA, BVS/MS.")
    st.caption("Caráter informativo — não substitui orientação profissional.")

# Área principal — header
st.markdown("# NutriAgents 🥗")
st.caption("Alimentação com restrições de saúde — diabetes, hipertensão, doença celíaca, alergias e mais.")
st.divider()

# Histórico de mensagens
if not st.session_state.messages and not st.session_state.pending_query:
    st.markdown(
        "<div style='text-align:center;padding:3rem 0;'>"
        "<div style='font-size:3rem;'>🥗</div>"
        "<br>"
        "</div>",
        unsafe_allow_html=True
    )
    st.info(
        "Olá! Posso ajudar com dúvidas sobre alimentação saudável considerando restrições "
        "como diabetes, hipertensão, doença celíaca e alergias.\n\n"
        "**Exemplos:** *\"O que celíacos não podem comer?\"* · *\"Alimentos que hipertensos devem evitar\"*"
    )
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # Balão escuro para o usuário (HTML com cores explícitas)
            st.markdown(
                f'<p class="msg-label" style="color:#999;">Você</p>'
                f'<div class="msg-user">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            # Resposta do assistente — componente nativo, cor do tema
            st.markdown(f'<p class="msg-label" style="color:#999;">NutriAgents</p>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(msg["content"])

            if msg.get("chunks"):
                with st.expander(f"📄  Ver {len(msg['chunks'])} fonte(s) consultada(s)"):
                    for i, chunk in enumerate(msg["chunks"], 1):
                        st.markdown(
                            f"**[{i}] {chunk.get('title','Documento')}**"
                            f" — p. {chunk.get('page','?')}"
                            f" | score: `{chunk.get('score','?')}`"
                        )
                        st.caption(chunk.get("text","")[:300] + "...")
                        if i < len(msg["chunks"]):
                            st.divider()

            if msg.get("mcp_foods"):
                with st.expander(f"🔌  Ver {len(msg['mcp_foods'])} alimento(s) consultado(s) via MCP OpenNutrition"):
                    for food_name, summary in msg["mcp_foods"].items():
                        st.markdown(summary)
                        st.divider()

# Processar query pendente (segundo rerun — mensagem do usuário já visível)
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

    # Detecta rota para animação do pipeline
    query_lower = query.lower()
    is_auto = any(w in query_lower for w in ["gera", "cria", "monta", "cardápio", "plano", "dieta", "elabora"])
    pipeline_nodes = PIPELINE_NODES_AUTO if is_auto else PIPELINE_NODES_QA
    spinner_msg = "Gerando plano alimentar com MCP OpenNutrition..." if is_auto else "Consultando documentos nutricionais..."

    with st.spinner(spinner_msg):
        try:
            render_pipeline(active_node="supervisor", done_nodes=[], nodes=pipeline_nodes)
            result = run(query)
            intent = result.get("intent", "qa")
            pipeline_nodes = PIPELINE_NODES_AUTO if intent == "automation" else PIPELINE_NODES_QA
            final_response = result.get("final_response", "Não foi possível gerar uma resposta.")
            chunks = result.get("state", {}).get("chunks", [])
            mcp_foods = result.get("state", {}).get("mcp_foods", {})
            render_pipeline(done_nodes=[n[0] for n in pipeline_nodes], nodes=pipeline_nodes)
            st.session_state.pipeline_state = result
            msg_data = {"role": "assistant", "content": final_response, "chunks": chunks}
            if mcp_foods:
                msg_data["mcp_foods"] = mcp_foods
            st.session_state.messages.append(msg_data)
        except Exception as e:
            render_pipeline(error_node="supervisor", nodes=pipeline_nodes)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Erro: `{e}`\n\nVerifique se o Ollama está rodando.",
                "chunks": [],
            })

    render_meta(st.session_state.pipeline_state)
    st.rerun()

# Botão limpar + chat input
if st.session_state.messages:
    _, col_clear = st.columns([11, 1])
    with col_clear:
        if st.button("🗑", help="Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pipeline_state = {}
            st.rerun()

user_input = st.chat_input("Digite sua pergunta sobre alimentação...")

if user_input and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    st.session_state.pending_query = user_input.strip()
    st.rerun()