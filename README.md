# 🥗 NutriAgents

https://docs.google.com/presentation/d/1HGShT7fhpTPOIVgXc8tK2Kv0O6aEcqK-L0RqUWIqdpQ/edit?slide=id.g3d067735a21_0_120#slide=id.g3d067735a21_0_120

> Sistema multiagente de assistência nutricional para pessoas com restrições alimentares, baseado em documentos públicos brasileiros de saúde.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-LangGraph-green)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![LLM](https://img.shields.io/badge/LLM-Ollama%20qwen2.5%3A3b-orange)](https://ollama.com)

---

## 📋 Índice

- [Problema e público-alvo](#problema-e-público-alvo)
- [Arquitetura](#arquitetura)
- [Stack técnica](#stack-técnica)
- [Instalação](#instalação)
- [Como usar](#como-usar)
- [Agentes](#agentes)
- [Automação](#automação)
- [MCP — mcp-opennutrition](#mcp--mcp-opennutrition)
- [Avaliação](#avaliação)
- [Limitações e próximos passos](#limitações-e-próximos-passos)
- [Fontes documentais](#fontes-documentais)

---

## Problema e público-alvo

Pessoas com restrições alimentares (diabetes, doença celíaca, hipertensão, intolerância à lactose, entre outras) enfrentam dificuldade em encontrar orientações nutricionais confiáveis, baseadas em evidências e adaptadas à sua condição. O acesso a nutricionistas é limitado no Brasil, especialmente para populações vulneráveis.

O **NutriAgents** oferece:
- Respostas a perguntas sobre nutrição com **citações de documentos públicos**
- Geração de **planos alimentares semanais personalizados** com dados nutricionais reais
- Mecanismo **anti-alucinação** via Self-RAG
- **Disclaimer obrigatório** em todas as respostas (não substitui orientação profissional)

**Público-alvo:** pessoas com restrições alimentares, cuidadores, estudantes de nutrição e profissionais de saúde que queiram uma segunda opinião baseada em evidências.

---

## Arquitetura

```
Usuário
   │
   ▼
┌─────────────┐
│  Supervisor  │ ──→ classifica intent: qa / automation / refuse
└──────┬──────┘
       │
   ┌───┴────────────────────────┐
   │ QA                         │ Automation
   ▼                            ▼
┌──────────┐            ┌───────────────┐
│ Retriever│            │ Automation    │
│  (FAISS) │            │    Agent      │
└────┬─────┘            │  ┌─────────┐ │
     │                  │  │   RAG   │ │
     ▼                  │  └────┬────┘ │
┌──────────┐            │       │      │
│ Answerer │            │  ┌────▼────┐ │
│(citações)│            │  │   MCP   │ │
└────┬─────┘            │  │OpenNutr.│ │
     │                  │  └─────────┘ │
     ▼                  └──────┬───────┘
┌──────────┐                   │
│Self-Check│◄──────────────────┘
│(Self-RAG)│
└────┬─────┘
     │
     ▼
┌──────────┐
│  Safety  │ ──→ disclaimer + bloqueio de conteúdo perigoso
└────┬─────┘
     │
     ▼
  Resposta final
```

O grafo é orquestrado via **LangGraph**. O Self-Check pode disparar uma re-busca (máximo 1 tentativa) caso a resposta não esteja suficientemente suportada pelas evidências.

---

## Stack técnica

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.10+ |
| Orquestração de agentes | LangChain + LangGraph |
| LLM | Ollama (`qwen2.5:3b`) — local, open source |
| Embeddings | HuggingFace `BAAI/bge-m3` |
| Vector store | FAISS (local, sem SaaS) |
| MCP | `mcp-opennutrition` (Node.js, local) |
| Interface | Streamlit |
| Avaliação | Métricas próprias + RAGAS (opcional) |

---

## Instalação

### Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.com) instalado e rodando
- Node.js 18+ (para o servidor MCP)

### 1. Clone o repositório

```bash
git clone https://github.com/VictorCSA/NutriAgents
cd NutriAgents
```

### 2. Instale as dependências Python

```bash
pip install -r requirements.txt
```

### 3. Baixe o modelo LLM

```bash
ollama pull qwen2.5:3b
```

### 4. Instale e compile o servidor MCP

```bash
git clone https://github.com/deadletterq/mcp-opennutrition
cd mcp-opennutrition
npm install

# Windows
npm run build

# Linux/Mac
npm run build
```

> **Importante:** coloque o arquivo `opennutrition-dataset-2025.1.zip` na pasta `mcp-opennutrition/data/` antes do build. O script `convert-data` usa esse arquivo para criar o banco SQLite.

### 5. Execute o pipeline de ingestão

```bash
python ingest/pipeline.py
```

### 6. Rode a aplicação

```bash
 python .\run_streamlit_patched.py 
```

---

## Como usar

**Perguntas (rota QA):**
- *"Quais alimentos um diabético deve evitar?"*
- *"O que celíacos não podem comer?"*
- *"Posso comer arroz e feijão tendo hipertensão?"*

**Geração de plano alimentar (rota Automation):**
- *"Gere um plano alimentar de 7 dias para celíaco"*
- *"Crie um cardápio semanal para diabético tipo 2"*
- *"Monte um plano para celíaco com hipertensão"*
- *"Elabore uma dieta sem lactose e vegetariana"*

---

## Agentes

### Supervisor
Classifica a mensagem do usuário em três intenções: `qa`, `automation` ou `refuse`. Usa o LLM com um prompt determinístico (temperature=0) e um pré-filtro por regex para cobrir variações de conjugação ("gere", "crie", "monte", etc.).

### Retriever
Busca densa no índice FAISS usando embeddings `BAAI/bge-m3`. Retorna os 5 chunks mais relevantes com metadados (título, fonte, página, score de similaridade).

### Answerer
Gera a resposta formatada em Markdown com citações inline `[1]`, `[2]`... usando exclusivamente os chunks recuperados. Nunca usa conhecimento externo.

### Self-Check (Self-RAG)
Avalia se as afirmações da resposta estão suportadas pelas evidências recuperadas, atribuindo um score de 1 a 5. Se o score for menor que 3, dispara uma re-busca (máximo 1 tentativa). Se falhar na segunda tentativa, recusa a responder.

### Automation Agent
Gera planos alimentares semanais em 9 chamadas separadas ao LLM (cabeçalho + 7 dias + rodapé), passando as refeições já geradas para evitar repetição. Integra com o MCP OpenNutrition para enriquecer o plano com dados nutricionais reais.

### Safety Agent
Detecta padrões proibidos (prescrições de dosagens, diagnósticos, promessas de cura) e bloqueia a resposta. Em todos os casos aprovados, injeta disclaimer obrigatório.

---

## Automação

O sistema implementa uma rota de automação completa para geração de planos alimentares:

1. **Detecção de restrições** — extrai restrições alimentares da mensagem via regex
2. **Consulta RAG** — busca diretrizes nutricionais no corpus
3. **Consulta MCP** — o LLM decide quais alimentos buscar (em inglês) no banco OpenNutrition
4. **Geração dia a dia** — 7 chamadas LLM separadas com memória de refeições anteriores
5. **Safety** — disclaimer obrigatório no rodapé

---

## MCP — mcp-opennutrition

### Servidor utilizado

**[mcp-opennutrition](https://github.com/deadletterq/mcp-opennutrition)** — servidor MCP local (TypeScript/Node.js) com 300.000+ alimentos do banco OpenNutrition (USDA, CNF, FRIDA, AUSNUT). Sem chamadas externas após instalação.

### Justificativa da escolha

O mcp-opennutrition foi escolhido por fornecer dados nutricionais reais (macros, vitaminas, minerais por 100g) que fundamentam as recomendações do plano alimentar com informações verificáveis, não apenas texto do LLM. Roda 100% localmente, alinhado com o espírito open source do projeto.

### Tools expostas (allowlist)

| Tool | Descrição | Parâmetros |
|------|-----------|------------|
| `search-food-by-name` | Busca alimentos por nome | `query: str`, `limit: int (1-20)` |
| `get-food-by-id` | Perfil nutricional completo por ID `fd_...` | `id: str` |
| `browse-foods` | Listagem paginada do catálogo | `page: int`, `pageSize: int` |
| `barcode-lookup` | Busca por código de barras EAN-13 | `barcode: str` |

### Controles de segurança

| Controle | Implementação |
|---|---|
| **Allowlist** | Apenas as 4 tools acima são permitidas. Qualquer outra levanta `ValueError` antes de chegar ao servidor |
| **Validação de ID** | IDs devem começar com `fd_` — IDs internos do LangChain (`lc_...`) são rejeitados |
| **Log de auditoria** | Todas as chamadas registradas em `logs/mcp_calls.jsonl` com timestamp, tool, parâmetros e resultado |
| **Sem acesso a arquivos** | O servidor não tem acesso ao sistema de arquivos do projeto |
| **Offline** | Sem chamadas HTTP externas após instalação |
| **Degradação graciosa** | Se o servidor MCP estiver offline, o plano é gerado sem dados nutricionais — sem travar |

**O agente NÃO pode via este MCP:**
- ❌ Ler ou escrever arquivos
- ❌ Executar comandos shell
- ❌ Fazer requisições HTTP
- ❌ Modificar o banco de dados
- ❌ Acessar dados pessoais do usuário

### Riscos conhecidos

| Risco | Mitigação |
|---|---|
| Supply-chain | Servidor compilado localmente do repositório oficial |
| Exfiltração | Servidor offline, parâmetros validados antes do envio |
| Loop infinito | Limite de 20 buscas por execução |
| Prompt injection via MCP | Resultados usados como contexto formatado, não executados |

---

## Avaliação

### 1. RAG — 15 perguntas rotuladas

| Métrica | Valor | Interpretação |
|---|---|---|
| Perguntas avaliadas | 15/15 | 100% sem erros de pipeline |
| **Context Recall** | **0.652** | Chunks cobrem ~65% dos tópicos esperados |
| **Answer Relevancy** | **0.580** | Respostas abordam ~58% dos tópicos esperados |
| **Faithfulness** | **0.813** | 81% das respostas aprovadas pelo Self-Check — alta aderência às evidências |
| Latência média | 42.7s | Esperado em CPU local com modelo 3B + embeddings bge-m3 |
| Chunks recuperados | 5.0 | Top-K fixo |

> **Nota metodológica:** Context Recall e Answer Relevancy são calculados como a fração de tópicos esperados encontrados nos chunks/resposta. Faithfulness = `self_check_score / 5`. RAGAS oficial não utilizado nesta versão por limitação de infraestrutura.

**Destaques por pergunta:**

| Pergunta | Context Recall | Faithfulness |
|---|---|---|
| "O que celíacos não podem comer?" | 1.0 | 1.0 |
| "Qual o papel das fibras para diabéticos?" | 1.0 | 0.8 |
| "Como substituir o trigo na alimentação celíaca?" | 1.0 | 0.8 |
| "Quais alimentos têm alto índice glicêmico?" | 0.0 | 0.8 |
| "Sintomas de contaminação por glúten" | 0.0 | 0.8 |

> Context Recall 0.0 indica que o corpus não contém esses tópicos explicitamente, oportunidade de expansão.

---

### 2. Automação — 5 tarefas

| Métrica | Valor |
|---|---|
| Tarefas avaliadas | 5 |
| **Taxa de sucesso** | **4/5 (80%)** |
| Tempo médio por plano | 108.4s |
| Steps médios (estimado) | 14 (9 LLM + 5 RAG) |
| Variação de refeições | 100% de linhas únicas |
| Alimentos MCP por plano | 14.4 |

**Resultados por tarefa:**

| ID | Descrição | Sucesso | Dias | Tempo | MCP foods |
|---|---|---|---|---|---|
| auto_01 | Celíaco simples | ❌* | 7/7 | 123s | 15 |
| auto_02 | Diabético tipo 2 | ✅ | 7/7 | 98s | 14 |
| auto_03 | Celíaco + hipertenso | ✅ | 7/7 | 111s | 14 |
| auto_04 | Sem lactose + vegetariano | ✅ | 7/7 | 116s | 15 |
| auto_05 | Hipertenso baixo sódio | ✅ | 7/7 | 94s | 14 |

> \* auto_01 falhou pelo critério `contains_forbidden_keywords` — o modelo citou "glúten" em contexto explicativo ("evite glúten"), não como recomendação. O plano em si estava correto (7 dias, 4 refeições, 100% variação).

---

### 3. MCP — 6 testes de segurança

| Teste | Resultado |
|---|---|
| Servidor disponível | ✅ |
| `search-food-by-name` retorna dados | ✅ (1.7s, 5 resultados) |
| `get-food-by-id` com ID válido (`fd_...`) | ✅ (0.35s) |
| `get-food-by-id` com ID inválido (`lc_...`) rejeitado | ✅ |
| Allowlist bloqueia tool proibida (`execute_shell`) | ✅ |
| Log de auditoria existe e tem entradas | ✅ (479 entradas) |

**6/6 testes passaram.**

---

## Limitações e próximos passos

**Limitações atuais:**
- Latência alta (~40s QA, ~108s automação) por uso de CPU local com modelo 3B
- `qwen2.5:3b` tem janela de contexto pequena, limitando a complexidade das respostas
- Corpus restrito a documentos brasileiros disponíveis publicamente — perguntas muito específicas podem não ter cobertura
- Dataset OpenNutrition em inglês — tradução PT→EN pode não cobrir todos os alimentos

**Próximos passos:**
- Migrar para `qwen2.5:7b` ou `llama3.2:3b` para melhorar qualidade das respostas
- Expandir corpus com mais documentos (ANVISA, CFN, publicações do IBGE)
- Implementar reranking dos chunks (CrossEncoder) para melhorar Context Precision
- Adicionar avaliação RAGAS oficial com dataset rotulado por nutricionista
- Cache de embeddings para reduzir latência de primeira resposta
- Containerizar com Docker para facilitar deploy

---

## Fontes documentais

O corpus indexado inclui documentos públicos brasileiros de saúde:

- **Guia Alimentar para a População Brasileira** — Ministério da Saúde, 2014
- **Diretrizes da Sociedade Brasileira de Diabetes (SBD)** — Nutrição, 2023
- **Tabela Brasileira de Composição de Alimentos (TACO)** — UNICAMP
- **Manual de Doença Celíaca** — FENACELBRA / Ministério da Saúde
- **Protocolo Clínico de Hipertensão Arterial** — MS/RENAME

---

## Estrutura do projeto

```
NutriAgents/
├── app/
│   └── streamlit_app.py        # Interface Streamlit
├── src/
│   ├── agents/
│   │   ├── supervisor.py       # Roteador de intents
│   │   ├── retriever.py        # Busca FAISS
│   │   ├── answerer.py         # Geração com citações
│   │   ├── self_check.py       # Self-RAG anti-alucinação
│   │   ├── automation.py       # Geração de planos alimentares
│   │   └── safety.py           # Disclaimer e bloqueio
│   ├── graph/
│   │   └── graph.py            # Grafo LangGraph
│   └── nutritools/
│       └── opennutrition_client.py  # Cliente MCP
├── ingest/
│   ├── pipeline.py             # Pipeline de ingestão
│   ├── extract.py
│   ├── clean.py
│   ├── chunk.py
│   └── embed_and_index.py
├── eval/
│   ├── evaluate.py             # Script de avaliação
│   └── results/                # Resultados JSON + summary.md
├── data/
│   ├── raw/                    # PDFs e CSVs originais
│   └── processed/              # Chunks, embeddings, índice FAISS
├── logs/
│   └── mcp_calls.jsonl         # Log de auditoria MCP
├── mcp-opennutrition/          # Servidor MCP (submódulo)
├── requirements.txt
└── LICENSE
```

---

## Aviso

> As informações fornecidas pelo NutriAgents têm caráter **exclusivamente informativo** e são baseadas em documentos públicos de saúde. **Não substituem consulta com nutricionista, médico ou outro profissional de saúde habilitado.**

---

## Licença

MIT License — veja [LICENSE](LICENSE).

## Autores

- Victor Carneiro Santos Angelo
- Pedro Henrique Veloso Fernandes
