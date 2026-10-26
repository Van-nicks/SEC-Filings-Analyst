# SEC Filings Analyst — Multi-Agent GraphRAG

An end-to-end **GraphRAG** and **Multi-Agent** financial analysis system built over SEC Form 10-K filings for **Apple (`AAPL`)**, **Microsoft (`MSFT`)**, **NVIDIA (`NVDA`)**, **Amazon (`AMZN`)**, and **Tesla (`TSLA`)**.

The project combines a **Neo4j Knowledge Graph** (structured XBRL financial facts + vector-indexed 10-K text chunks + risk classifications) with a **LangGraph Supervisor** orchestrating specialized research and financial analysis agents powered by **Amazon Bedrock (Claude Sonnet 5.5 & Titan Text Embeddings v2)**.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.14 | Core implementation |
| **LLM (Agents & Judge)** | Amazon Bedrock — Claude Sonnet 5.5 (`eu.anthropic.claude-sonnet-5-5`) | Multi-agent reasoning, synthesis, and Ragas evaluation |
| **Embeddings** | Amazon Bedrock — Titan Text Embeddings v2 (`amazon.titan-embed-text-v2:0`) | 1024-dim semantic chunk & query embeddings |
| **Agent Orchestration** | LangGraph & `langgraph-supervisor` | Multi-agent `supervisor` -> `researcher` -> `analyst` workflow & memory |
| **LLM Framework** | LangChain (`langchain`, `langchain-aws`, `langchain-neo4j`) | Tool binding, Bedrock Converse streaming, and vector store integration |
| **Graph & Vector DB** | Neo4j 5.x (Docker) | Knowledge graph (`Company`, `Filing`, `Fact`, `Chunk`, `RiskCategory`) + HNSW vector index |
| **SEC Data Ingestion** | `edgartools`, BeautifulSoup4, `pandas` | SEC EDGAR Form 10-K text & XBRL financial fact extraction |
| **Evaluation & Testing** | Ragas, Hugging Face `datasets`, `pytest` | RAG faithfulness/relevancy/precision/recall benchmarking & unit testing |

---

## Architecture Overview

1. **SEC EDGAR Ingestion (`filing_analyst.ingest`)**
   - Downloads the latest Form 10-K filings (`Item 1 Business`, `Item 1A Risk Factors`, and structured XBRL financial facts: `revenue`, `net_income`, `total_assets`).
   - Splits filing sections into 819 deterministic chunks (`chunks.jsonl`) with exact character offsets and stable IDs (`<ticker>-<accession>-<section>-<NNN>`).

2. **Neo4j Knowledge Graph & Vector Index (`filing_analyst.graph`)**
   - **Nodes:** `Company` (5), `Filing` (5), `Fact` (30), `Chunk` (819), `RiskCategory` (10).
   - **Relationships:**
     - `(:Company)-[:FILED]->(:Filing)`
     - `(:Filing)-[:REPORTS]->(:Fact)`
     - `(:Filing)-[:HAS_CHUNK]->(:Chunk)`
     - `(:Chunk)-[:MENTIONS_RISK]->(:RiskCategory)`
   - **Hybrid Retrieval:** Combines Titan v2 semantic vector search (`chunk_embeddings`) with structured Cypher graph traversal (`search_chunks`, `query_facts`, `query_companies_by_risk`).

3. **Multi-Agent System (`filing_analyst.agent`)**
   - **`supervisor`**: Routes user queries, enforces strict brevity, and streams clean final answers.
   - **`researcher`**: Queries the Neo4j knowledge graph first (`GRAPH-FIRST RULE`) and explicitly tags any out-of-graph fallback knowledge.
   - **`analyst`**: Performs exact financial arithmetic (YoY growth, net profit margins, multi-company aggregations) via an AST-validated `calculator` tool.
   - **Interactive CLI**: Supports clean word-by-word answer streaming by default and a live `/verbose` toggle to inspect the full multi-agent trace, tool calls, and `[chunk_id]` / `[fact_id]` citations.

4. **Testing & Ragas Evaluation (`tests/` & `eval/`)**
   - **Unit Tests (`tests/test_chunk.py`)**: Verifies chunk ID uniqueness, contiguous numbering, minimum chunk length, and exact source-file character offsets.
   - **Ragas Evaluation (`eval/eval_ragas.py`)**: Evaluates `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall` across a 30-question golden dataset (`eval/eval_dataset.json`).

---

## Project Structure

```text
filings_analyst/
├── data/
│   ├── raw/                        # Raw 10-K sections (*_business.txt, *_risk_factors.txt, *_meta.json)
│   └── processed/                  # facts.csv, chunks.jsonl, risk_labels.jsonl
├── eval/
│   ├── eval_dataset.json           # 30-question golden benchmark dataset verified against 10-K filings
│   ├── eval_ragas.py               # Ragas evaluation script (faithfulness, relevancy, precision, recall)
│   └── eval_results.json           # Detailed per-question Ragas evaluation output
├── src/filing_analyst/
│   ├── config.py                   # Central configuration, paths, model IDs, and constants
│   ├── ingest/
│   │   ├── sec_fetch.py            # SEC EDGAR 10-K text & XBRL fact downloader
│   │   └── chunk.py                # Deterministic 10-K text chunker
│   ├── graph/
│   │   ├── build_index.py          # Idempotent Neo4j vector index builder (Titan v2 embeddings)
│   │   ├── load_facts.py           # Knowledge graph loader (Company, Filing, Fact, RiskCategory)
│   │   └── retriever.py            # Hybrid vector + Cypher graph retrieval tools
│   ├── llm/
│   │   └── models.py               # Cached Amazon Bedrock chat & embedding clients
│   └── agent/
│       └── agents.py               # LangGraph supervisor, researcher, analyst & interactive CLI
├── tests/
│   └── test_chunk.py               # Unit tests for 10-K chunking pipeline
├── pyproject.toml
├── requirements.txt
└── requirements.lock.txt
```

---

## Setup & Installation

### 1. Create Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and populate your credentials:

```bash
cp .env.example .env
```

Required variables in `.env`:
- `AWS_BEARER_TOKEN_BEDROCK` (or standard AWS credentials)
- `AWS_DEFAULT_REGION`
- `EDGAR_IDENTITY` (e.g., `YourName your.email@example.com`)
- `NEO4J_URI` (e.g., `bolt://localhost:7687`)
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

---

## End-to-End Pipeline Usage

### Step 1: Fetch SEC 10-K Filings
Downloads the latest 10-K `business` and `risk_factors` sections to `data/raw/` and structured XBRL metrics to `data/processed/facts.csv`:

```bash
python -m filing_analyst.ingest.sec_fetch
```

### Step 2: Build the Vector Index (Run Once)
Chunks the 10-K text, embeds each chunk with Bedrock Titan v2, and creates the `chunk_embeddings` vector index in Neo4j. Automatically skips if all 819 chunks are already indexed (pass `--force` to rebuild):

```bash
python -m filing_analyst.graph.build_index
```

### Step 3: Populate the Knowledge Graph & Risk Categories (Run Once)
Loads `Company`, `Filing`, `Fact`, and `RiskCategory` nodes into Neo4j and links `:FILED`, `:REPORTS`, `:HAS_CHUNK`, and `:MENTIONS_RISK` edges:

```bash
python -m filing_analyst.graph.load_facts
```

### Step 4: Run the Interactive Multi-Agent CLI
Start the interactive assistant:

```bash
python -m filing_analyst.agent.agents
```

- **Default Mode (`verbose=OFF`)**: Displays concise status updates while sub-agents query Neo4j and streams a clean final answer.
- **Trace Mode (`/verbose`)**: Type `/verbose` at the prompt to toggle the full multi-agent trace, tool invocations, and `[chunk_id]` / `[fact_id]` source citations.
- **Quit**: Type `q` or `exit`.

---

## Testing & Evaluation

### Run Unit Tests
Verifies chunking invariants, contiguous IDs, and exact character offsets:

```bash
pytest tests/test_chunk.py -v
```

### Run Ragas Evaluation
Evaluates the multi-agent system against the 30-question golden benchmark dataset (`eval/eval_dataset.json`):

```bash
python eval/eval_ragas.py
```

#### Benchmark Results (`eval/eval_results.json`)

| Metric | In-Scope 10-K Questions (`Q01`–`Q25`) | All 30 Questions (incl. Out-of-Graph Boundary Tests) |
|---|---|---|
| **Faithfulness** | **0.9723 (97.2%)** | 0.9020 (90.2%) |
| **Context Recall** | **0.9600 (96.0%)** | 0.9556 (95.6%) |
| **Answer Relevancy** | **0.9126 (91.3%)** | 0.8133 (81.3%) |
| **Context Precision** | **0.8933 (89.3%)** | 0.8944 (89.4%) |

---

## Naming Conventions

- **GitHub Repository**: `SEC-Filings-Analyst`
- **Project Directory**: `filings_analyst`
- **Python Package (import name)**: `filing_analyst` (`src/filing_analyst`)
- **Distribution Name (`pyproject.toml`)**: `filings-analyst`
