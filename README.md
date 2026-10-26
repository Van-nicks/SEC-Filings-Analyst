# SEC Filings Analyst

Work in progress: GraphRAG pipeline for SEC 10-K filings (EDGAR ingestion, Neo4j graph, Bedrock LLM, LangGraph agent)

Status: Ingestion, chunking, vector index, and Neo4j knowledge graph (`Company`, `Filing`, `Fact`, `Chunk`, `RiskCategory`) complete; LangGraph agent next.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # then fill in the values
```

## Required environment variables

Set these in `.env` (see `.env.example`):

- `AWS_BEARER_TOKEN_BEDROCK`
- `AWS_DEFAULT_REGION`
- `EDGAR_IDENTITY`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

## Run the fetch

Downloads the latest 10-K business and risk-factor sections to `data/raw/` and key financial facts to `data/processed/facts.csv`:

```bash
python -m filing_analyst.ingest.sec_fetch
```

## Build the vector index (run once)

Chunks the 10-K text in `data/raw/*.txt` (with metadata from `*_meta.json`), embeds it with Bedrock Titan and stores it in Neo4j as `Chunk` nodes. Skips automatically if the index is already built; use `--force` to wipe the `Chunk` nodes and rebuild:

```bash
python -m filing_analyst.graph.build_index [--force]
```

## Connect the knowledge graph & risk categories (run once)

Loads `Company`, `Filing`, and numeric `Fact` nodes from `data/processed/facts.csv`, classifies `risk_factors` chunks into `RiskCategory` nodes (cached in `data/processed/risk_labels.jsonl`), and links `:FILED`, `:REPORTS`, `:HAS_CHUNK`, and `:MENTIONS_RISK`:

```bash
python -m filing_analyst.graph.load_facts
```

## Query (every request)

Connects to the existing graph/index and embeds only the question text:

```python
from filing_analyst.graph.retriever import retrieve, search_chunks, query_facts, query_companies_by_risk

docs = retrieve("What was Apple's revenue in 2024?")
facts = query_facts(ticker="AAPL", metric="revenue")
risks = query_companies_by_risk("supply_chain")
chunks = search_chunks("China GPU restrictions", ticker="NVDA", risk_category="export_controls_and_geopolitics")
```

Quick manual test: `python -m filing_analyst.graph.retriever`

## Naming

The names differ on purpose and should be left as they are:

- GitHub repo: `SEC-Filings-Analyst`
- Project folder: `filings_analyst`
- Python package (import name): `filing_analyst` (in `src/filing_analyst`)
- Distribution name in `pyproject.toml`: `filings-analyst`
