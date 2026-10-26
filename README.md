# SEC Filings Analyst

Work in progress: GraphRAG pipeline for SEC 10-K filings (EDGAR ingestion, Neo4j graph, Bedrock LLM, LangGraph agent)

Status: Objective 1 (fetch) done; chunking and graph build in progress

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

Embeds `data/processed/facts.csv` with Bedrock Titan and stores it in Neo4j. Skips automatically if the index is already built; use `--force` to wipe the `FactChunk` nodes and rebuild:

```bash
python -m filing_analyst.graph.build_index [--force]
```

## Query (every request)

Connects to the existing index and embeds only the question text:

```python
from filing_analyst.graph.retriever import retrieve
docs = retrieve("What was Apple's revenue in 2024?")
```

Quick manual test: `python -m filing_analyst.graph.retriever`

## Naming

The names differ on purpose and should be left as they are:

- GitHub repo: `SEC-Filings-Analyst`
- Project folder: `filings_analyst`
- Python package (import name): `filing_analyst` (in `src/filing_analyst`)
- Distribution name in `pyproject.toml`: `filings-analyst`
