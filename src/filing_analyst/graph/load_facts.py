"""Populate Company, Filing, Fact, and RiskCategory nodes in Neo4j and connect all Chunk nodes.

Graph schema created:
    (:Company {ticker, name})-[:FILED]->(:Filing {accession, ticker, form, filing_date, period_of_report, fiscal_year})
    (:Filing)-[:HAS_CHUNK]->(:Chunk)
    (:Filing)-[:REPORTS]->(:Fact {fact_id, ticker, metric, fiscal_year, value, unit, filing_date, accession})
    (:Chunk)-[:MENTIONS_RISK]->(:RiskCategory {name, description})

Risk labels are cached in data/processed/risk_labels.jsonl so Bedrock is only called
once per chunk; subsequent runs load straight from disk with 0 LLM calls.

Run:
    python -m filing_analyst.graph.load_facts
"""
import argparse
import csv
import json
import re

from neo4j import GraphDatabase

from filing_analyst.config import (
    CHUNKS_JSONL,
    COMPANIES,
    FACTS_CSV,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    RISK_CATEGORIES,
    RISK_LABELS_JSONL,
)
from filing_analyst.ingest.chunk import load_and_split, load_meta, save_chunks_jsonl
from filing_analyst.llm.models import get_chat_llm

CONSTRAINTS = [
    "CREATE CONSTRAINT company_ticker_unique IF NOT EXISTS FOR (co:Company) REQUIRE co.ticker IS UNIQUE",
    "CREATE CONSTRAINT filing_accession_unique IF NOT EXISTS FOR (f:Filing) REQUIRE f.accession IS UNIQUE",
    "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (fa:Fact) REQUIRE fa.fact_id IS UNIQUE",
    "CREATE CONSTRAINT risk_category_name_unique IF NOT EXISTS FOR (rc:RiskCategory) REQUIRE rc.name IS UNIQUE",
]

BATCH_SIZE = 15


def read_facts_csv() -> list[dict]:
    """Read data/processed/facts.csv and coerce numeric fields (fiscal_year -> int, value -> float)."""
    rows = []
    with FACTS_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ticker = r["ticker"].strip()
            metric = r["metric"].strip()
            fiscal_year = int(r["fiscal_year"])
            rows.append({
                "fact_id": f"{ticker}-{metric}-{fiscal_year}",
                "ticker": ticker,
                "metric": metric,
                "fiscal_year": fiscal_year,
                "value": float(r["value"]),
                "unit": "USD",
                "filing_date": r["filing_date"].strip(),
                "accession": r["accession"].strip(),
            })
    return rows


def _load_cached_risk_labels() -> dict[str, list[str]]:
    """Return {chunk_id: [category, ...]} already saved in data/processed/risk_labels.jsonl."""
    if not RISK_LABELS_JSONL.exists():
        return {}
    cached: dict[str, list[str]] = {}
    with RISK_LABELS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                cached[obj["chunk_id"]] = [c for c in obj["categories"] if c in RISK_CATEGORIES]
    return cached


def _classify_batch(batch: list[dict]) -> dict[str, list[str]]:
    """Classify a batch of risk_factors chunks into 1-3 RISK_CATEGORIES using Bedrock."""
    taxonomy_lines = "\n".join(f"- {name}: {desc}" for name, desc in RISK_CATEGORIES.items())
    items_text = "\n\n".join(
        f"ID: {item['chunk_id']}\nTEXT: {item['text'][:900]}" for item in batch
    )
    prompt = (
        "You are classifying SEC 10-K Item 1A Risk Factor excerpts into a fixed taxonomy.\n"
        "Allowed risk category names:\n"
        f"{taxonomy_lines}\n\n"
        "For EACH chunk below, assign 1 to 3 category names from the allowed list above that best match the excerpt.\n"
        "Return ONLY a valid JSON object mapping each chunk ID to its list of category strings, "
        'for example: {"AAPL-...-000": ["supply_chain", "competition_and_demand"]}.\n\n'
        f"{items_text}"
    )
    llm = get_chat_llm(temperature=0.0, max_tokens=2048)
    raw = str(llm.invoke(prompt).content)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    parsed = json.loads(match.group(0) if match else raw)
    result: dict[str, list[str]] = {}
    for item in batch:
        cid = item["chunk_id"]
        cats = [c for c in parsed.get(cid, []) if c in RISK_CATEGORIES]
        result[cid] = cats
    return result


def ensure_risk_labels(force_relabel: bool = False) -> dict[str, list[str]]:
    """Ensure every risk_factors chunk has risk category labels cached in risk_labels.jsonl."""
    if not CHUNKS_JSONL.exists():
        save_chunks_jsonl(load_and_split())

    risk_chunks = []
    with CHUNKS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["section"] == "risk_factors":
                risk_chunks.append(obj)

    if force_relabel and RISK_LABELS_JSONL.exists():
        RISK_LABELS_JSONL.unlink()

    cached = _load_cached_risk_labels()
    todo = [c for c in risk_chunks if c["chunk_id"] not in cached]

    if not todo:
        print(f"Risk labels already cached for all {len(cached)} risk_factors chunks (0 Bedrock calls).")
        return cached

    print(f"Labeling {len(todo)} risk_factors chunks via Bedrock in batches of {BATCH_SIZE}...")
    RISK_LABELS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RISK_LABELS_JSONL.open("a", encoding="utf-8") as out:
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i : i + BATCH_SIZE]
            batch_labels = _classify_batch(batch)
            for cid, cats in batch_labels.items():
                cached[cid] = cats
                out.write(json.dumps({"chunk_id": cid, "categories": cats}) + "\n")
            out.flush()
            print(f"  labeled {min(i + BATCH_SIZE, len(todo))}/{len(todo)} chunks")

    return cached


def load_graph(drop_legacy_fact_chunks: bool = True, force_relabel: bool = False) -> dict:
    """Create Company, Filing, Fact, and RiskCategory nodes, link everything, and return counts."""
    meta_by_ticker = load_meta()
    filings = [
        {
            "ticker": ticker,
            "name": COMPANIES.get(ticker, ticker),
            "accession": m["accession"],
            "filing_date": m["filing_date"],
            "period_of_report": m["period_of_report"],
            "fiscal_year": int(m["fiscal_year"]),
            "form": m["form"],
        }
        for ticker, m in sorted(meta_by_ticker.items())
    ]
    facts = read_facts_csv()
    risk_labels_map = ensure_risk_labels(force_relabel=force_relabel)
    risk_categories = [{"name": k, "description": v} for k, v in RISK_CATEGORIES.items()]
    risk_edges = [
        {"chunk_id": cid, "category": cat}
        for cid, cats in risk_labels_map.items()
        for cat in cats
    ]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver:
        for stmt in CONSTRAINTS:
            driver.execute_query(stmt)

        # 1. Company + Filing nodes and (:Company)-[:FILED]->(:Filing)
        driver.execute_query(
            """
            UNWIND $filings AS row
            MERGE (co:Company {ticker: row.ticker})
            SET co.name = row.name
            MERGE (f:Filing {accession: row.accession})
            SET f.ticker = row.ticker,
                f.form = row.form,
                f.filing_date = row.filing_date,
                f.period_of_report = row.period_of_report,
                f.fiscal_year = row.fiscal_year
            MERGE (co)-[:FILED]->(f)
            """,
            filings=filings,
        )

        # 2. Fact nodes (structured numeric values, no embedding) and (:Filing)-[:REPORTS]->(:Fact)
        driver.execute_query(
            """
            UNWIND $facts AS row
            MATCH (f:Filing {accession: row.accession})
            MERGE (fa:Fact {fact_id: row.fact_id})
            SET fa.ticker = row.ticker,
                fa.metric = row.metric,
                fa.fiscal_year = row.fiscal_year,
                fa.value = row.value,
                fa.unit = row.unit,
                fa.filing_date = row.filing_date,
                fa.accession = row.accession
            MERGE (f)-[:REPORTS]->(fa)
            """,
            facts=facts,
        )

        # 3. Connect existing (:Chunk) nodes to their (:Filing) by accession
        driver.execute_query(
            """
            MATCH (f:Filing), (c:Chunk)
            WHERE c.accession = f.accession
            MERGE (f)-[:HAS_CHUNK]->(c)
            """
        )

        # 4. RiskCategory nodes and (:Chunk)-[:MENTIONS_RISK]->(:RiskCategory)
        driver.execute_query(
            """
            UNWIND $categories AS row
            MERGE (rc:RiskCategory {name: row.name})
            SET rc.description = row.description
            """,
            categories=risk_categories,
        )
        if force_relabel:
            driver.execute_query("MATCH (:Chunk)-[r:MENTIONS_RISK]->(:RiskCategory) DELETE r")
        driver.execute_query(
            """
            UNWIND $edges AS row
            MATCH (c:Chunk {chunk_id: row.chunk_id})
            MATCH (rc:RiskCategory {name: row.category})
            MERGE (c)-[:MENTIONS_RISK]->(rc)
            """,
            edges=risk_edges,
        )

        # 5. Drop legacy embedded FactChunk nodes + local_facts_index if present
        if drop_legacy_fact_chunks:
            driver.execute_query("MATCH (n:FactChunk) DETACH DELETE n")
            driver.execute_query("DROP INDEX local_facts_index IF EXISTS")

        summary_rows, _, _ = driver.execute_query(
            """
            RETURN
                count { (:Company) } AS companies,
                count { (:Filing) } AS filings,
                count { (:Fact) } AS facts,
                count { (:Chunk) } AS chunks,
                count { (:RiskCategory) } AS risk_categories,
                count { (:Company)-[:FILED]->(:Filing) } AS filed_edges,
                count { (:Filing)-[:REPORTS]->(:Fact) } AS reports_edges,
                count { (:Filing)-[:HAS_CHUNK]->(:Chunk) } AS has_chunk_edges,
                count { (:Chunk)-[:MENTIONS_RISK]->(:RiskCategory) } AS mentions_risk_edges,
                count { (c:Chunk) WHERE NOT (:Filing)-[:HAS_CHUNK]->(c) } AS unlinked_chunks
            """
        )
        counts = dict(summary_rows[0])

    print(
        f"Graph connected: {counts['companies']} Company, {counts['filings']} Filing, "
        f"{counts['facts']} Fact, {counts['chunks']} Chunk, {counts['risk_categories']} RiskCategory nodes | "
        f"edges: {counts['filed_edges']} FILED, {counts['reports_edges']} REPORTS, "
        f"{counts['has_chunk_edges']} HAS_CHUNK, {counts['mentions_risk_edges']} MENTIONS_RISK "
        f"(unlinked chunks: {counts['unlinked_chunks']})"
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Company, Filing, Fact, and RiskCategory nodes and link Chunks.")
    parser.add_argument(
        "--keep-legacy",
        action="store_true",
        help="do not delete the old FactChunk nodes and local_facts_index",
    )
    parser.add_argument(
        "--force-relabel",
        action="store_true",
        help="re-run LLM risk-category labeling even if risk_labels.jsonl exists",
    )
    args = parser.parse_args()
    load_graph(drop_legacy_fact_chunks=not args.keep_legacy, force_relabel=args.force_relabel)


if __name__ == "__main__":
    main()
