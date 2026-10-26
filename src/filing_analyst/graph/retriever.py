"""Query path: connect to the already-built Neo4j graph/index and retrieve evidence.

No embedding of the corpus and no writes happen here. Safe to call from an agent or API:

    from filing_analyst.graph.retriever import retrieve, search_chunks, query_facts, query_companies_by_risk

    docs = retrieve("What was Apple's revenue in 2024?")
    chunks = search_chunks("China export controls", ticker="NVDA", risk_category="export_controls_and_geopolitics")
    facts = query_facts(ticker="AAPL", metric="revenue")
"""
from functools import lru_cache

from langchain_neo4j import Neo4jVector
from neo4j import GraphDatabase

from filing_analyst.config import (
    INDEX_NAME,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    TOP_K,
)
from filing_analyst.llm.models import get_embeddings


@lru_cache(maxsize=1)
def get_driver():
    """Return a shared Neo4j driver for read-only Cypher queries."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


@lru_cache(maxsize=1)
def get_vectorstore():
    """Connect to the existing index once per process (no inserts)."""
    try:
        return Neo4jVector.from_existing_index(
            embedding=get_embeddings(),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=INDEX_NAME,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Vector index '{INDEX_NAME}' not found. Run: python -m filing_analyst.graph.build_index"
        ) from exc


@lru_cache(maxsize=1)
def get_retriever():
    """Return the shared similarity retriever (top-k from config)."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    return retriever


def retrieve(query: str):
    """Return the top-k matching LangChain Documents for a user query."""
    return get_retriever().invoke(query)


def search_chunks(
    query: str,
    ticker: str | None = None,
    section: str | None = None,
    risk_category: str | None = None,
    k: int = TOP_K,
) -> list[dict]:
    """Embed `query` with Titan and run vector similarity search, narrowed by optional graph filters."""
    query_vector = get_embeddings().embed_query(query)
    candidate_k = max(k * 10, 40) if (ticker or section or risk_category) else k
    records, _, _ = get_driver().execute_query(
        """
        CALL db.index.vector.queryNodes($index_name, $candidate_k, $embedding)
        YIELD node AS c, score
        MATCH (co:Company)-[:FILED]->(f:Filing)-[:HAS_CHUNK]->(c)
        WHERE ($ticker IS NULL OR co.ticker = $ticker)
          AND ($section IS NULL OR c.section = $section)
          AND ($risk_category IS NULL OR EXISTS {
              MATCH (c)-[:MENTIONS_RISK]->(:RiskCategory {name: $risk_category})
          })
        OPTIONAL MATCH (c)-[:MENTIONS_RISK]->(rc:RiskCategory)
        RETURN c.chunk_id AS chunk_id,
               co.ticker AS ticker,
               co.name AS company,
               c.section AS section,
               f.accession AS accession,
               f.filing_date AS filing_date,
               f.fiscal_year AS fiscal_year,
               c.source AS source,
               c.start_index AS start_index,
               c.text AS text,
               collect(rc.name) AS risk_categories,
               score
        ORDER BY score DESC
        LIMIT $k
        """,
        index_name=INDEX_NAME,
        candidate_k=candidate_k,
        embedding=query_vector,
        ticker=ticker.upper() if ticker else None,
        section=section,
        risk_category=risk_category,
        k=k,
    )
    return [dict(r) for r in records]


def query_facts(
    ticker: str | None = None,
    metric: str | None = None,
    fiscal_year: int | None = None,
) -> list[dict]:
    """Look up structured numeric Fact nodes via (:Company)-[:FILED]->(:Filing)-[:REPORTS]->(:Fact)."""
    records, _, _ = get_driver().execute_query(
        """
        MATCH (co:Company)-[:FILED]->(f:Filing)-[:REPORTS]->(fa:Fact)
        WHERE ($ticker IS NULL OR co.ticker = $ticker)
          AND ($metric IS NULL OR fa.metric = $metric)
          AND ($fiscal_year IS NULL OR fa.fiscal_year = $fiscal_year)
        RETURN fa.fact_id AS fact_id,
               co.ticker AS ticker,
               co.name AS company,
               fa.metric AS metric,
               fa.fiscal_year AS fiscal_year,
               fa.value AS value,
               fa.unit AS unit,
               f.accession AS accession,
               f.filing_date AS filing_date
        ORDER BY co.ticker, fa.metric, fa.fiscal_year
        """,
        ticker=ticker.upper() if ticker else None,
        metric=metric,
        fiscal_year=fiscal_year,
    )
    return [dict(r) for r in records]


def query_companies_by_risk(risk_category: str) -> list[dict]:
    """Return every company that cites `risk_category`, with chunk count and sample chunk IDs."""
    records, _, _ = get_driver().execute_query(
        """
        MATCH (co:Company)-[:FILED]->(f:Filing)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS_RISK]->(rc:RiskCategory {name: $risk_category})
        WITH co, f, count(c) AS chunk_count, collect(c.chunk_id)[0..3] AS sample_chunk_ids
        RETURN co.ticker AS ticker,
               co.name AS company,
               f.accession AS accession,
               chunk_count,
               sample_chunk_ids
        ORDER BY chunk_count DESC, co.ticker
        """,
        risk_category=risk_category,
    )
    return [dict(r) for r in records]


if __name__ == "__main__":
    query = "What are the core facts in the document?"
    matched_docs = retrieve(query)

    print("\n--- Similarity Search Results ---")
    for i, doc in enumerate(matched_docs):
        print(f"\nMatch #{i+1}:")
        print(doc.page_content[:300])
