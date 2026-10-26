"""One-time build of the Neo4j vector index from data/processed/facts.csv.

Run once:
    python -m filing_analyst.graph.build_index           # builds only if needed
    python -m filing_analyst.graph.build_index --force   # wipe FactChunk nodes + index, rebuild

Embedding with Bedrock costs money, so the build is skipped when the index
already exists and holds the expected number of chunks.
"""
import argparse

from langchain_neo4j import Neo4jVector
from neo4j import GraphDatabase

from filing_analyst.config import (
    INDEX_NAME,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    NODE_LABEL,
)
from filing_analyst.ingest.chunk import load_and_split
from filing_analyst.llm.models import get_embeddings


def index_status() -> tuple[bool, int]:
    """Return (index_exists, number_of_FactChunk_nodes). Plain Neo4j queries; no Bedrock calls."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver:
        records, _, _ = driver.execute_query(
            "SHOW INDEXES YIELD name WHERE name = $name RETURN name", name=INDEX_NAME
        )
        exists = len(records) > 0
        records, _, _ = driver.execute_query(f"MATCH (n:`{NODE_LABEL}`) RETURN count(n) AS c")
        return exists, records[0]["c"]


def build_index(force: bool = False):
    """Embed the chunks and store them in Neo4j. Returns the vectorstore, or None if skipped."""
    docs = load_and_split()

    if not force:
        exists, count = index_status()
        if exists and count == len(docs):
            print(f"Index '{INDEX_NAME}' already built with {count} chunks. Skipping (no Bedrock calls).")
            return None
        if exists:
            print(
                f"Index '{INDEX_NAME}' exists but holds {count} chunks while facts.csv produces "
                f"{len(docs)}. Skipping to avoid unintended Bedrock cost; re-run with --force to rebuild."
            )
            return None

    embeddings = get_embeddings()

    print("Generating vectors via Bedrock and saving to local Docker Neo4j...")
    vectorstore = Neo4jVector.from_documents(
        documents=docs,
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=INDEX_NAME,
        node_label=NODE_LABEL,
        pre_delete_collection=force,  # False (default behaviour) unless --force
    )
    print("Success! Your Docker Neo4j instance is populated and indexed.")
    return vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Neo4j vector index (run once).")
    parser.add_argument("--force", action="store_true", help="wipe existing FactChunk nodes/index and rebuild")
    args = parser.parse_args()
    build_index(force=args.force)


if __name__ == "__main__":
    main()
