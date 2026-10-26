"""Query path: connect to the already-built Neo4j index and retrieve matching chunks.

No embedding of the corpus and no writes happen here. The only Bedrock call per
query is embedding the question text. Safe to call from an API handler:

    from filing_analyst.graph.retriever import retrieve
    docs = retrieve("What was Apple's revenue in 2024?")
"""
from functools import lru_cache

from langchain_neo4j import Neo4jVector

from filing_analyst.config import (
    INDEX_NAME,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    TOP_K,
)
from filing_analyst.llm.models import get_embeddings


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
    """Return the top-k matching documents for a user query."""
    return get_retriever().invoke(query)


if __name__ == "__main__":
    query = "What are the core facts in the document?"
    matched_docs = retrieve(query)

    print("\n--- Similarity Search Results ---")
    for i, doc in enumerate(matched_docs):
        print(f"\nMatch #{i+1}:")
        print(doc.page_content[:300])
