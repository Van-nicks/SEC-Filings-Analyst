"""Central configuration: loads .env once and holds every constant used by the pipeline.

Import this module before creating any AWS / Neo4j client so the environment
variables (Bedrock token, region, Neo4j credentials) are already loaded.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/filing_analyst/config.py -> project root
load_dotenv(PROJECT_ROOT / ".env")

# --- Data ---
FACTS_CSV = PROJECT_ROOT / "data" / "processed" / "facts.csv"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Embeddings (Amazon Bedrock) ---
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
INDEX_NAME = "local_facts_index"
NODE_LABEL = "FactChunk"

# --- Retrieval ---
TOP_K = 3
