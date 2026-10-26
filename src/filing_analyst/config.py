"""Central configuration: loads .env once and holds every constant used by the pipeline.

Import this module before creating any AWS / Neo4j client so the environment
variables (Bedrock token, region, Neo4j credentials) are already loaded.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/filing_analyst/config.py -> project root
load_dotenv(PROJECT_ROOT / ".env")

# --- Directories ---
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- Data ---
FACTS_CSV = PROCESSED_DIR / "facts.csv"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
RISK_LABELS_JSONL = PROCESSED_DIR / "risk_labels.jsonl"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 150   # chunks shorter than this are merged into the next chunk

# --- Amazon Bedrock Models ---
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
CHAT_MODEL_ID = os.getenv("BEDROCK_CHAT_MODEL_ID", "eu.anthropic.claude-sonnet-5-5")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
INDEX_NAME = "chunk_embeddings"   # vector index over filing-text chunks
NODE_LABEL = "Chunk"              # label of the embedded text-chunk nodes

# --- Companies (ticker -> name); used for Company nodes ---
COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "TSLA": "Tesla, Inc.",
}

# --- Risk Taxonomy (name -> description); used for RiskCategory nodes ---
RISK_CATEGORIES = {
    "supply_chain": "Supplier dependency, single-source components, manufacturing/foundry capacity, logistics, geographic concentration (e.g. Taiwan/Asia).",
    "export_controls_and_geopolitics": "Export controls, trade restrictions, tariffs, sanctions, China or geopolitical tensions.",
    "cybersecurity_and_data": "Cyberattacks, ransomware, data breaches, system/cloud outages, network security.",
    "regulation_and_legal": "Government regulation, antitrust, compliance, litigation, data privacy laws, regulatory investigations.",
    "competition_and_demand": "Market competition, pricing pressure, product adoption, customer concentration, rapid technological/AI change.",
    "macroeconomic_and_financial": "Inflation, interest rates, currency/FX fluctuations, credit risk, liquidity, tax rates and liabilities.",
    "intellectual_property": "Patents, IP protection or infringement claims, licensing, trade secrets.",
    "workforce_and_governance": "Attracting/retaining key personnel, reliance on executives, corporate governance, labor relations.",
    "product_quality_and_liability": "Product defects, recalls, safety issues, warranty reserves, product liability claims.",
    "climate_and_disasters": "Natural disasters, earthquakes, climate change, environmental rules, energy/power disruptions.",
}

# --- Retrieval ---
TOP_K = 3
