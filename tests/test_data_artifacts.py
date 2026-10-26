"""Unit tests for committed data and evaluation artifacts (runs in CI without raw SEC files or AWS keys)."""
import csv
import json

from filing_analyst.config import COMPANIES, FACTS_CSV, PROJECT_ROOT, RISK_CATEGORIES, RISK_LABELS_JSONL


def test_facts_csv_schema_and_completeness():
    assert FACTS_CSV.exists(), "data/processed/facts.csv must be committed"
    with FACTS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30, "Expected 30 structured facts (5 companies x 3 metrics x 2 years)"
    tickers = {r["ticker"] for r in rows}
    metrics = {r["metric"] for r in rows}
    assert tickers == set(COMPANIES.keys())
    assert metrics == {"revenue", "net_income", "total_assets"}
    for r in rows:
        assert float(r["value"]) > 0
        assert int(r["fiscal_year"]) in {2024, 2025, 2026}


def test_risk_labels_jsonl_schema_and_taxonomy():
    assert RISK_LABELS_JSONL.exists(), "data/processed/risk_labels.jsonl must be committed"
    with RISK_LABELS_JSONL.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 585, "Expected 585 risk_factors chunk label records"
    total_edges = 0
    for r in rows:
        assert "chunk_id" in r and "-risk_factors-" in r["chunk_id"]
        cats = r["categories"]
        assert isinstance(cats, list) and len(cats) <= len(RISK_CATEGORIES)
        for c in cats:
            assert c in RISK_CATEGORIES, f"Unknown risk category: {c}"
        total_edges += len(cats)
    assert total_edges == 971


def test_eval_dataset_schema():
    eval_path = PROJECT_ROOT / "eval" / "eval_dataset.json"
    assert eval_path.exists()
    items = json.loads(eval_path.read_text(encoding="utf-8"))
    assert len(items) == 30
    ids = [item["id"] for item in items]
    assert ids == [f"Q{i:02d}" for i in range(1, 31)]
    for item in items:
        assert {"id", "category", "question", "ground_truth", "expected_sources", "expected_tickers", "expected_numbers"} <= item.keys()
