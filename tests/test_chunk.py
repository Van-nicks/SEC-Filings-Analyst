"""Unit tests for 10-K text chunking (src/filing_analyst/ingest/chunk.py)."""
import json
import shutil

import pytest

import filing_analyst.ingest.chunk as chunk_mod
from filing_analyst.config import COMPANIES, MIN_CHUNK_CHARS, RAW_DIR


@pytest.fixture(scope="module")
def docs():
    return chunk_mod.load_and_split()


def test_chunk_ids_unique_and_contiguous(docs):
    ids = [d.metadata["chunk_id"] for d in docs]
    assert len(ids) == len(set(ids)), "chunk_ids must be globally unique"

    per_file: dict[tuple[str, str], list[int]] = {}
    for d in docs:
        md = d.metadata
        idx = int(md["chunk_id"].rsplit("-", 1)[1])
        per_file.setdefault((md["ticker"], md["section"]), []).append(idx)

    for key, numbers in per_file.items():
        assert numbers == list(range(len(numbers))), f"Non-contiguous numbering for {key}"


def test_every_ticker_has_both_sections(docs):
    present = {(d.metadata["ticker"], d.metadata["section"]) for d in docs}
    expected = {(ticker, section) for ticker in COMPANIES for section in ("business", "risk_factors")}
    assert present == expected


def test_minimum_length_and_exact_start_index(docs):
    for d in docs:
        assert len(d.page_content) >= MIN_CHUNK_CHARS
        md = d.metadata
        raw_text = (RAW_DIR / md["source"]).read_text(encoding="utf-8")
        start = md["start_index"]
        assert raw_text[start : start + len(d.page_content)] == d.page_content


def test_identical_ids_on_rerun(docs):
    second = chunk_mod.load_and_split()
    assert [d.metadata["chunk_id"] for d in docs] == [d.metadata["chunk_id"] for d in second]


def test_stray_files_ignored_and_missing_meta_raises(tmp_path, monkeypatch):
    raw_copy = tmp_path / "raw"
    shutil.copytree(RAW_DIR, raw_copy)
    (raw_copy / "stray_notes.txt").write_text("should be ignored", encoding="utf-8")
    monkeypatch.setattr(chunk_mod, "RAW_DIR", raw_copy)

    baseline = chunk_mod.load_and_split()
    assert len(baseline) == 819

    (raw_copy / "AAPL_meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="AAPL_meta.json"):
        chunk_mod.load_and_split()


def test_save_chunks_jsonl(docs, tmp_path):
    out = chunk_mod.save_chunks_jsonl(docs, path=tmp_path / "chunks.jsonl")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(docs)
    first = json.loads(lines[0])
    assert {"chunk_id", "ticker", "section", "accession", "filing_date", "fiscal_year", "source", "start_index", "text"} <= first.keys()
