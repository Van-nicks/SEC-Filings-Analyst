"""Load the raw 10-K text sections, attach filing metadata, split into chunks, clean them, and give each a stable ID.

Free and local: no API calls. Output chunk metadata:
ticker, section, accession, filing_date, fiscal_year, source, start_index, chunk_id

Cleaning after the split (see _clean_and_merge):
  * drop page numbers / "Form 10-K | N" footers / bare "Item 1A" titles
  * merge chunks shorter than MIN_CHUNK_CHARS into the following chunk (a risk heading stays with its text)
Merged chunks are re-cut from the original file text, so source[start_index : start_index + len(text)] is always exact.
"""
import json
import re
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from filing_analyst.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_JSONL, MIN_CHUNK_CHARS, RAW_DIR

_FOOTER = re.compile(r"Form 10-K\s*\|\s*\d+\s*$")                   # "Apple Inc. | 2025 Form 10-K | 4"
_SECTION_TITLE = re.compile(r"\s*item\s+1a?\b", re.IGNORECASE)      # "Item 1A.  Risk Factors", "ITEM 1. BUSINESS"
_LEADING_PAGE_NUM = re.compile(r"^\s*\d{1,3}\s*\n\s*\n+")           # "31\n\n\nDelaware law ..." (page number glued on)


def load_meta() -> dict:
    """Return {ticker: meta dict} from every data/raw/*_meta.json."""
    meta = {}
    for p in RAW_DIR.glob("*_meta.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        meta[m["ticker"]] = m
    return meta


def _is_noise(text: str) -> bool:
    """True for chunks that carry no content: page numbers, 10-K footers, bare section titles."""
    t = text.strip()
    if not re.search(r"[A-Za-z]", t):
        return True
    if len(t) < 80 and _FOOTER.search(t):
        return True
    if len(t) < 100 and _SECTION_TITLE.match(t):
        return True
    return False


def _clean_and_merge(chunks: list[Document], file_text: str) -> list[Document]:
    """Drop noise chunks and merge short ones forward, working on character spans of the source file."""
    spans = []
    for c in sorted(chunks, key=lambda c: c.metadata["start_index"]):
        start = c.metadata["start_index"]
        end = start + len(c.page_content)
        lead = _LEADING_PAGE_NUM.match(c.page_content)
        if lead:                                   # strip the glued-on page number, keep the offset exact
            start += lead.end()
        if _is_noise(file_text[start:end]):
            continue
        spans.append([start, end])

    merged = []
    i = 0
    while i < len(spans):
        start, end = spans[i]
        while end - start < MIN_CHUNK_CHARS and i + 1 < len(spans):   # short -> absorb the next chunk
            i += 1
            end = max(end, spans[i][1])
        merged.append([start, end])
        i += 1
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < MIN_CHUNK_CHARS:   # short tail -> absorb into previous
        tail = merged.pop()
        merged[-1][1] = max(merged[-1][1], tail[1])

    base = chunks[0].metadata
    return [
        Document(
            page_content=file_text[start:end],
            metadata={**{k: v for k, v in base.items() if k != "start_index"}, "start_index": start},
        )
        for start, end in merged
    ]


def load_and_split():
    """Return cleaned chunked Documents built from data/raw/*.txt with filing metadata and chunk_id."""
    txt_loader = DirectoryLoader(
        str(RAW_DIR),                 # the folder (e.g. PROJECT_ROOT / "data" / "raw")
        glob=("*_business.txt", "*_risk_factors.txt"),   # only the section files written by sec_fetch
        loader_cls=TextLoader,        # loader used for EACH matched file
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,)

    txt_docs = sorted(txt_loader.load(), key=lambda d: Path(d.metadata["source"]).name)
    meta = load_meta()

    # 1. enrich each file-level Document BEFORE splitting
    for doc in txt_docs:
        file_name = Path(doc.metadata["source"]).name          # AAPL_risk_factors.txt
        ticker, section = Path(file_name).stem.split("_", 1)   # AAPL, risk_factors
        if ticker not in meta:
            raise FileNotFoundError(f"{file_name} has no {ticker}_meta.json in {RAW_DIR}")
        m = meta[ticker]
        doc.metadata = {
            "ticker": ticker,
            "section": section,
            "accession": m["accession"],
            "filing_date": m["filing_date"],
            "fiscal_year": m["fiscal_year"],
            "source": file_name,          # file name only, not the absolute path
        }

    # 2. split (metadata is copied onto every chunk; add_start_index stores each chunk's character offset)
    rc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    docs = []
    for file_doc in txt_docs:
        chunks = rc_splitter.split_documents([file_doc])
        # 3. drop noise, merge tiny chunks (re-cut from the original text so start_index stays exact)
        docs.extend(_clean_and_merge(chunks, file_doc.page_content))

    # 4. stable chunk_id: <ticker>-<accession>-<section>-<NNN>, numbered per file AFTER cleaning
    counters = {}
    for d in docs:
        md = d.metadata
        key = (md["ticker"], md["section"])
        n = counters.get(key, 0)
        counters[key] = n + 1
        md["chunk_id"] = f"{md['ticker']}-{md['accession']}-{md['section']}-{n:03d}"
    return docs


def save_chunks_jsonl(docs: list[Document] | None = None, path: Path = CHUNKS_JSONL) -> Path:
    """Write chunks to data/processed/chunks.jsonl (one JSON object per line)."""
    if docs is None:
        docs = load_and_split()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            record = {**d.metadata, "text": d.page_content}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    chunks = load_and_split()
    out = save_chunks_jsonl(chunks)
    print(f"Wrote {len(chunks)} chunks to {out}")
