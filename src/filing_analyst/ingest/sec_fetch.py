"""Fetch the latest 10-K text sections and key financial facts from SEC EDGAR."""
import csv
import json
import logging
import os

from dotenv import load_dotenv
from edgar import Company, set_identity

from filing_analyst.config import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
SECTIONS = {"Item 1": "business", "Item 1A": "risk_factors"}
METRICS = {
    "revenue": "get_revenue",
    "net_income": "get_net_income",
    "total_assets": "get_total_assets",
}

log = logging.getLogger("sec_fetch")


def fetch_latest_tenk(ticker: str):
    """Return (filing_metadata, tenk_object) for the company's latest 10-K."""
    filing = Company(ticker).get_filings(form="10-K", amendments=False).latest()
    if filing is None:
        raise ValueError(f"No 10-K found for {ticker}")
    tenk = filing.obj()
    period = str(tenk.period_of_report)
    meta = {
        "ticker": ticker,
        "accession": str(filing.accession_number),
        "filing_date": str(tenk.filing_date),
        "period_of_report": period,
        "fiscal_year": int(period[:4]),
        "form": str(filing.form)
    }
    return meta, tenk


def save_sections(tenk, meta: dict) -> None:
    """Write each wanted 10-K section to data/raw/{ticker}_{name}.txt."""
    for item, name in SECTIONS.items():
        text = tenk.get(item)  # non-raising: returns None if the section is missing
        if not text:
            log.warning("%s: section %s not found", meta["ticker"], item)
            continue
        path = RAW_DIR / f"{meta['ticker']}_{name}.txt"
        path.write_text(str(text), encoding="utf-8")
        log.info("%s: saved %s (%d chars)", meta["ticker"], path.name, len(str(text)))


def extract_facts(tenk, meta: dict) -> list[dict]:
    """Current and prior-year revenue, net income and total assets."""
    financials = tenk.financials
    if financials is None:
        log.warning("%s: no financials available", meta["ticker"])
        return []
    rows = []
    for metric, method_name in METRICS.items():
        for offset in (0, 1):  # 0 = current fiscal year, 1 = prior year
            value = getattr(financials, method_name)(period_offset=offset)
            if value is None:
                log.warning("%s: %s (offset %d) is missing", meta["ticker"], metric, offset)
                continue
            rows.append({
                "ticker": meta["ticker"],
                "metric": metric,
                "fiscal_year": meta["fiscal_year"] - offset,
                "value": value,
                "filing_date": meta["filing_date"],
                "accession": meta["accession"],
            })
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(PROJECT_ROOT / ".env")
    identity = os.getenv("EDGAR_IDENTITY")
    if not identity:
        raise SystemExit("Set EDGAR_IDENTITY in .env, e.g. 'Your Name your@email.com'")
    set_identity(identity)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_facts = []
    for ticker in TICKERS:
        try:
            meta, tenk = fetch_latest_tenk(ticker)
            save_sections(tenk, meta)
            (RAW_DIR / f"{ticker}_meta.json").write_text(json.dumps(meta, indent=2))
            all_facts.extend(extract_facts(tenk, meta))
        except Exception:
            log.exception("%s: failed, continuing with the next ticker", ticker)

    if all_facts:
        out = PROCESSED_DIR / "facts.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_facts[0].keys()))
            writer.writeheader()
            writer.writerows(all_facts)
        log.info("Wrote %d fact rows to %s", len(all_facts), out)


if __name__ == "__main__":
    main()
