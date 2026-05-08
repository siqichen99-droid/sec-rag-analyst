"""Pull 10-K filings from SEC EDGAR for a fixed set of companies and years."""
import os
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Config ---------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

USER_AGENT = os.getenv("SEC_USER_AGENT")
if not USER_AGENT:
    raise RuntimeError("Set SEC_USER_AGENT in your .env (e.g. 'Jane Doe jane@example.com')")

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

COMPANIES = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
}
TARGET_FISCAL_YEARS = {2023, 2024, 2025}

# --- Helpers --------------------------------------------------------------
def _get(url: str) -> requests.Response:
    """SEC asks for <=10 req/sec. We're way under, but be polite."""
    time.sleep(0.15)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r

def list_10k_filings(cik: str) -> list[dict]:
    """Return all 10-K filings for a CIK, newest first."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = _get(url).json()
    recent = data["filings"]["recent"]
    out = []
    for form, acc, primary_doc, filing_date, report_date in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["primaryDocument"],
        recent["filingDate"],
        recent["reportDate"],
    ):
        if form == "10-K":
            out.append({
                "accession": acc,
                "primary_doc": primary_doc,
                "filing_date": filing_date,
                "report_date": report_date,
            })
    return out

def download_filing(ticker: str, cik: str, filing: dict) -> Path:
    """Download the primary 10-K HTML doc to disk."""
    acc_nodash = filing["accession"].replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{filing['primary_doc']}"
    fy = filing["report_date"][:4]
    out_path = DATA_DIR / f"{ticker}_{fy}_10K.html"

    if out_path.exists():
        print(f"  cached: {out_path.name}")
        return out_path

    print(f"  fetching {ticker} FY{fy} ({filing['report_date']})")
    r = _get(url)
    out_path.write_bytes(r.content)
    return out_path

# --- Main ----------------------------------------------------------------
def main():
    manifest = []
    for ticker, cik in COMPANIES.items():
        print(f"\n{ticker} (CIK {cik})")
        filings = list_10k_filings(cik)
        keepers = [f for f in filings if int(f["report_date"][:4]) in TARGET_FISCAL_YEARS]

        for f in keepers:
            path = download_filing(ticker, cik, f)
            manifest.append({
                "ticker": ticker,
                "fiscal_year": int(f["report_date"][:4]),
                "report_date": f["report_date"],
                "filing_date": f["filing_date"],
                "accession": f["accession"],
                "path": str(path.relative_to(DATA_DIR.parent)),
            })

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved manifest with {len(manifest)} filings -> {manifest_path}")

if __name__ == "__main__":
    main()