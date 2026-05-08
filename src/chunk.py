"""Chunk parsed 10-K sections into LangChain Documents with metadata."""
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.parse import parse_filing, TARGET_ITEMS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Chunk sizing rationale:
# - text-embedding-3-small handles up to 8191 tokens but quality is best
#   around 500-1000 tokens per chunk for retrieval.
# - 4000 chars ≈ 1000 tokens for English prose.
# - 400 char overlap (~10%) preserves context across chunk boundaries
#   without inflating the index too much.
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 400


def chunk_filing(filing: dict) -> list[Document]:
    html_path = ROOT / filing["path"]
    sections = parse_filing(html_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Prefer paragraph breaks > line breaks > sentences > words.
        # This is what gives us clean semantic chunks vs naive char-counting.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: list[Document] = []
    for item_id, section_text in sections.items():
        for i, chunk in enumerate(splitter.split_text(section_text)):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "ticker": filing["ticker"],
                    "fiscal_year": filing["fiscal_year"],
                    "item": item_id,
                    "item_title": TARGET_ITEMS[item_id],
                    "source": Path(filing["path"]).name,
                    "chunk_index": i,
                },
            ))
    return docs


def chunk_all_filings() -> list[Document]:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    all_docs: list[Document] = []
    for f in manifest:
        docs = chunk_filing(f)
        items = sorted({d.metadata["item"] for d in docs})
        print(f"  {f['ticker']} FY{f['fiscal_year']}: {len(docs)} chunks across items {items}")
        all_docs.extend(docs)
    return all_docs


def main():
    print("Chunking all filings...\n")
    all_docs = chunk_all_filings()

    print(f"\nTotal: {len(all_docs)} chunks from {len({(d.metadata['ticker'], d.metadata['fiscal_year']) for d in all_docs})} filings")

    # Save a sample so we can eyeball the output before embedding
    sample = [
        {"metadata": d.metadata, "preview": d.page_content[:300]}
        for d in all_docs[::max(1, len(all_docs) // 8)][:8]
    ]
    sample_path = DATA_DIR / "chunks_sample.json"
    sample_path.write_text(json.dumps(sample, indent=2))
    print(f"Sample of 8 chunks saved to {sample_path.name}")


if __name__ == "__main__":
    main()