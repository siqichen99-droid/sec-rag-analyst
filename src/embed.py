"""Embed chunked 10-K sections into a persistent Chroma vector store."""
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.chunk import chunk_all_filings

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "sec_10k"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_vectorstore(api_key: str | None = None) -> Chroma:
    """Return a Chroma store backed by a persistent on-disk directory.

    api_key: optional. If provided, used directly. If None, falls back to
    OPENAI_API_KEY env var. Lets the same code work in CLI mode (env var
    from .env) and in the Streamlit demo (visitor pastes their own key).
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key required. Either set OPENAI_API_KEY in .env or "
            "pass api_key explicitly."
        )
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def main():
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f"Existing index found at {CHROMA_DIR}")
        choice = input("Rebuild from scratch? [y/N]: ").strip().lower()
        if choice == "y":
            shutil.rmtree(CHROMA_DIR)
            print("Deleted.\n")
        else:
            print("Keeping existing index. Exiting.")
            return

    print("Chunking filings...")
    docs = chunk_all_filings()

    total_chars = sum(len(d.page_content) for d in docs)
    est_tokens = total_chars / 4
    est_cost = est_tokens / 1_000_000 * 0.02
    print(f"\nEmbedding {len(docs)} chunks (~{est_tokens:,.0f} tokens, est ${est_cost:.4f})\n")

    vs = get_vectorstore()  # uses .env for CLI

    batch_size = 100
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding"):
        vs.add_documents(docs[i:i + batch_size])

    print(f"\nDone. Persisted to {CHROMA_DIR}\n")

    print("--- Sanity check: 'What are the main risks related to AI?' ---\n")
    results = vs.similarity_search("What are the main risks related to AI?", k=3)
    for i, r in enumerate(results, 1):
        m = r.metadata
        print(f"{i}. {m['ticker']} FY{m['fiscal_year']} — Item {m['item']} ({m['item_title']})")
        print(f"   {r.page_content[:250].strip()}...\n")


if __name__ == "__main__":
    main()