from collections import Counter
from src.embed import get_vectorstore

vs = get_vectorstore()
all_data = vs.get()
counts = Counter()
for meta in all_data["metadatas"]:
    counts[(meta["ticker"], meta["fiscal_year"], meta["item"])] += 1

for key in sorted(counts):
    ticker, year, item = key
    print(f"{ticker} FY{year} Item {item:<3}: {counts[key]} chunks")