"""Parse 10-K HTML into the standard SEC Item structure."""
import re
from pathlib import Path
from bs4 import BeautifulSoup
import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Items that matter for analyst use cases. Skipping things like Item 4 (Mine
# Safety) and Items 10-14 (usually incorporated by reference from proxy).
TARGET_ITEMS = {
    "1":  "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2":  "Properties",
    "3":  "Legal Proceedings",
    "5":  "Market for Registrant's Common Equity",
    "7":  "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8":  "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
}

# Matches an "Item" header at the start of a line.
# Lookahead ensures "Item 1" doesn't match the prefix of "Item 1A".
ITEM_PATTERN = re.compile(
    r'(?:^|\n)\s*Item\s+(\d{1,2}[A-Z]?)(?=[\s\.\-—–])',
    re.IGNORECASE | re.MULTILINE,
)


def html_to_text(html_path: Path) -> str:
    """Extract clean text from a 10-K HTML file, preserving paragraph breaks."""
    soup = BeautifulSoup(html_path.read_bytes(), "lxml")

    # Drop noise that confuses both regex and embeddings
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()

    # separator='\n' inserts newlines between block-level elements,
    # which preserves the visual structure we need for section detection.
    text = soup.get_text(separator="\n")

    # Normalize whitespace: collapse multiple spaces, but keep paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def find_item_boundaries(text: str) -> dict[str, tuple[int, int]]:
    """Locate the start/end char offsets of each Item section in the text.

    Strategy: a 10-K typically mentions "Item 1A" at least twice — once in the
    Table of Contents, once as the actual section header. We take the LAST
    occurrence of each Item ID, which (empirically, for big-tech filings) is
    almost always the body section start. Then we sort and use the next Item's
    position as the end boundary.
    """
    matches = list(ITEM_PATTERN.finditer(text))

    by_item: dict[str, list[int]] = {}
    for m in matches:
        item_id = m.group(1).upper()
        by_item.setdefault(item_id, []).append(m.start())

    # Last occurrence per Item ID
    item_starts = {item_id: positions[-1] for item_id, positions in by_item.items()}

    # Sort by position so we can compute end boundaries
    sorted_items = sorted(item_starts.items(), key=lambda kv: kv[1])

    boundaries: dict[str, tuple[int, int]] = {}
    for i, (item_id, start) in enumerate(sorted_items):
        end = sorted_items[i + 1][1] if i + 1 < len(sorted_items) else len(text)
        boundaries[item_id] = (start, end)
    return boundaries


def parse_filing(html_path: Path) -> dict[str, str]:
    """Return {item_id: section_text} for the items we care about."""
    text = html_to_text(html_path)
    boundaries = find_item_boundaries(text)

    sections: dict[str, str] = {}
    for item_id in TARGET_ITEMS:
        if item_id not in boundaries:
            continue
        start, end = boundaries[item_id]
        body = text[start:end].strip()
        # Skip tiny sections — almost always means we caught a ToC entry
        # rather than a real body section.
        if len(body) < 500:
            continue
        sections[item_id] = body
    return sections


# --- Inspection mode -----------------------------------------------------
def main():
    """Run parse.py directly to sanity-check extraction across all filings."""
    import json
    data_dir = Path(__file__).resolve().parent.parent / "data"
    manifest = json.loads((data_dir / "manifest.json").read_text())

    print(f"{'Filing':<18} | {'Items found':<55} | Total chars")
    print("-" * 100)
    for f in manifest:
        path = Path(__file__).resolve().parent.parent / f["path"]
        sections = parse_filing(path)
        items_str = ", ".join(sorted(sections.keys(), key=lambda x: (int(re.match(r'\d+', x).group()), x)))
        total = sum(len(s) for s in sections.values())
        label = f"{f['ticker']} FY{f['fiscal_year']}"
        print(f"{label:<18} | {items_str:<55} | {total:,}")


if __name__ == "__main__":
    main()