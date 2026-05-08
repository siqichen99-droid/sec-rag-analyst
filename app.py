"""Streamlit UI for the SEC 10-K RAG analyst."""
import streamlit as st

from src.rag import ask, stratified_ask
from src.embed import CHROMA_DIR

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SEC 10-K Analyst",
    page_icon="📊",
    layout="wide",
)

# Custom CSS — minimal, targeted polish
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .stButton button {border-radius: 6px; font-weight: 500;}
    h1 {color: #1f4e79; font-weight: 700;}
    h2, h3 {color: #2c3e50;}
    [data-testid="stMetricValue"] {color: #1f4e79; font-weight: 600;}
    [data-testid="stMetricLabel"] {color: #6b7280;}
    .streamlit-expanderHeader {font-weight: 500;}
    section[data-testid="stSidebar"] {border-right: 1px solid #e5e7eb;}
    hr {margin: 2rem 0;}
</style>
""", unsafe_allow_html=True)

# Hero
st.title("📊 SEC 10-K Analyst")
st.markdown(
    "<p style='color:#6b7280; font-size:1.05rem; margin-top:-0.5rem;'>"
    "Conversational Q&A over Apple, Microsoft, and Google 10-K filings · FY2023–FY2025"
    "</p>",
    unsafe_allow_html=True,
)

# Corpus stats — instant credibility signal
col1, col2, col3, col4 = st.columns(4)
col1.metric("Companies", "3", help="Apple (AAPL), Microsoft (MSFT), Alphabet (GOOGL)")
col2.metric("Fiscal years", "3", help="FY2023, FY2024, FY2025")
col3.metric("10-K filings", "9")
col4.metric("Chunks indexed", "685")

st.divider()

if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
    st.error("No index found. Run `python -m src.embed` first, then refresh this page.")
    st.stop()


# ---------------------------------------------------------------------------
# Example presets
# ---------------------------------------------------------------------------
EXAMPLES = [
    {
        "label": "AI risks across all 3 companies",
        "query": "What are the main risks related to AI mentioned across these companies?",
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "years": [],
        "items": ["1A"],
        "strategy": "Stratified (per-group)",
        "stratify_field": "ticker",
    },
    {
        "label": "Apple risk factors over time",
        "query": "How did Apple's risk factors evolve from FY2023 to FY2025? Focus on what's new.",
        "tickers": ["AAPL"],
        "years": [2023, 2024, 2025],
        "items": ["1A"],
        "strategy": "Stratified (per-group)",
        "stratify_field": "fiscal_year",
    },
    {
        "label": "Cybersecurity comparison (FY2024)",
        "query": "Compare cybersecurity disclosures of MSFT, AAPL, and GOOGL in FY2024.",
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "years": [2024],
        "items": ["1C"],
        "strategy": "Stratified (per-group)",
        "stratify_field": "ticker",
    },
    {
        "label": "Apple business segments (FY2025)",
        "query": "What are Apple's main business segments and product categories?",
        "tickers": ["AAPL"],
        "years": [2025],
        "items": ["1"],
        "strategy": "Standard MMR",
        "stratify_field": "fiscal_year",
    },
]

ITEM_LABELS = {
    "1":  "1 — Business",
    "1A": "1A — Risk Factors",
    "1C": "1C — Cybersecurity",
    "7":  "7 — MD&A",
    "7A": "7A — Market Risk",
    "8":  "8 — Financials",
}


# ---------------------------------------------------------------------------
# Session state init + preset application
# ---------------------------------------------------------------------------
SS_DEFAULTS = {
    "query": "",
    "tickers": [],
    "years": [],
    "items": [],
    "strategy": "Standard MMR",
    "stratify_field": "fiscal_year",
    "api_key": "",
}
for k, v in SS_DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(ex: dict):
    for field in ("query", "tickers", "years", "items", "strategy", "stratify_field"):
        st.session_state[field] = ex[field]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔑 OpenAI API Key")
    st.text_input(
        "Paste your key (sk-proj-...)",
        type="password",
        key="api_key",
        label_visibility="collapsed",
        help=(
            "Your key is sent to OpenAI's API for embeddings and chat, but isn't "
            "stored or logged by this app. Get one at "
            "https://platform.openai.com/api-keys"
        ),
    )
    if not st.session_state["api_key"].strip():
        st.warning("Add your OpenAI key to enable the demo.")
    else:
        st.success("Key loaded.")

    st.divider()

    st.markdown("### 🔎 Filters")
    st.multiselect("Companies", options=["AAPL", "MSFT", "GOOGL"], key="tickers")
    st.multiselect("Fiscal years", options=[2023, 2024, 2025], key="years")
    st.multiselect(
        "10-K Items",
        options=list(ITEM_LABELS.keys()),
        format_func=lambda x: ITEM_LABELS[x],
        key="items",
    )

    st.divider()

    st.markdown("### ⚙️ Retrieval")
    st.radio(
        "Strategy",
        options=["Standard MMR", "Stratified (per-group)"],
        key="strategy",
        help=(
            "Standard: one MMR call across the filter set. "
            "Stratified: separate retrieval per group — better for "
            "comparison/evolution queries."
        ),
    )

    if st.session_state["strategy"].startswith("Stratified"):
        st.selectbox("Stratify by", options=["fiscal_year", "ticker", "item"], key="stratify_field")
        k_param = st.slider("Chunks per group", 1, 6, 3)
    else:
        k_param = st.slider("Chunks to retrieve (k)", 4, 16, 8)


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------
def build_filters() -> dict | None:
    clauses = []
    if st.session_state["tickers"]:
        clauses.append({"ticker": {"$in": st.session_state["tickers"]}})
    if st.session_state["years"]:
        clauses.append({"fiscal_year": {"$in": st.session_state["years"]}})
    if st.session_state["items"]:
        clauses.append({"item": {"$in": st.session_state["items"]}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def get_stratify_values(field: str) -> list:
    if field == "fiscal_year":
        return st.session_state["years"] or [2023, 2024, 2025]
    if field == "ticker":
        return st.session_state["tickers"] or ["AAPL", "MSFT", "GOOGL"]
    if field == "item":
        return st.session_state["items"] or list(ITEM_LABELS.keys())
    return []


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.markdown("### 💡 Example queries")
st.caption("Each preset configures filters + retrieval strategy for the best answer.")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(
        ex["label"],
        use_container_width=True,
        on_click=apply_preset,
        args=(ex,),
    )

st.markdown("### ❓ Ask a question")
st.text_area(
    "Your question",
    height=80,
    placeholder="e.g. What did Microsoft say about generative AI investments?",
    key="query",
    label_visibility="collapsed",
)

ask_disabled = not (
    st.session_state["query"].strip() and st.session_state["api_key"].strip()
)

if st.button("🚀 Ask", type="primary", disabled=ask_disabled):
    with st.spinner("Retrieving relevant filings and reasoning..."):
        try:
            filters = build_filters()
            if st.session_state["strategy"].startswith("Stratified"):
                answer, sources = stratified_ask(
                    question=st.session_state["query"],
                    stratify_by=st.session_state["stratify_field"],
                    values=get_stratify_values(st.session_state["stratify_field"]),
                    base_filters=filters,
                    k_per_group=k_param,
                    api_key=st.session_state["api_key"],
                )
            else:
                answer, sources = ask(
                    question=st.session_state["query"],
                    k=k_param,
                    filters=filters,
                    api_key=st.session_state["api_key"],
                )
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.divider()
    st.markdown("### 📝 Answer")
    st.markdown(answer)

    st.markdown(f"### 📚 Sources ({len(sources)} chunks retrieved)")
    for i, s in enumerate(sources, 1):
        m = s.metadata
        label = f"{i}. **{m['ticker']}** · FY{m['fiscal_year']} · Item {m['item']} ({m['item_title']})"
        with st.expander(label):
            st.text(s.page_content)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#9ca3af; padding:1rem 0; font-size:0.85rem;">
      Built with LangChain · Chroma · OpenAI &nbsp;·&nbsp;
      Filings from <a href="https://www.sec.gov/edgar" target="_blank" style="color:#1f4e79;">SEC EDGAR</a>
    </div>
    """,
    unsafe_allow_html=True,
)