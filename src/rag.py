"""RAG chain for querying SEC 10-K filings."""
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.embed import get_vectorstore

load_dotenv()

LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a senior financial analyst answering questions about SEC 10-K filings.

You will be given context from 10-K filings — chunks of text, each tagged with the company \
ticker, fiscal year, and the SEC Item the chunk came from. Use ONLY the information in the \
provided context to answer.

Guidelines:
- Cite every factual claim with its source in the format [TICKER FYxxxx, Item X]. Multiple \
sources can be stacked: [AAPL FY2024, Item 1A][MSFT FY2024, Item 1A].
- If the answer isn't in the context, say so explicitly. Never fabricate or guess.
- For comparison questions across companies, organize the answer to make differences crisp.
- For evolution questions (year-over-year change), organize chronologically.
- Quote financial figures precisely; do not round.
- Use plain English; briefly define jargon the first time you use it.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\n---\n\nQuestion: {question}"),
])


def format_docs(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        m = d.metadata
        header = f"[{m['ticker']} FY{m['fiscal_year']}, Item {m['item']} ({m['item_title']})]"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _resolve_key(api_key: str | None) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI API key required. Set OPENAI_API_KEY in .env or pass api_key."
        )
    return key


def _generate_answer(question: str, docs: list[Document], api_key: str) -> str:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=api_key)
    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "context": format_docs(docs),
        "question": question,
    })


def ask(
    question: str,
    k: int = 8,
    filters: dict | None = None,
    api_key: str | None = None,
) -> tuple[str, list[Document]]:
    api_key = _resolve_key(api_key)
    vs = get_vectorstore(api_key=api_key)
    docs = vs.max_marginal_relevance_search(
        question, k=k, fetch_k=k * 3, filter=filters,
    )
    answer = _generate_answer(question, docs, api_key=api_key)
    return answer, docs


def stratified_ask(
    question: str,
    stratify_by: str,
    values: list,
    base_filters: dict | None = None,
    k_per_group: int = 3,
    api_key: str | None = None,
) -> tuple[str, list[Document]]:
    api_key = _resolve_key(api_key)
    vs = get_vectorstore(api_key=api_key)

    docs: list[Document] = []
    for value in values:
        group_filter = {stratify_by: value}
        if base_filters:
            existing = base_filters["$and"] if "$and" in base_filters else [base_filters]
            combined = {"$and": existing + [group_filter]}
        else:
            combined = group_filter
        group_docs = vs.max_marginal_relevance_search(
            question, k=k_per_group, fetch_k=k_per_group * 3, filter=combined,
        )
        docs.extend(group_docs)

    answer = _generate_answer(question, docs, api_key=api_key)
    return answer, docs


def _print_result(question: str, answer: str, sources: list[Document]):
    print("\n" + "=" * 80)
    print(f"Q: {question}")
    print("=" * 80)
    print(f"\n{answer}\n")
    print("--- Sources retrieved ---")
    for s in sources:
        m = s.metadata
        print(f"  - {m['ticker']} FY{m['fiscal_year']} Item {m['item']:<3} (chunk {m['chunk_index']})")


def main():
    q = "What are the main risks related to AI mentioned across these companies?"
    answer, sources = ask(q, k=8)
    _print_result(q, answer, sources)

    q = "How did Apple's risk factors evolve from FY2023 to FY2025? Focus on what's new."
    answer, sources = stratified_ask(
        question=q,
        stratify_by="fiscal_year",
        values=[2023, 2024, 2025],
        base_filters={"$and": [{"ticker": "AAPL"}, {"item": "1A"}]},
        k_per_group=3,
    )
    _print_result(q, answer, sources)

    q = "Compare the cybersecurity disclosures of Microsoft, Apple, and Google in FY2024."
    answer, sources = stratified_ask(
        question=q,
        stratify_by="ticker",
        values=["AAPL", "MSFT", "GOOGL"],
        base_filters={"$and": [{"item": "1C"}, {"fiscal_year": 2024}]},
        k_per_group=2,
    )
    _print_result(q, answer, sources)

    q = "What are Apple's main business segments and product categories per its FY2025 10-K?"
    answer, sources = ask(
        q, k=8,
        filters={"$and": [{"ticker": "AAPL"}, {"fiscal_year": 2025}, {"item": "1"}]},
    )
    _print_result(q, answer, sources)


if __name__ == "__main__":
    main()