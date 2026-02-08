"""Validate retrieval against ground truth."""

import asyncio
import sqlite3

from dotenv import load_dotenv
load_dotenv()

from pipeline.config import Config
from pipeline.chroma import ChromaClient
from pipeline.embed import EmbeddingClient


def format_val(v: float) -> str:
    """Format budget value with commas."""
    if v >= 1_000_000:
        return f"${v:,.0f}"
    return f"${v:,.2f}"


def find_value_in_chunks(docs: list[str], value: float) -> bool:
    """Check if a budget value appears in any chunk text."""
    int_val = int(value)
    formats = [
        str(int_val),                          # 375620069
        f"{int_val:,}",                        # 375,620,069
        f"{value:,.0f}",                       # 375,620,069
        f"{value:,.2f}",                       # 375,620,069.00
    ]
    if int_val >= 1000:
        formats.append(f"{int_val // 1000:,}")  # 375,620 (thousands)

    for doc in docs:
        for fmt in formats:
            if fmt in doc:
                return True
    return False


async def main():
    config = Config.from_env()
    chroma = ChromaClient(config)
    embedder = EmbeddingClient(config)

    conn = sqlite3.connect("pipeline_state.db")
    conn.row_factory = sqlite3.Row

    # Get every ground truth record
    rows = conn.execute(
        "SELECT state, city, year, expense, budget FROM validation ORDER BY state, city, year, expense"
    ).fetchall()

    print(f"\n{'='*80}")
    print(f"RETRIEVAL VALIDATION: {len(rows)} ground truth records")
    print(f"{'='*80}")

    # Pre-compute query embeddings (only 3 unique queries needed)
    query_map = {}
    for expense_type in ["General Fund", "Police", "Education"]:
        query = f"{expense_type} total expenditure budget"
        emb = await embedder.embed_texts([query])
        query_map[expense_type] = emb[0]

    total = 0
    found_semantic = 0
    found_keyword = 0
    found_any = 0
    results = []

    for row in rows:
        state = row["state"]
        city = row["city"]
        year = row["year"]
        expense = row["expense"]
        expected = row["budget"]
        total += 1

        # ChromaDB stores lowercase city with underscores
        chroma_city = city.lower().replace(" ", "_")
        query_embedding = query_map[expense]

        # 1) Semantic only
        sem_results = chroma.query_budget(
            query_embedding=query_embedding,
            city=chroma_city,
            year=year,
            n_results=20,
            state=state,
        )
        sem_docs = sem_results["documents"][0]
        sem_found = find_value_in_chunks(sem_docs, expected)
        if sem_found:
            found_semantic += 1

        # 2) With keyword filtering
        kw = expense.upper()
        if kw == "GENERAL FUND":
            keywords = ["GENERAL FUND"]
        else:
            keywords = [kw]

        kw_results = chroma.query_budget(
            query_embedding=query_embedding,
            city=chroma_city,
            year=year,
            n_results=20,
            keywords=keywords,
            state=state,
        )
        kw_docs = kw_results["documents"][0]
        kw_found = find_value_in_chunks(kw_docs, expected)
        if kw_found:
            found_keyword += 1

        any_found = sem_found or kw_found
        if any_found:
            found_any += 1

        status = "OK" if any_found else "MISS"
        sem_tag = "sem" if sem_found else "   "
        kw_tag = "kw" if kw_found else "  "
        results.append(
            f"  {status}  {sem_tag} {kw_tag}  {state} {city:15s} {year} {expense:15s} {format_val(expected)}"
        )

    for r in results:
        print(r)

    print(f"\n{'='*80}")
    print(f"TOTALS: {total} records")
    print(f"  Semantic only:    {found_semantic}/{total} ({100*found_semantic/total:.0f}%)")
    print(f"  Keyword filter:   {found_keyword}/{total} ({100*found_keyword/total:.0f}%)")
    print(f"  Either (best):    {found_any}/{total} ({100*found_any/total:.0f}%)")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
