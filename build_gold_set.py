"""One-time script: discover ingested cities in ChromaDB, match against
validation table, and save as a gold standard validation set.
Also caches the 3 query embeddings so LLM testing needs zero OpenAI calls."""

import asyncio
import json
import os
import sqlite3

from dotenv import load_dotenv
load_dotenv()

from pipeline.config import Config
from pipeline.chroma import ChromaClient
from pipeline.embed import EmbeddingClient


async def main():
    config = Config.from_env()
    chroma = ChromaClient(config)
    embedder = EmbeddingClient(config)
    conn = sqlite3.connect("pipeline_state.db")
    conn.row_factory = sqlite3.Row

    # Get all distinct (state, city, year) from ChromaDB
    print("Querying ChromaDB for ingested documents...")
    sample = chroma.collection.get(
        where={"city": {"$ne": ""}},
        limit=200000,
        include=["metadatas"],
    )
    ingested = set()
    for m in sample["metadatas"]:
        city = m.get("city", "")
        state = m.get("state", "")
        year = m.get("year", 0)
        if city and state:
            ingested.add((state, city, year))

    print(f"Found {len(ingested)} distinct (state, city, year) combos in ChromaDB")

    # Match against validation table
    all_rows = conn.execute(
        "SELECT state, city, year, expense, budget_type, budget FROM validation"
    ).fetchall()

    gold = []
    for r in all_rows:
        key = (r["state"].lower(), r["city"].lower().replace(" ", "_"), r["year"])
        if key in ingested:
            gold.append({
                "state": r["state"],
                "city": r["city"],
                "year": r["year"],
                "expense": r["expense"],
                "budget_type": r["budget_type"],
                "budget": r["budget"],
            })

    print(f"Matched {len(gold)} validation records across {len(set((g['state'], g['city'], g['year']) for g in gold))} city/year combos")

    with open("gold_validation_set.json", "w") as f:
        json.dump(gold, f, indent=2)

    print("Saved to gold_validation_set.json")

    # Cache query embeddings
    print("\nGenerating query embeddings...")
    query_embeddings = {}
    for expense_type in ["General Fund", "Police", "Education"]:
        query = f"{expense_type} total expenditure budget"
        emb = await embedder.embed_texts([query])
        query_embeddings[expense_type] = emb[0]

    with open("gold_query_embeddings.json", "w") as f:
        json.dump(query_embeddings, f)

    print(f"Saved 3 query embeddings to gold_query_embeddings.json")

    # Cache retrieved chunks for each gold record
    print(f"\nRetrieving chunks for {len(gold)} gold records...")
    chunks_cache = []
    for i, g in enumerate(gold):
        state, city, year = g["state"], g["city"], g["year"]
        expense = g["expense"]
        chroma_city = city.lower().replace(" ", "_")
        query_embedding = query_embeddings[expense]

        results = chroma.query_budget(
            query_embedding=query_embedding,
            city=chroma_city,
            year=year,
            n_results=20,
            state=state,
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        chunks_cache.append({
            "state": state,
            "city": city,
            "year": year,
            "expense": expense,
            "budget_type": g.get("budget_type", ""),
            "budget": g["budget"],
            "chunks": [
                {"text": doc, "metadata": meta}
                for doc, meta in zip(docs, metas)
            ],
        })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(gold)} records cached...")

    with open("gold_chunks_cache.json", "w") as f:
        json.dump(chunks_cache, f)

    cache_size = os.path.getsize("gold_chunks_cache.json") / 1024 / 1024
    print(f"Saved {len(chunks_cache)} records to gold_chunks_cache.json ({cache_size:.1f} MB)")

    # Preview
    cities = sorted(set((g["state"], g["city"], g["year"]) for g in gold))
    for state, city, year in cities:
        n = sum(1 for g in gold if g["state"] == state and g["city"] == city and g["year"] == year)
        print(f"  {state:3s} {city:20s} {year}  ({n} records)")


if __name__ == "__main__":
    asyncio.run(main())
