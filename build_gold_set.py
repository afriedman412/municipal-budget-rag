"""One-time script: discover ingested cities in ChromaDB, match against
validation table, and save as a gold standard validation set.
Also caches the 3 query embeddings so LLM testing needs zero OpenAI calls."""

import asyncio
import json
import os
import sqlite3

from dotenv import load_dotenv
load_dotenv()

from paths import TRAINING_DIR
from pipeline.config import Config
from pipeline.chroma import ChromaClient
from pipeline.embed import EmbeddingClient


async def main():
    config = Config.from_env()
    embedder = EmbeddingClient(config)
    conn = sqlite3.connect("pipeline_state.db")
    conn.row_factory = sqlite3.Row

    # Get all distinct (state, city, year) from both parser collections
    collections = {"aryn": "budgets-aryn", "pymupdf": "budgets-pymupdf"}
    ingested = set()
    print("Querying ChromaDB for ingested documents...")
    for name, coll_name in collections.items():
        cfg = Config.from_env()
        cfg.chroma_collection = coll_name
        try:
            c = ChromaClient(cfg)
            sample = c.collection.get(
                where={"city": {"$ne": ""}},
                limit=200000,
                include=["metadatas"],
            )
            for m in sample["metadatas"]:
                city = m.get("city", "")
                state = m.get("state", "")
                year = m.get("year", 0)
                if city and state:
                    ingested.add((state, city, year))
            print(f"  {name}: {len(sample['metadatas'])} chunks")
        except Exception as e:
            print(f"  {name}: skipped ({e})")

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

    with open(TRAINING_DIR / "gold_validation_set.json", "w") as f:
        json.dump(gold, f, indent=2)

    print(f"Saved to {TRAINING_DIR}/gold_validation_set.json")

    # Cache query embeddings
    print("\nGenerating query embeddings...")
    query_embeddings = {}
    for expense_type in ["General Fund", "Police", "Education"]:
        query = f"{expense_type} total expenditure budget"
        emb = await embedder.embed_texts([query])
        query_embeddings[expense_type] = emb[0]

    with open(TRAINING_DIR / "gold_query_embeddings.json", "w") as f:
        json.dump(query_embeddings, f)

    print(f"Saved 3 query embeddings to {TRAINING_DIR}/gold_query_embeddings.json")

    # Cache retrieved chunks for each gold record, per parser collection
    collections = {
        "aryn": "budgets-aryn",
        "pymupdf": "budgets-pymupdf",
    }

    for parser_name, collection_name in collections.items():
        print(f"\nRetrieving chunks from {collection_name}...")
        config_copy = Config.from_env()
        config_copy.chroma_collection = collection_name
        try:
            parser_chroma = ChromaClient(config_copy)
        except Exception as e:
            print(f"  Skipping {collection_name}: {e}")
            continue

        chunks_cache = []
        for i, g in enumerate(gold):
            state, city, year = g["state"], g["city"], g["year"]
            expense = g["expense"]
            chroma_city = city.lower().replace(" ", "_")
            query_embedding = query_embeddings[expense]

            try:
                results = parser_chroma.query_budget(
                    query_embedding=query_embedding,
                    city=chroma_city,
                    year=year,
                    n_results=40,
                    state=state,
                )
                docs = results["documents"][0]
                metas = results["metadatas"][0]
            except Exception:
                docs, metas = [], []

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

        outfile = TRAINING_DIR / f"gold_chunks_{parser_name}.json"
        with open(outfile, "w") as f:
            json.dump(chunks_cache, f)

        cache_size = os.path.getsize(outfile) / 1024 / 1024
        print(f"Saved {len(chunks_cache)} records to {outfile} ({cache_size:.1f} MB)")

    # Preview
    cities = sorted(set((g["state"], g["city"], g["year"]) for g in gold))
    for state, city, year in cities:
        n = sum(1 for g in gold if g["state"] == state and g["city"] == city and g["year"] == year)
        print(f"  {state:3s} {city:20s} {year}  ({n} records)")


if __name__ == "__main__":
    asyncio.run(main())
