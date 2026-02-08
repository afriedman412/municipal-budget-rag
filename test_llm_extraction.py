"""Test LLM extraction: load gold set, retrieve chunks from ChromaDB, send to Ollama.
Requires: python build_gold_set.py (run once first)
No OpenAI calls needed — uses cached embeddings."""

import json

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from pipeline.config import Config
from pipeline.chroma import ChromaClient


EXTRACTION_PROMPT = """You are a budget analyst. Given the following excerpts from a municipal budget document, extract the total {expense} expenditure/budget amount for {city}, {state}.

Return ONLY the numeric dollar amount (e.g. "$1,234,567"). If you cannot find the value, return "NOT FOUND".

Budget document excerpts:
---
{chunks}
---

Total {expense} expenditure amount:"""


def main():
    config = Config.from_env()
    chroma = ChromaClient(config)

    # Ollama OpenAI-compatible client
    llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    # Load gold set and cached embeddings
    with open("gold_validation_set.json") as f:
        gold = json.load(f)
    with open("gold_query_embeddings.json") as f:
        query_embeddings = json.load(f)

    print(f"Loaded {len(gold)} gold records, testing 1\n")
    rows = gold[:1]

    for row in rows:
        state, city, year = row["state"], row["city"], row["year"]
        expense, expected = row["expense"], row["budget"]

        print(f"\n{'='*70}")
        print(f"  {city}, {state.upper()} ({year}) — {expense}")
        print(f"  Expected: ${expected:,.0f}")
        print(f"{'='*70}")

        chroma_city = city.lower().replace(" ", "_")
        query_embedding = query_embeddings[expense]

        # Retrieve chunks
        results = chroma.query_budget(
            query_embedding=query_embedding,
            city=chroma_city,
            year=year,
            n_results=5,
            state=state,
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            print("  No chunks found in ChromaDB for this city/year.")
            continue

        print(f"  Retrieved {len(docs)} chunks")
        print(f"  First chunk preview: {docs[0][:150]}...")

        # Build context from chunks
        chunk_text = "\n\n".join(
            f"[Chunk {i+1} | {m.get('filename','')} | parser: {m.get('parser','')}]\n{doc}"
            for i, (doc, m) in enumerate(zip(docs, metas))
        )

        # Ask Ollama
        prompt = EXTRACTION_PROMPT.format(
            expense=expense, city=city, state=state, chunks=chunk_text
        )

        print("  Querying Ollama...")
        response = llm.chat.completions.create(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        answer = response.choices[0].message.content.strip()
        print(f"  LLM answer:  {answer}")
        print(f"  Expected:    ${expected:,.0f}")


if __name__ == "__main__":
    main()
