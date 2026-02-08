"""Test LLM extraction: load cached chunks, send to LLM, compare to ground truth.
Requires: python build_gold_set.py (run once first on machine with ChromaDB)
No OpenAI calls, no ChromaDB needed — uses cached chunks.

Usage:
  python test_llm_extraction.py                          # Ollama (default)
  python test_llm_extraction.py --llm-url http://localhost:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct  # vLLM
  python test_llm_extraction.py --limit 10               # Test 10 records
  python test_llm_extraction.py --n-chunks 5             # Use only top 5 chunks
"""

import argparse
import json
import os

from openai import OpenAI


EXTRACTION_PROMPT = """You are a budget analyst. Given the following excerpts from a municipal budget document, extract the total {expense} expenditure/budget amount for {city}, {state}.

Return ONLY the numeric dollar amount (e.g. "$1,234,567"). If you cannot find the value, return "NOT FOUND".

Budget document excerpts:
---
{chunks}
---

Total {expense} expenditure amount:"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-url", default="http://localhost:11434/v1",
                        help="LLM API base URL (default: Ollama)")
    parser.add_argument("--model", default="llama3.1:8b",
                        help="Model name (default: llama3.1:8b for Ollama)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of gold records to test")
    parser.add_argument("--n-chunks", type=int, default=20,
                        help="Max chunks to use per query")
    args = parser.parse_args()

    llm = OpenAI(base_url=args.llm_url, api_key="not-needed")

    with open("gold_chunks_cache.json") as f:
        cache = json.load(f)

    rows = cache[:args.limit]
    print(f"Testing {len(rows)}/{len(cache)} gold records")
    print(f"LLM: {args.llm_url} model={args.model} chunks={args.n_chunks}\n")

    for row in rows:
        state, city, year = row["state"], row["city"], row["year"]
        expense, expected = row["expense"], row["budget"]

        print(f"\n{'='*70}")
        print(f"  {city}, {state.upper()} ({year}) — {expense}")
        print(f"  Expected: ${expected:,.0f}")
        print(f"{'='*70}")

        chunks = row["chunks"][:args.n_chunks]

        if not chunks:
            print("  No cached chunks for this record.")
            continue

        print(f"  Using {len(chunks)} chunks")

        chunk_text = "\n\n".join(
            f"[Chunk {i+1} | {c['metadata'].get('filename','')} | parser: {c['metadata'].get('parser','')}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )

        prompt = EXTRACTION_PROMPT.format(
            expense=expense, city=city, state=state, chunks=chunk_text
        )

        print(f"  Querying LLM...")
        response = llm.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        answer = response.choices[0].message.content.strip()
        print(f"  LLM answer:  {answer}")
        print(f"  Expected:    ${expected:,.0f}")


if __name__ == "__main__":
    main()
