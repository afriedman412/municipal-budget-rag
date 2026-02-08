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


EXTRACTION_PROMPT = """You are a municipal budget analyst. Extract the total {expense} EXPENDITURE amount for {city}, {state} for fiscal year {year}.

Rules:
- Return EXPENDITURES, not revenue or appropriations
- Return the BUDGETED amount, not actual/unaudited figures
- Prefer ADOPTED budget over proposed or mayor's recommended
- Return ONLY the numeric dollar amount (e.g. "$1,234,567")
- If you cannot find the value, return "NOT FOUND"

Budget document excerpts:
---
{chunks}
---

Total {expense} budgeted expenditure for FY {year}:"""


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
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print chunks sent to LLM")
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
            expense=expense, city=city, state=state, year=year, chunks=chunk_text
        )

        if args.verbose:
            print(f"\n  --- CHUNKS SENT TO LLM ---")
            for i, c in enumerate(chunks):
                print(f"  [Chunk {i+1} | {c['metadata'].get('filename','')}]")
                # Print first 300 chars of each chunk
                preview = c['text'][:300].replace('\n', '\n  ')
                print(f"  {preview}")
                if len(c['text']) > 300:
                    print(f"  ... ({len(c['text'])} chars total)")
                print()

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
