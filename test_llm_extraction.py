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
import re
from datetime import datetime, timezone

from openai import OpenAI


def extract_numbers(text):
    """Extract all integer values from text, stripping $, commas, and decimals."""
    # Find patterns like $1,234,567 or 1234567 or 1,234,567.00
    raw = re.findall(r'[\$]?[\d,]+(?:\.\d+)?', text)
    nums = set()
    for r in raw:
        cleaned = r.replace('$', '').replace(',', '')
        # Drop decimal portion (e.g. "1234567.00" -> "1234567")
        if '.' in cleaned:
            cleaned = cleaned.split('.')[0]
        if cleaned.isdigit() and len(cleaned) >= 4:  # skip short numbers (years, etc. handled by comparison)
            nums.add(int(cleaned))
    return nums


EXTRACTION_PROMPT = """You are a municipal budget analyst. Extract the total {expense} EXPENDITURE amount for {city}, {state} for fiscal year {year}.

Rules:
- Return EXPENDITURES, not revenue
- Return the BUDGETED amount or the APPROPRIATED amount, not actual values
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
    parser.add_argument("--limit", type=int,
                        help="Number of gold records to test (default: use test_budgets.json)")
    parser.add_argument("--n-chunks", type=int, default=20,
                        help="Max chunks to use per query")
    parser.add_argument("--cache", default="gold_chunks_cache.json",
                        help="Chunks cache file (default: gold_chunks_cache.json)")
    parser.add_argument("--city", nargs=3, action="append", metavar=("CITY", "STATE", "YEAR"),
                        help="Filter to specific city/state/year (repeatable)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print chunks sent to LLM")
    args = parser.parse_args()

    llm = OpenAI(base_url=args.llm_url, api_key="not-needed")

    with open(args.cache) as f:
        cache = json.load(f)

    if args.city:
        city_filters = {(c.lower().replace(" ", "_"), s.lower(), int(y)) for c, s, y in args.city}
    elif args.limit:
        rows = cache[:args.limit]
        city_filters = None
    else:
        # Default: load test set from test_budgets.json
        with open("test_budgets.json") as f:
            test_set = json.load(f)
        city_filters = {(t["city"].lower(), t["state"].lower(), t["year"]) for t in test_set}

    if city_filters:
        rows = [r for r in cache if (r["city"].lower().replace(" ", "_"), r["state"].lower(), r["year"]) in city_filters]
    elif not args.limit:
        rows = cache
    print(f"Testing {len(rows)}/{len(cache)} gold records")
    print(f"LLM: {args.llm_url} model={args.model} chunks={args.n_chunks}\n")

    # Table header
    print(f"{'State':<6} {'City':<20} {'Year':<6} {'Expense':<15} {'LLM Answer':<45} {'Expected':<20} {'In?':<5} {'Match'}")
    print("-" * 125)

    exact = 0
    total = 0
    in_chunks_count = 0
    results = []

    for row in rows:
        state, city, year = row["state"], row["city"], row["year"]
        expense, expected = row["expense"], row["budget"]

        chunks = row["chunks"][:args.n_chunks]

        if not chunks:
            print(f"{state.upper():<6} {city:<20} {year:<6} {expense:<15} {'NO CHUNKS':<45} ${expected:>14,.0f}      --    --")
            results.append({"state": state, "city": city, "year": year, "expense": expense,
                            "expected": expected, "answer": None, "match": False,
                            "in_chunks": False, "no_chunks": True})
            total += 1
            continue

        # Check if expected value appears in chunks
        expected_int = int(expected)
        # Build search patterns: "110,304,890" and "110304890"
        expected_formatted = f"{expected_int:,}"
        expected_plain = str(expected_int)
        all_chunk_text = " ".join(c["text"] for c in chunks)
        in_chunks = expected_formatted in all_chunk_text or expected_plain in all_chunk_text
        if in_chunks:
            in_chunks_count += 1
        in_flag = "Y" if in_chunks else "N"

        chunk_text = "\n\n".join(
            f"[Chunk {i+1} | {c['metadata'].get('filename', '')} | parser: {c['metadata'].get('parser', '')}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )

        prompt = EXTRACTION_PROMPT.format(
            expense=expense, city=city, state=state, year=year, chunks=chunk_text
        )

        if args.verbose:
            print(f"\n  --- CHUNKS SENT TO LLM ({city}, {state.upper()} {year} {expense}) ---")
            for i, c in enumerate(chunks):
                print(f"  [Chunk {i+1} | {c['metadata'].get('filename', '')}]")
                preview = c['text'][:300].replace('\n', '\n  ')
                print(f"  {preview}")
                if len(c['text']) > 300:
                    print(f"  ... ({len(c['text'])} chars total)")
                print()

        response = llm.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        answer = response.choices[0].message.content.strip()
        expected_int = int(expected)
        expected_str = f"${expected:,.0f}"

        # Extract all numbers from LLM response and check if expected value is among them
        answer_nums = extract_numbers(answer)
        match = expected_int in answer_nums
        if match:
            exact += 1
        total += 1
        flag = "OK" if match else "MISS"

        results.append({"state": state, "city": city, "year": year, "expense": expense,
                        "expected": expected, "answer": answer, "match": match,
                        "in_chunks": in_chunks, "no_chunks": False})

        # Truncate long answers for display
        answer_display = answer[:40] + "..." if len(answer) > 40 else answer
        print(f"{state.upper():<6} {city:<20} {year:<6} {expense:<15} {answer_display:<45} {expected_str:<20} {in_flag:<5} {flag}")

    print("-" * 125)
    if total:
        print(f"Match: {exact}/{total} ({100*exact/total:.0f}%)  |  Answer in chunks: {in_chunks_count}/{total}")
    else:
        print("No records tested.")

    # Save structured results
    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "n_chunks": args.n_chunks,
        "cache_file": args.cache,
        "total": total,
        "match": exact,
        "in_chunks": in_chunks_count,
        "results": results,
    }
    os.makedirs("runs", exist_ok=True)
    # Slug: model name (last path component) + chunks + timestamp
    model_slug = args.model.rstrip("/").split("/")[-1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"runs/{model_slug}_c{args.n_chunks}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(run, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
