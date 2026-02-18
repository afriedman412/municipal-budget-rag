"""Diagnose extraction misses: ask the LLM what the wrong values represent.

Takes a run JSON + chunks cache, finds misses where the model returned a wrong
number, and asks the LLM "what does $X represent in these chunks?"

Usage:
  python diagnose_misses.py                                              # Most recent run in runs/
  python diagnose_misses.py runs/some_run.json                           # Specific local file
  python diagnose_misses.py --llm-url http://localhost:8000/v1 --model budget-mistral-lora-merged
  python diagnose_misses.py --limit 10                                   # Only diagnose 10 misses
"""

import argparse
import json
import os
import re
import sys
from openai import OpenAI


DIAGNOSIS_PROMPT = """You are a municipal budget analyst. A colleague extracted "${wrong_answer}" as the total {expense} EXPENDITURE for {city}, {state} for fiscal year {year}, but the correct answer is {expected}.

Look at these budget document excerpts and explain what "${wrong_answer}" actually represents. Is it:
- Revenue instead of expenditure?
- Proposed/recommended instead of adopted?
- A different fiscal year?
- A sub-department or line item instead of the total?
- A different fund (e.g. total budget instead of General Fund)?
- Something else?

Be specific and concise (1-2 sentences). If you can find "${wrong_answer}" in the excerpts, quote the surrounding context.

Budget document excerpts:
---
{chunks}
---

What does "${wrong_answer}" represent?"""


def extract_numbers(text):
    """Extract all integer values from text."""
    raw = re.findall(r'[\$]?[\d,]+(?:\.\d+)?', text)
    nums = set()
    for r in raw:
        cleaned = r.replace('$', '').replace(',', '')
        if '.' in cleaned:
            cleaned = cleaned.split('.')[0]
        if cleaned.isdigit() and len(cleaned) >= 4:
            nums.add(int(cleaned))
    return nums


def find_latest_run():
    """Find the most recent run JSON in the local runs/ directory."""
    runs_dir = "runs"
    if not os.path.isdir(runs_dir):
        print("No runs/ directory found.")
        sys.exit(1)
    files = sorted(f for f in os.listdir(runs_dir) if f.endswith(".json") and f != "index.json")
    if not files:
        print("No run files found in runs/.")
        sys.exit(1)
    latest = files[-1]
    path = os.path.join(runs_dir, latest)
    print(f"Using most recent run: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_file", nargs="?", default=None,
                        help="Path to run JSON file (default: fetch latest from GCS)")
    parser.add_argument("--cache", default="gold_chunks_cache.json",
                        help="Chunks cache file")
    parser.add_argument("--llm-url", default="http://localhost:11434/v1",
                        help="LLM API base URL")
    parser.add_argument("--model", default="llama3.1:8b",
                        help="Model name")
    parser.add_argument("--limit", type=int,
                        help="Max number of misses to diagnose")
    parser.add_argument("--n-chunks", type=int, default=5,
                        help="Max chunks to include in diagnosis prompt")
    args = parser.parse_args()

    if args.run_file:
        with open(args.run_file) as f:
            run = json.load(f)
        print(f"Loaded {args.run_file}")
    else:
        run_path = find_latest_run()
        with open(run_path) as f:
            run = json.load(f)

    with open(args.cache) as f:
        cache = json.load(f)

    # Index cache by (city, state, year, expense)
    cache_map = {}
    for r in cache:
        key = (r["city"].lower(), r["state"].lower(), r["year"], r["expense"].lower())
        cache_map[key] = r

    llm = OpenAI(base_url=args.llm_url, api_key="not-needed")

    # Find misses with a numeric wrong answer
    misses = []
    for r in run["results"]:
        if r["match"] or r.get("no_chunks") or not r.get("answer"):
            continue
        answer_nums = extract_numbers(r["answer"])
        expected = int(r["expected"])
        if not answer_nums or expected in answer_nums:
            continue
        # Pick the main wrong number (closest to expected)
        wrong = min(answer_nums, key=lambda x: abs(x - expected))
        misses.append((r, wrong))

    if args.limit:
        misses = misses[:args.limit]

    print(f"Diagnosing {len(misses)} misses from {args.run_file}\n")

    # Run diagnosis for each miss
    diagnoses = []
    for i, (r, wrong_int) in enumerate(misses):
        state, city, year = r["state"], r["city"], r["year"]
        expense, expected = r["expense"], int(r["expected"])

        key = (city.lower(), state.lower(), year, expense.lower())
        cached = cache_map.get(key)
        if not cached:
            diagnoses.append((r, wrong_int, "NO CACHED CHUNKS"))
            continue

        chunks = cached["chunks"][:args.n_chunks]
        chunk_text = "\n\n".join(
            f"[Chunk {i+1}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )

        wrong_str = f"${wrong_int:,}"
        expected_str = f"${expected:,}"

        prompt = DIAGNOSIS_PROMPT.format(
            wrong_answer=wrong_str,
            expense=expense,
            city=city,
            state=state,
            year=year,
            expected=expected_str,
            chunks=chunk_text,
        )

        try:
            response = llm.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            diagnosis = response.choices[0].message.content.strip()
        except Exception as e:
            diagnosis = f"ERROR: {e}"

        diagnoses.append((r, wrong_int, diagnosis))
        print(f"  [{i+1}/{len(misses)}] {state.upper()} {city} {year} {expense}", flush=True)

    # Print summary table
    print(f"\n{'State':<6} {'City':<20} {'Year':<6} {'Expense':<15} {'Wrong':>15} {'Expected':>15}  Diagnosis")
    print("-" * 130)
    for r, wrong_int, diagnosis in diagnoses:
        state, city, year = r["state"], r["city"], r["year"]
        expense, expected = r["expense"], int(r["expected"])
        diag_short = diagnosis.replace('\n', ' ')[:80]
        if len(diagnosis) > 80:
            diag_short += "..."
        print(f"{state.upper():<6} {city:<20} {year:<6} {expense:<15} ${wrong_int:>14,} ${expected:>14,}  {diag_short}")

    # Print full diagnoses
    print(f"\n{'='*130}")
    print("FULL DIAGNOSES:\n")
    for r, wrong_int, diagnosis in diagnoses:
        state, city, year = r["state"], r["city"], r["year"]
        expense, expected = r["expense"], int(r["expected"])
        print(f"--- {state.upper()} {city} {year} {expense} ---")
        print(f"  Wrong: ${wrong_int:,}  Expected: ${expected:,}")
        print(f"  {diagnosis}\n")


if __name__ == "__main__":
    main()
