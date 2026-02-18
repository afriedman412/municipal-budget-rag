"""Test LLM extraction: load cached chunks, send to LLM, compare to ground truth.
Requires: python build_gold_set.py (run once first on machine with ChromaDB)
No OpenAI calls, no ChromaDB needed — uses cached chunks.

Usage:
  python test_llm_extraction.py                          # Ollama (default)
  python test_llm_extraction.py --llm-url http://localhost:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct  # vLLM
  python test_llm_extraction.py --limit 10               # Test 10 records
  python test_llm_extraction.py --n-chunks 5             # Use only top 5 chunks
  python test_llm_extraction.py --workers 8              # Parallel requests to vLLM
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def find_value_in_chunks(chunks, expected_int):
    """Find which chunks contain the expected value. Returns (in_chunks, chunk_info)."""
    expected_formatted = f"{expected_int:,}"
    expected_plain = str(expected_int)
    found_indices = []
    found_in_table = False
    parser = None

    for i, c in enumerate(chunks):
        text = c["text"]
        if expected_formatted in text or expected_plain in text:
            found_indices.append(i)
            if c["metadata"].get("has_table"):
                found_in_table = True
            if not parser and c["metadata"].get("parser"):
                parser = c["metadata"]["parser"]

    in_chunks = len(found_indices) > 0
    # Count table vs text chunks in the set sent to LLM
    n_table_chunks = sum(1 for c in chunks if c["metadata"].get("has_table"))
    n_text_chunks = len(chunks) - n_table_chunks

    return in_chunks, {
        "found_in_indices": found_indices,
        "found_in_table": found_in_table,
        "parser": parser or chunks[0]["metadata"].get("parser", "unknown"),
        "n_table_chunks": n_table_chunks,
        "n_text_chunks": n_text_chunks,
    }


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


def call_llm(llm, model, prompt):
    """Make a single LLM call. Returns the answer string."""
    response = llm.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


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
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Parallel LLM requests (default: 1, try 8 for vLLM)")
    parser.add_argument("--gcs-bucket",
                        help="GCS bucket to upload results (e.g. muni-budget-runs)")
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
    print(f"LLM: {args.llm_url} model={args.model} chunks={args.n_chunks} workers={args.workers}\n")

    # Phase 1: Prepare all tasks
    tasks = []  # (index, row_data, prompt_or_None)
    for idx, row in enumerate(rows):
        state, city, year = row["state"], row["city"], row["year"]
        expense, expected = row["expense"], row["budget"]
        chunks = row["chunks"][:args.n_chunks]

        if not chunks:
            tasks.append((idx, {
                "state": state, "city": city, "year": year, "expense": expense,
                "expected": expected, "answer": None, "match": False,
                "in_chunks": False, "no_chunks": True,
            }, None))
            continue

        expected_int = int(expected)
        in_chunks, chunk_info = find_value_in_chunks(chunks, expected_int)

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

        tasks.append((idx, {
            "state": state, "city": city, "year": year, "expense": expense,
            "expected": expected, "in_chunks": in_chunks, "no_chunks": False,
            **chunk_info,
        }, prompt))

    # Phase 2: Run LLM calls (parallel or sequential)
    llm_tasks = [(idx, data, prompt) for idx, data, prompt in tasks if prompt is not None]
    answers = {}  # idx -> answer string

    if args.workers > 1 and llm_tasks:
        print(f"Sending {len(llm_tasks)} requests with {args.workers} workers...", end=" ", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(call_llm, llm, args.model, prompt): idx
                for idx, _, prompt in llm_tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    answers[idx] = future.result()
                except Exception as e:
                    answers[idx] = f"ERROR: {e}"
                done += 1
                if done % 20 == 0 or done == len(llm_tasks):
                    print(f"{done}/{len(llm_tasks)}", end=" ", flush=True)
        print()
    else:
        for idx, data, prompt in llm_tasks:
            answers[idx] = call_llm(llm, args.model, prompt)

    # Phase 3: Compile results and print table
    print(f"{'State':<6} {'City':<20} {'Year':<6} {'Expense':<15} {'LLM Answer':<45} {'Expected':<20} {'In?':<5} {'Match'}")
    print("-" * 125)

    exact = 0
    total = 0
    in_chunks_count = 0
    results = []

    for idx, data, prompt in tasks:
        state, city, year = data["state"], data["city"], data["year"]
        expense, expected = data["expense"], data["expected"]

        if data["no_chunks"]:
            print(f"{state.upper():<6} {city:<20} {year:<6} {expense:<15} {'NO CHUNKS':<45} ${expected:>14,.0f}      --    --")
            results.append(data)
            total += 1
            continue

        if data["in_chunks"]:
            in_chunks_count += 1

        answer = answers.get(idx, "ERROR: no response")
        expected_int = int(expected)
        expected_str = f"${expected:,.0f}"

        answer_nums = extract_numbers(answer)
        match = expected_int in answer_nums
        if match:
            exact += 1
        total += 1
        flag = "OK" if match else "MISS"
        in_flag = "Y" if data["in_chunks"] else "N"

        result = {**data, "answer": answer, "match": match}
        results.append(result)

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
        "workers": args.workers,
        "total": total,
        "match": exact,
        "in_chunks": in_chunks_count,
        "results": results,
    }
    os.makedirs("runs", exist_ok=True)
    model_slug = args.model.rstrip("/").split("/")[-1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"runs/{model_slug}_c{args.n_chunks}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(run, f, indent=2)
    print(f"Results saved to {out_path}")

    # Update runs index for dashboard auto-loading
    run_files = sorted(f for f in os.listdir("runs") if f.endswith(".json") and f != "index.json")
    with open("runs/index.json", "w") as f:
        json.dump(run_files, f)

    # Upload to GCS bucket if specified
    if args.gcs_bucket:
        bucket_path = f"gs://{args.gcs_bucket}/runs/"
        print(f"Uploading to {bucket_path}...")
        for upload_file in [out_path, "runs/index.json"]:
            result = subprocess.run(
                ["gsutil", "cp", upload_file, bucket_path],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"  ERROR uploading {upload_file}: {result.stderr.strip()}")
            else:
                print(f"  Uploaded {upload_file}")
        public_url = f"https://storage.googleapis.com/{args.gcs_bucket}/runs/"
        print(f"Dashboard URL: dashboard.html?bucket={args.gcs_bucket}")


if __name__ == "__main__":
    main()
