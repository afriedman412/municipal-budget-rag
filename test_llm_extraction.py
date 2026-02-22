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
from paths import TRAINING_DIR, RUNS_DIR


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


SYSTEM_PROMPT = """You are a municipal budget analyst. You will be given excerpts from a municipal budget document and asked to extract a specific expenditure amount. Follow these rules:
- Return EXPENDITURES only — never revenue or income figures
- Return the ADOPTED or APPROVED budget amount — never proposed, recommended, or estimated
- If both proposed and adopted values appear, you MUST return the adopted value
- Return the BUDGETED or APPROPRIATED amount, not actual/historical spending
- For Police, return the General Fund police expenditure, not all-funds or total city budget
- Return ONLY the numeric dollar amount (e.g. "$1,234,567")
- If you cannot find the value, return "NOT FOUND"
"""

USER_TEMPLATE = """Extract the total {expense} EXPENDITURE amount for {city}, {state} for fiscal year {year}.

Budget document excerpts:
---
{chunks}
---

ADOPTED {expense} expenditure for FY {year}:"""


def call_llm(llm, model, system, prompt):
    """Make a single LLM call. Returns the answer string."""
    response = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=50,
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
    parser.add_argument("--cache",
                        default=str(TRAINING_DIR / "gold_chunks_cache.json"),
                        help="Chunks cache file")
    parser.add_argument("--city", nargs=3, action="append", metavar=("CITY", "STATE", "YEAR"),
                        help="Filter to specific city/state/year (repeatable)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print chunks sent to LLM")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Parallel LLM requests (default: 1, try 8 for vLLM)")
    parser.add_argument("--version",
                        help="Model version tag (e.g. V5, V6) — stored in run JSON for dashboard")
    parser.add_argument("--gcs-bucket",
                        help="GCS bucket to upload results (e.g. muni-budget-runs)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to Weights & Biases")
    parser.add_argument("--wandb-project", default="muni-budget-rag",
                        help="W&B project name")
    args = parser.parse_args()

    llm = OpenAI(base_url=args.llm_url, api_key="not-needed")

    with open(args.cache) as f:
        cache = json.load(f)

    if args.city:
        city_filters = {(c.lower().replace(" ", "_"), s.lower(), int(y)) for c, s, y in args.city}
        rows = [r for r in cache if (r["city"].lower().replace(" ", "_"), r["state"].lower(), r["year"]) in city_filters]
    elif args.limit:
        rows = cache[:args.limit]
    else:
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
            f"[Chunk {i+1}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )
        prompt = USER_TEMPLATE.format(
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

    # Phase 2+3: Run LLM calls and print results as they arrive
    llm_tasks = [(idx, data, prompt) for idx, data, prompt in tasks if prompt is not None]
    task_map = {idx: data for idx, data, _ in tasks}

    print(f"{'State':<6} {'City':<20} {'Year':<6} {'Expense':<15} {'LLM Answer':<45} {'Expected':<20} {'In?':<5} {'Match'}")
    print("-" * 125)

    exact = 0
    total = 0
    in_chunks_count = 0
    results = []

    def process_result(idx, answer):
        """Score a result and print it. Returns the result dict."""
        nonlocal exact, total, in_chunks_count
        data = task_map[idx]
        state, city, year = data["state"], data["city"], data["year"]
        expense, expected = data["expense"], data["expected"]

        if data["in_chunks"]:
            in_chunks_count += 1

        expected_int = int(expected)
        expected_str = f"${expected:,.0f}"
        answer_nums = extract_numbers(answer)
        match = expected_int in answer_nums
        if match:
            exact += 1
        total += 1
        flag = "OK" if match else "MISS"
        in_flag = "Y" if data["in_chunks"] else "N"

        answer_display = answer[:40] + "..." if len(answer) > 40 else answer
        print(f"{state.upper():<6} {city:<20} {year:<6} {expense:<15} {answer_display:<45} {expected_str:<20} {in_flag:<5} {flag}")

        return {**data, "answer": answer, "match": match}

    # Print no-chunks rows first
    for idx, data, prompt in tasks:
        if data.get("no_chunks"):
            state, city, year = data["state"], data["city"], data["year"]
            expense, expected = data["expense"], data["expected"]
            print(f"{state.upper():<6} {city:<20} {year:<6} {expense:<15} {'NO CHUNKS':<45} ${expected:>14,.0f}      --    --")
            results.append(data)
            total += 1

    # Run LLM calls
    if args.workers > 1 and llm_tasks:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(call_llm, llm, args.model, SYSTEM_PROMPT, prompt): idx
                for idx, _, prompt in llm_tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    answer = future.result()
                except Exception as e:
                    answer = f"ERROR: {e}"
                results.append(process_result(idx, answer))
    else:
        for idx, data, prompt in llm_tasks:
            answer = call_llm(llm, args.model, SYSTEM_PROMPT, prompt)
            results.append(process_result(idx, answer))

    print("-" * 125)
    if total:
        print(f"Match: {exact}/{total} ({100*exact/total:.0f}%)  |  Answer in chunks: {in_chunks_count}/{total}")
    else:
        print("No records tested.")

    # Classify test set
    cache_name = os.path.basename(args.cache)
    has_distractors = "_d4" in cache_name or "adversarial" in cache_name
    if "val" in cache_name or total == 60:
        test_split = "validation"
    elif total <= 12:
        test_split = "test_budgets"
    else:
        test_split = "full"

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
        "version": args.version,
        "test_split": test_split,
        "distractors_test": has_distractors,
        "results": results,
    }
    os.makedirs(RUNS_DIR, exist_ok=True)
    model_slug = args.model.rstrip("/").split("/")[-1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(RUNS_DIR / f"{model_slug}_c{args.n_chunks}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(run, f, indent=2)
    print(f"Results saved to {out_path}")

    # Update runs index for dashboard auto-loading
    run_files = sorted(f for f in os.listdir(RUNS_DIR) if f.endswith(".json") and f != "index.json")
    with open(RUNS_DIR / "index.json", "w") as f:
        json.dump(run_files, f)

    # Log to W&B if specified
    if args.wandb:
        import wandb

        cache_name = os.path.basename(args.cache)
        run_name = f"{model_slug}_c{args.n_chunks}_{cache_name}"
        wb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model,
                "n_chunks": args.n_chunks,
                "cache_file": args.cache,
                "workers": args.workers,
                "total_records": total,
            },
        )

        # Summary metrics
        wandb.log({
            "exact_match": exact,
            "total": total,
            "exact_match_pct": 100 * exact / total if total else 0,
            "in_chunks": in_chunks_count,
            "in_chunks_pct": 100 * in_chunks_count / total if total else 0,
        })

        # Per-record table
        columns = ["state", "city", "year", "expense", "expected", "answer", "match", "in_chunks"]
        table = wandb.Table(columns=columns)
        for r in results:
            table.add_data(
                r.get("state", ""), r.get("city", ""), r.get("year", ""),
                r.get("expense", ""), r.get("expected", ""),
                r.get("answer", ""), r.get("match", False),
                r.get("in_chunks", False),
            )
        wandb.log({"results": table})
        wb_run.finish()
        print(f"W&B run logged: {run_name}")

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
