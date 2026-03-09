"""Build a gold chunk cache from PDF pages directly (no ChromaDB needed).

Parses each PDF page-by-page with the specified parser, then builds a cache
in the same format as build_gold_set.py so test_llm_extraction.py can use it.

Usage:
  python build_page_cache.py --parser marker
  python build_page_cache.py --parser pymupdf
  python build_page_cache.py --parser marker --test-only
"""

import argparse
import json
import os
import sqlite3
import sys
import time

from paths import TRAINING_DIR, PDF_DIR
from pipeline.parsers import parse_page_text


def find_pdf(state, city, year):
    state = state.lower()
    city = city.lower().replace(" ", "_")
    yr = str(year)[2:]
    for pdf in PDF_DIR.glob("*.pdf"):
        if pdf.name.startswith(f"{state}_{city}_{yr}"):
            return pdf
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build gold chunk cache from PDF pages (no ChromaDB)",
    )
    parser.add_argument(
        "--parser", "-p", default="pymupdf",
        help="Parser to use (pymupdf, marker, pdfplumber, aryn)",
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Only build cache for test set (60 records)",
    )
    parser.add_argument(
        "--context-pages", type=int, default=4,
        help="Number of surrounding pages to include as context (default: 4)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file (default: training/gold_chunks_{parser}.json)",
    )
    args = parser.parse_args()

    output = args.output or str(
        TRAINING_DIR / f"gold_chunks_{args.parser}.json"
    )

    conn = sqlite3.connect("pipeline_state.db")
    conn.row_factory = sqlite3.Row

    if args.test_only:
        # Use test_budgets.json if it exists
        test_file = TRAINING_DIR / "test_budgets.json"
        if test_file.exists():
            with open(test_file) as f:
                test_keys = {
                    (r["state"].lower(), r["city"].lower().replace(" ", "_"), r["year"])
                    for r in json.load(f)
                }
        else:
            test_keys = None

    rows = conn.execute("""
        SELECT state, city, year, expense, budget_type, budget, pdf_page
        FROM validation
        WHERE pdf_page IS NOT NULL AND pdf_page > 0
    """).fetchall()

    print(f"Parser: {args.parser}")
    print(f"Validation records with pdf_page: {len(rows)}")

    cache = []
    skipped_no_pdf = 0
    skipped_test = 0
    errors = 0
    t0 = time.time()

    # Cache parsed pages per PDF to avoid re-parsing
    pdf_page_cache = {}

    for i, r in enumerate(rows):
        state, city, year = r["state"], r["city"], r["year"]
        expense = r["expense"]
        target_page = r["pdf_page"] - 1  # 0-indexed

        if args.test_only and test_keys is not None:
            key = (state.lower(), city.lower().replace(" ", "_"), year)
            if key not in test_keys:
                skipped_test += 1
                continue

        pdf_path = find_pdf(state, city, year)
        if not pdf_path:
            skipped_no_pdf += 1
            continue

        pdf_key = str(pdf_path)

        try:
            # Get target page
            target_text = parse_page_text(pdf_path, target_page, parser=args.parser)
            if not target_text:
                errors += 1
                continue

            # Build chunks: target page + surrounding context pages
            chunks = []

            # Target page as first chunk
            chunks.append({
                "text": target_text,
                "metadata": {
                    "filename": pdf_path.name,
                    "state": state.lower(),
                    "city": city.lower().replace(" ", "_"),
                    "year": year,
                    "page": target_page,
                    "parser": args.parser,
                    "has_table": "|" in target_text,
                    "is_target": True,
                },
            })

            # Add surrounding pages as context
            for offset in range(1, args.context_pages + 1):
                for page_idx in [target_page - offset, target_page + offset]:
                    if page_idx < 0:
                        continue
                    text = parse_page_text(pdf_path, page_idx, parser=args.parser)
                    if text:
                        chunks.append({
                            "text": text,
                            "metadata": {
                                "filename": pdf_path.name,
                                "state": state.lower(),
                                "city": city.lower().replace(" ", "_"),
                                "year": year,
                                "page": page_idx,
                                "parser": args.parser,
                                "has_table": "|" in text,
                                "is_target": False,
                            },
                        })

            cache.append({
                "state": state,
                "city": city,
                "year": year,
                "expense": expense,
                "budget_type": r["budget_type"],
                "budget": r["budget"],
                "chunks": chunks,
            })

        except Exception as e:
            errors += 1
            print(f"  ERROR: {state} {city} {year} {expense}: {e}")
            continue

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {len(cache)} cached, {i + 1}/{len(rows)} processed... ({elapsed:.0f}s)")

    with open(output, "w") as f:
        json.dump(cache, f)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output) / 1024 / 1024
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Cached: {len(cache)} records")
    print(f"Skipped (no PDF): {skipped_no_pdf}")
    if args.test_only:
        print(f"Skipped (not in test set): {skipped_test}")
    print(f"Errors: {errors}")
    print(f"Saved to {output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
