"""Select ~100 new PDFs for PyMuPDF training/validation, avoiding existing gold set.

Checks:
1. Has ground truth for BOTH GF and Police in validation table
2. Not already in gold_chunks_pymupdf.json
3. PDF exists locally
4. PyMuPDF can extract meaningful text (not garbled/empty)
5. Target page text contains the expected budget value

Outputs a JSON file with train/validation split.
"""

import json
import random
import re
import sqlite3
import sys
from pathlib import Path

import fitz

from paths import TRAINING_DIR, PDF_DIR

GOLD_CHUNKS_FILE = TRAINING_DIR / "gold_chunks_pymupdf.json"
OUTPUT_FILE = TRAINING_DIR / "pymupdf_split.json"

random.seed(42)


def find_pdf(state, city, year):
    """Find the local PDF for a given city/year."""
    state = state.lower()
    city = city.lower().replace(" ", "_")
    yr = str(year)[2:]
    for pdf in PDF_DIR.glob("*.pdf"):
        if pdf.name.startswith(f"{state}_{city}_{yr}"):
            return pdf
    return None


def check_pymupdf_readable(pdf_path, target_page, expected_budget):
    """Check if PyMuPDF can extract usable text and the target page has the value."""
    try:
        doc = fitz.open(str(pdf_path))
        if target_page < 1 or target_page > len(doc):
            doc.close()
            return False, "page out of range"

        # Check target page has text
        page_text = doc[target_page - 1].get_text()
        doc.close()

        if len(page_text.strip()) < 100:
            return False, "page too short"

        # Check if expected value appears on target page
        expected_str = f"{int(expected_budget):,}"
        expected_plain = str(int(expected_budget))
        if expected_str not in page_text and expected_plain not in page_text:
            return False, "value not on target page"

        return True, "ok"
    except Exception as e:
        return False, str(e)


def main():
    # Load existing gold set to exclude
    with open(GOLD_CHUNKS_FILE) as f:
        gold = json.load(f)
    used = {(r["state"].lower(), r["city"].lower().replace(" ", "_"), r["year"]) for r in gold}
    print(f"Excluding {len(used)} already-used (state, city, year) tuples")

    # Get validation records with both GF and Police
    conn = sqlite3.connect("pipeline_state.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT state, city, year, expense, budget, pdf_page
        FROM validation
        WHERE expense IN ('General Fund', 'Police')
          AND budget > 0
          AND pdf_page > 0
    """).fetchall()

    # Group by (state, city, year)
    by_city_year = {}
    for r in rows:
        key = (r["state"].lower(), r["city"].lower().replace(" ", "_"), r["year"])
        if key not in by_city_year:
            by_city_year[key] = {}
        by_city_year[key][r["expense"]] = {
            "budget": r["budget"],
            "pdf_page": r["pdf_page"],
            "state": r["state"],
            "city": r["city"],
            "year": r["year"],
        }

    # Filter to those with BOTH GF and Police, not already used
    candidates = {}
    for key, expenses in by_city_year.items():
        if key in used:
            continue
        if "General Fund" in expenses and "Police" in expenses:
            candidates[key] = expenses

    print(f"Candidates with both GF+Police, not in gold set: {len(candidates)}")

    # Check each candidate: PDF exists + PyMuPDF readable + value on page
    valid = []
    skip_reasons = {"no_pdf": 0, "gf_fail": 0, "police_fail": 0}

    for key, expenses in sorted(candidates.items()):
        state, city, year = key
        pdf = find_pdf(expenses["General Fund"]["state"],
                       expenses["General Fund"]["city"],
                       year)
        if not pdf:
            skip_reasons["no_pdf"] += 1
            continue

        gf = expenses["General Fund"]
        police = expenses["Police"]

        ok_gf, reason_gf = check_pymupdf_readable(pdf, gf["pdf_page"], gf["budget"])
        if not ok_gf:
            skip_reasons["gf_fail"] += 1
            continue

        ok_police, reason_police = check_pymupdf_readable(pdf, police["pdf_page"], police["budget"])
        if not ok_police:
            skip_reasons["police_fail"] += 1
            continue

        valid.append({
            "state": gf["state"],
            "city": gf["city"],
            "year": year,
            "pdf": pdf.name,
            "general_fund": {"budget": gf["budget"], "pdf_page": gf["pdf_page"]},
            "police": {"budget": police["budget"], "pdf_page": police["pdf_page"]},
        })

    print(f"\nSkip reasons: {skip_reasons}")
    print(f"Valid candidates: {len(valid)}")

    if len(valid) < 100:
        print(f"WARNING: Only {len(valid)} valid candidates (wanted 100)")
        selected = valid
    else:
        selected = random.sample(valid, 100)

    # Split: 30 validation, 70 training
    random.shuffle(selected)
    validation = selected[:30]
    training = selected[30:]

    result = {
        "training": training,
        "validation": validation,
        "metadata": {
            "total_candidates": len(valid),
            "selected": len(selected),
            "training_count": len(training),
            "validation_count": len(validation),
            "skip_reasons": skip_reasons,
        }
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {len(training)} training + {len(validation)} validation to {OUTPUT_FILE}")

    # Summary stats
    for split_name, split_data in [("Training", training), ("Validation", validation)]:
        states = set(r["state"] for r in split_data)
        print(f"  {split_name}: {len(split_data)} cities across {len(states)} states")


if __name__ == "__main__":
    main()
