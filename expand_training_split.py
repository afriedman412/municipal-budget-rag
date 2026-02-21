"""Expand pymupdf_split.json to use ALL valid candidates for training.

Keeps the same 30 validation cities. Adds all remaining valid candidates
(both the unused 586 from the original pool and the existing 103 training)
to the training set.
"""

import json
import random
import sqlite3

import fitz
from tqdm import tqdm

from paths import TRAINING_DIR, PDF_DIR

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

        page_text = doc[target_page - 1].get_text()
        doc.close()

        if len(page_text.strip()) < 100:
            return False, "page too short"

        expected_str = f"{int(expected_budget):,}"
        expected_plain = str(int(expected_budget))
        if expected_str not in page_text and expected_plain not in page_text:
            return False, "value not on target page"

        return True, "ok"
    except Exception as e:
        return False, str(e)


def main():
    # Load current split to preserve validation set
    with open(OUTPUT_FILE) as f:
        current = json.load(f)

    val_cities = current["validation"]
    val_keys = {(r["state"].lower(), r["city"].lower().replace(" ", "_"), r["year"])
                for r in val_cities}
    print(f"Preserving {len(val_keys)} validation cities")

    # Get ALL validation records with both GF and Police from DB
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

    # Filter to those with BOTH GF and Police, excluding validation cities
    candidates = {}
    for key, expenses in by_city_year.items():
        if key in val_keys:
            continue
        if "General Fund" in expenses and "Police" in expenses:
            candidates[key] = expenses

    print(f"Training candidates (both GF+Police, not in val): {len(candidates)}")

    # Check each: PDF exists + PyMuPDF readable + value on page
    valid = []
    skip_reasons = {"no_pdf": 0, "gf_fail": 0, "police_fail": 0}

    for key, expenses in tqdm(sorted(candidates.items()), desc="Checking"):
        state, city, year = key
        pdf = find_pdf(expenses["General Fund"]["state"],
                       expenses["General Fund"]["city"],
                       year)
        if not pdf:
            skip_reasons["no_pdf"] += 1
            continue

        gf = expenses["General Fund"]
        police = expenses["Police"]

        ok_gf, _ = check_pymupdf_readable(pdf, gf["pdf_page"], gf["budget"])
        if not ok_gf:
            skip_reasons["gf_fail"] += 1
            continue

        ok_police, _ = check_pymupdf_readable(pdf, police["pdf_page"], police["budget"])
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
    print(f"Valid training candidates: {len(valid)}")

    # Use ALL valid candidates for training
    random.shuffle(valid)

    result = {
        "training": valid,
        "validation": val_cities,
        "metadata": {
            "total_candidates_checked": len(candidates),
            "valid_training": len(valid),
            "validation_count": len(val_cities),
            "skip_reasons": skip_reasons,
        }
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    train_states = set(r["state"] for r in valid)
    val_states = set(r["state"] for r in val_cities)
    print(f"\nSaved to {OUTPUT_FILE}:")
    print(f"  Training: {len(valid)} cities across {len(train_states)} states")
    print(f"  Validation: {len(val_cities)} cities across {len(val_states)} states (unchanged)")
    print(f"  Training examples will be: {len(valid) * 2} (x2 for GF + Police)")


if __name__ == "__main__":
    main()
