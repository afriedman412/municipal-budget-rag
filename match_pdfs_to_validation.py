"""Find PDFs that match validation records and are suitable for training."""

import sqlite3
import re
from pathlib import Path

PDF_DIR = Path("pdfs_2026")

# Skip these suffixes — not adopted budgets
SKIP_SUFFIXES = {"proposed", "prop", "mayor", "funds", "draft", "midbi",
                 "gf", "summary", "plan", "update", "expenses", "expenditures",
                 "biennial", "bi", "mayor_final", "adoption", "tentative"}

conn = sqlite3.connect("pipeline_state.db")
conn.row_factory = sqlite3.Row

# Get all unique (state, city, year) combos from validation
val_cities = conn.execute("""
    SELECT DISTINCT LOWER(state) as state, LOWER(REPLACE(city, ' ', '_')) as city, year
    FROM validation
    ORDER BY state, city, year
""").fetchall()

print(f"Validation has {len(val_cities)} unique (state, city, year) combos")

# Try to match each to a PDF
matched = []
unmatched = 0

for row in val_cities:
    state, city, year = row["state"], row["city"], row["year"]
    yy = str(year)[-2:]  # 2022 -> "22"

    # Try SS_city_YY.pdf pattern
    candidates = list(PDF_DIR.glob(f"{state}_{city}_{yy}*.pdf"))

    if not candidates:
        # Try city_SS_YY pattern (some files use this)
        candidates = list(PDF_DIR.glob(f"{city}_{state}_{yy}*.pdf"))

    if not candidates:
        unmatched += 1
        continue

    # Filter out bad suffixes
    good = []
    for c in candidates:
        name = c.stem.lower()
        # Extract suffix after the year digits
        # e.g. "ak_fairbanks_22_adopted" -> "adopted"
        parts = name.split("_")
        suffix_parts = []
        found_year = False
        for p in parts:
            if found_year:
                suffix_parts.append(p)
            if re.match(r"^\d{2}$", p):
                found_year = True

        suffix = "_".join(suffix_parts) if suffix_parts else ""

        if suffix in SKIP_SUFFIXES or any(s in suffix for s in ["prop", "draft", "mayor", "funds", "gf", "summary"]):
            continue
        good.append(c)

    if not good:
        # All filtered out — skip this one, no suitable PDF
        if candidates:
            unmatched += 1
        continue

    if good:
        # Pick the best one (prefer adopted > approved > plain)
        best = good[0]
        for g in good:
            if "adopted" in g.stem:
                best = g
                break
            if "approved" in g.stem:
                best = g

        matched.append((state, city, year, best.name))

print(f"Matched: {len(matched)} PDFs")
print(f"Unmatched: {unmatched}")

# Deduplicate by filename (same PDF may match multiple years)
unique_files = sorted(set(m[3] for m in matched))
print(f"Unique PDFs: {len(unique_files)}")

# Show state distribution
from collections import Counter
state_counts = Counter(m[0] for m in matched)
print(f"\nTop 15 states:")
for state, count in state_counts.most_common(15):
    print(f"  {state.upper()}: {count} records")

# Show how many validation records we'd cover
print(f"\nValidation records covered: {len(matched)}")
total_val = conn.execute("SELECT COUNT(*) FROM validation").fetchone()[0]
print(f"Total validation records: {total_val}")
print(f"Coverage: {100*len(matched)/total_val:.0f}%")

# Write the file list
with open("matched_pdfs.txt", "w") as f:
    for name in unique_files:
        f.write(name + "\n")

print(f"\nWrote {len(unique_files)} filenames to matched_pdfs.txt")
