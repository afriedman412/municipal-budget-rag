"""Generate fine-tuning training data from validation table + local PDFs.

For each training example:
  - Parse the target page (where the answer is) + random distractor pages
  - Build the same prompt format used at inference time
  - Output: JSONL with chat-format messages (compatible with most fine-tuning frameworks)

Usage:
  python build_training_data.py                    # 500 examples (default)
  python build_training_data.py --n 1000           # More examples
  python build_training_data.py --distractors 5    # More distractor pages per example
"""

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path

import fitz  # PyMuPDF


PDF_DIR = Path("pdfs_2026")

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


def find_pdf(state, city, year):
    """Find the local PDF for a given city/year."""
    state = state.lower()
    city = city.lower().replace(" ", "_")
    yr = str(year)[2:]

    for pdf in PDF_DIR.glob("*.pdf"):
        if pdf.name.startswith(f"{state}_{city}_{yr}"):
            return pdf
    return None


def extract_page_text(pdf_path, page_num):
    """Extract text from a single PDF page (0-indexed)."""
    doc = fitz.open(pdf_path)
    if page_num < 0 or page_num >= len(doc):
        return None
    text = doc[page_num].get_text()
    doc.close()
    return text.strip()


def get_total_pages(pdf_path):
    """Get total page count of a PDF."""
    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()
    return n


def page_features(page):
    """Compute structural features for a PDF page (pass a fitz Page object)."""
    text = page.get_text()
    blocks = page.get_text("dict")["blocks"]
    n_text = sum(1 for b in blocks if b["type"] == 0)
    n_image = sum(1 for b in blocks if b["type"] == 1)
    text_len = len(text)
    # Table heuristic: lines with 3+ number-like tokens
    lines = text.split('\n')
    table_lines = sum(1 for l in lines if len(re.findall(r'\d[\d,]+', l)) >= 3)
    has_tables = table_lines > 3
    return {
        "n_text": n_text, "n_image": n_image,
        "text_len": text_len, "has_tables": has_tables,
    }


def page_similarity(target_feat, candidate_feat):
    """Score how similar a candidate page is to the target (lower = more similar)."""
    # Penalize text length difference
    len_diff = abs(target_feat["text_len"] - candidate_feat["text_len"]) / max(target_feat["text_len"], 1)
    # Reward matching table presence
    table_bonus = -0.5 if target_feat["has_tables"] == candidate_feat["has_tables"] else 0.0
    # Penalize image-only pages (no text blocks)
    image_penalty = 2.0 if candidate_feat["n_text"] == 0 else 0.0
    # Penalize very short pages (TOC, cover, dividers)
    short_penalty = 1.0 if candidate_feat["text_len"] < 200 else 0.0
    return len_diff + table_bonus + image_penalty + short_penalty


def pick_distractors(pdf_path, target_page, n_distractors):
    """Pick distractor pages that structurally resemble the target page."""
    doc = fitz.open(pdf_path)
    total = len(doc)
    target_feat = page_features(doc[target_page])

    # Score all other pages
    candidates = []
    for p in range(total):
        if p == target_page:
            continue
        feat = page_features(doc[p])
        score = page_similarity(target_feat, feat)
        candidates.append((p, score))
    doc.close()

    # Sort by similarity (lowest score = most similar), take top N with some randomness
    candidates.sort(key=lambda x: x[1])
    # Pick from top 3x candidates to keep some variety
    pool_size = min(len(candidates), n_distractors * 3)
    pool = candidates[:pool_size]
    chosen = random.sample(pool, min(n_distractors, len(pool)))
    return [p for p, _ in chosen]


def format_budget(value):
    """Format budget as dollar string."""
    return f"${value:,.0f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500,
                        help="Number of training examples to generate")
    parser.add_argument("--distractors", type=int, default=4,
                        help="Number of random distractor pages per example")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", default="training_data.jsonl",
                        help="Output file")
    args = parser.parse_args()

    random.seed(args.seed)

    conn = sqlite3.connect("pipeline_state.db")
    conn.row_factory = sqlite3.Row

    # Get all validation records with page numbers
    rows = conn.execute("""
        SELECT state, city, year, expense, budget_type, budget, pdf_page
        FROM validation
        WHERE pdf_page IS NOT NULL AND pdf_page > 0
    """).fetchall()

    # Filter to records where we have the PDF locally, grouped by expense
    by_expense = {}
    for r in rows:
        pdf_path = find_pdf(r["state"], r["city"], r["year"])
        if pdf_path:
            expense = r["expense"]
            if expense not in by_expense:
                by_expense[expense] = []
            by_expense[expense].append((dict(r), pdf_path))

    for expense, cands in sorted(by_expense.items()):
        print(f"  {expense}: {len(cands)} candidates")

    # Stratified sampling: equal per expense type (GF + Police)
    target_expenses = ["General Fund", "Police"]
    per_type = args.n // len(target_expenses)
    sample = []
    for expense in target_expenses:
        cands = by_expense.get(expense, [])
        n = min(per_type, len(cands))
        sample.extend(random.sample(cands, n))
        print(f"  Sampled {n} {expense}")

    random.shuffle(sample)

    # Generate training examples
    examples = []
    skipped = 0
    for i, (record, pdf_path) in enumerate(sample):
        target_page = record["pdf_page"] - 1  # Convert to 0-indexed

        # Extract target page text
        target_text = extract_page_text(pdf_path, target_page)
        if not target_text:
            skipped += 1
            continue

        # Pick distractor pages that structurally resemble the target page
        distractor_pages = pick_distractors(pdf_path, target_page, args.distractors)

        # Extract distractor texts
        distractor_texts = []
        for dp in distractor_pages:
            text = extract_page_text(pdf_path, dp)
            if text:
                distractor_texts.append(text)

        # Combine: shuffle target among distractors
        all_chunks = [(target_text, "target")] + [(t, "distractor") for t in distractor_texts]
        random.shuffle(all_chunks)

        # Build chunk text (same format as inference)
        chunk_text = "\n\n".join(
            f"[Page {j+1}]\n{text}"
            for j, (text, _) in enumerate(all_chunks)
        )

        # Build messages
        user_msg = USER_TEMPLATE.format(
            expense=record["expense"],
            city=record["city"],
            state=record["state"],
            year=record["year"],
            chunks=chunk_text,
        )

        assistant_msg = format_budget(record["budget"])

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }
        examples.append(example)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sample)} examples generated...")

    # Write output
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nGenerated {len(examples)} training examples ({skipped} skipped)")
    print(f"Saved to {args.output}")

    # Stats
    expenses = {}
    for ex in examples:
        msg = ex["messages"][1]["content"]
        for etype in ["General Fund", "Police", "Education"]:
            if etype in msg:
                expenses[etype] = expenses.get(etype, 0) + 1
                break
    print(f"Expense distribution: {expenses}")


if __name__ == "__main__":
    main()
