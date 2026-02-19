"""Generate fine-tuning training data from validation table + local PDFs,
and/or from cached retrieval chunks (Aryn/PyMuPDF).

For PDF-based examples:
  - Parse the target page (where the answer is) + structurally similar distractor pages
  - Build the same prompt format used at inference time

For chunk-cache-based examples:
  - Use pre-retrieved chunks from gold_chunks_*.json files
  - Simulates real inference conditions (same chunks the model will see)
  - No PDF parsing needed — fast and reliable

Output: JSONL with chat-format messages (compatible with most fine-tuning frameworks)

Usage:
  python build_training_data.py                                        # PDF-based, 500 examples
  python build_training_data.py --chunks-cache training/gold_chunks_cache.json  # From cached chunks
  python build_training_data.py --chunks-only training/gold_chunks_aryn.json   # Only cached chunks, no PDFs
"""

import argparse
import json
import os
import random
import re
import sqlite3
import time

from paths import TRAINING_DIR, PDF_DIR

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
    import fitz
    doc = fitz.open(pdf_path)
    if page_num < 0 or page_num >= len(doc):
        return None
    text = doc[page_num].get_text()
    doc.close()
    return text.strip()


def _page_features(page):
    """Compute structural features for a single fitz Page object."""
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


FEATURE_CACHE_FILE = "page_features_cache.json"


def _load_feature_cache():
    """Load persistent page features cache from disk."""
    if os.path.exists(FEATURE_CACHE_FILE):
        with open(FEATURE_CACHE_FILE) as f:
            # JSON keys are strings, page nums need int conversion
            raw = json.load(f)
            return {k: {int(p): v for p, v in pages.items()} for k, pages in raw.items()}
    return {}


def _save_feature_cache(cache):
    """Save page features cache to disk."""
    with open(FEATURE_CACHE_FILE, "w") as f:
        json.dump(cache, f)


_feature_cache = None


def get_pdf_features(pdf_path):
    """Get page features for all pages in a PDF (cached to disk)."""
    global _feature_cache
    if _feature_cache is None:
        _feature_cache = _load_feature_cache()
    key = str(pdf_path)
    if key not in _feature_cache:
        import fitz
        doc = fitz.open(pdf_path)
        _feature_cache[key] = {p: _page_features(doc[p]) for p in range(len(doc))}
        doc.close()
        _save_feature_cache(_feature_cache)
    return _feature_cache[key]


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
    features = get_pdf_features(pdf_path)
    target_feat = features[target_page]

    # Score all other pages
    candidates = [
        (p, page_similarity(target_feat, feat))
        for p, feat in features.items()
        if p != target_page
    ]

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


def build_example(record, chunk_text):
    """Build a training example from a record and chunk text."""
    user_msg = USER_TEMPLATE.format(
        expense=record["expense"],
        city=record["city"],
        state=record["state"],
        year=record["year"],
        chunks=chunk_text,
    )
    assistant_msg = format_budget(record["budget"])
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def generate_pdf_examples(f, sample, args_distractors):
    """Generate training examples from PDFs. Writes to file handle f."""
    count = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    for i, (record, pdf_path) in enumerate(sample):
        try:
            target_page = record["pdf_page"] - 1  # Convert to 0-indexed

            target_text = extract_page_text(pdf_path, target_page)
            if not target_text:
                skipped += 1
                continue

            distractor_pages = pick_distractors(pdf_path, target_page, args_distractors)

            distractor_texts = []
            for dp in distractor_pages:
                text = extract_page_text(pdf_path, dp)
                if text:
                    distractor_texts.append(text)

            all_chunks = [(target_text, "target")] + [(t, "distractor") for t in distractor_texts]
            random.shuffle(all_chunks)

            chunk_text = "\n\n".join(
                f"[Page {j+1}]\n{text}"
                for j, (text, _) in enumerate(all_chunks)
            )

            example = build_example(record, chunk_text)
            f.write(json.dumps(example) + "\n")
            f.flush()
            count += 1

        except Exception as e:
            errors += 1
            print(f"  ERROR on {record['state']} {record['city']} {record['year']}: {e}")
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [PDF] {i + 1}/{len(sample)} processed, {count} written... ({elapsed:.0f}s elapsed)")

    return count, skipped, errors


def generate_chunk_cache_examples(f, cache_path, n_chunks_per_example, n_examples=None):
    """Generate training examples from cached chunks. Writes to file handle f."""
    with open(cache_path) as cf:
        cache = json.load(cf)

    # Filter to GF + Police with valid budget
    target_expenses = {"general fund", "police"}
    records = [r for r in cache if r.get("budget") and r["expense"].lower() in target_expenses]

    if n_examples and n_examples < len(records):
        records = random.sample(records, n_examples)

    count = 0
    t0 = time.time()

    for i, record in enumerate(records):
        chunks = record.get("chunks", [])
        if not chunks:
            continue

        # Use top N chunks (same as inference)
        top_chunks = chunks[:n_chunks_per_example]
        chunk_text = "\n\n".join(
            f"[Chunk {j+1}]\n{c['text']}"
            for j, c in enumerate(top_chunks)
        )

        example = build_example(record, chunk_text)
        f.write(json.dumps(example) + "\n")
        f.flush()
        count += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [Chunks] {i + 1}/{len(records)} processed, {count} written... ({elapsed:.0f}s elapsed)")

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500,
                        help="Number of PDF-based training examples to generate")
    parser.add_argument("--distractors", type=int, default=4,
                        help="Number of distractor pages per PDF example")
    parser.add_argument("--n-chunks", type=int, default=10,
                        help="Number of chunks per cache-based example")
    parser.add_argument("--chunks-cache", type=str, action="append", default=[],
                        help="Cached chunks file(s) to generate examples from (can repeat)")
    parser.add_argument("--chunks-only", type=str, action="append", default=[],
                        help="Like --chunks-cache but skips PDF-based examples entirely")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", default=str(TRAINING_DIR / "training_data.jsonl"),
                        help="Output file")
    args = parser.parse_args()

    random.seed(args.seed)

    all_chunk_caches = args.chunks_cache + args.chunks_only
    skip_pdfs = len(args.chunks_only) > 0 and len(args.chunks_cache) == 0

    total_count = 0

    with open(args.output, "w") as f:

        # --- PDF-based examples ---
        if not skip_pdfs:
            print("=== PDF-based examples ===")
            conn = sqlite3.connect("pipeline_state.db")
            conn.row_factory = sqlite3.Row

            rows = conn.execute("""
                SELECT state, city, year, expense, budget_type, budget, pdf_page
                FROM validation
                WHERE pdf_page IS NOT NULL AND pdf_page > 0
            """).fetchall()

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

            target_expenses = ["General Fund", "Police"]
            per_type = args.n // len(target_expenses)
            sample = []
            for expense in target_expenses:
                cands = by_expense.get(expense, [])
                n = min(per_type, len(cands))
                sample.extend(random.sample(cands, n))
                print(f"  Sampled {n} {expense}")

            random.shuffle(sample)

            count, skipped, errors = generate_pdf_examples(f, sample, args.distractors)
            total_count += count
            print(f"  PDF: {count} examples ({skipped} skipped, {errors} errors)")

        # --- Chunk-cache-based examples ---
        for cache_path in all_chunk_caches:
            print(f"\n=== Chunk-cache examples: {cache_path} ===")
            count = generate_chunk_cache_examples(f, cache_path, args.n_chunks)
            total_count += count
            print(f"  Cache: {count} examples from {cache_path}")

    print(f"\nTotal: {total_count} training examples")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
