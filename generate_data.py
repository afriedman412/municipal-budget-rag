"""Generate training data (JSONL) or test chunk caches (JSON) from pymupdf_split.json.

Replaces the separate generate_v3_training.py, generate_v3_test_cache.py,
generate_v3_test_adversarial.py, and generate_v4_training.py scripts.

Usage:
  # Training data (no distractors)
  python generate_data.py train

  # Training data with 4 random distractor pages
  python generate_data.py train --distractors 4

  # Test cache for validation set (no distractors)
  python generate_data.py test --split val

  # Test cache for full set with distractors
  python generate_data.py test --split full --distractors 4

  # Custom output path
  python generate_data.py train --output training/my_custom.jsonl
"""

import argparse
import json
import random

import fitz
from tqdm import tqdm

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


def get_pages(doc, target_idx, n_distractors, rng):
    """Get target page + optional random distractor pages, shuffled.

    Returns list of (page_text, page_number_1indexed) tuples.
    """
    target_text = doc[target_idx].get_text().strip()
    if len(target_text) < 50:
        return None

    pages = [(target_text, target_idx + 1)]

    if n_distractors > 0:
        other = [p for p in range(len(doc)) if p != target_idx]
        rng.shuffle(other)
        added = 0
        for p in other:
            if added >= n_distractors:
                break
            text = doc[p].get_text().strip()
            if len(text) > 100:
                pages.append((text, p + 1))
                added += 1
        rng.shuffle(pages)

    return pages


def build_training(split_records, n_distractors, rng):
    """Build chat-format training examples (JSONL)."""
    examples = []
    errors = 0

    for rec in tqdm(split_records, desc="Training"):
        pdf_path = PDF_DIR / rec["pdf"]
        if not pdf_path.exists():
            errors += 1
            continue

        doc = fitz.open(str(pdf_path))

        for expense_key, expense_label in [("general_fund", "General Fund"), ("police", "Police")]:
            info = rec[expense_key]
            page_num = info["pdf_page"]
            budget = info["budget"]

            if page_num < 1 or page_num > len(doc):
                errors += 1
                continue

            pages = get_pages(doc, page_num - 1, n_distractors, rng)
            if pages is None:
                errors += 1
                continue

            chunk_text = "\n\n".join(
                f"[Chunk {j+1}]\n{text}" for j, (text, _) in enumerate(pages)
            )

            user_msg = USER_TEMPLATE.format(
                expense=expense_label,
                city=rec["city"],
                state=rec["state"],
                year=rec["year"],
                chunks=chunk_text,
            )

            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": f"${int(budget):,}"},
                ]
            })

        doc.close()

    return examples, errors


def build_test_cache(split_records, n_distractors, rng):
    """Build test chunk cache (JSON)."""
    records = []

    for rec in tqdm(split_records, desc="Test cache"):
        pdf_path = PDF_DIR / rec["pdf"]
        if not pdf_path.exists():
            continue

        doc = fitz.open(str(pdf_path))

        for expense_key, expense_label in [("general_fund", "General Fund"), ("police", "Police")]:
            info = rec[expense_key]
            page_num = info["pdf_page"]
            budget = info["budget"]

            if page_num < 1 or page_num > len(doc):
                continue

            pages = get_pages(doc, page_num - 1, n_distractors, rng)
            if pages is None:
                continue

            chunks = [
                {
                    "text": text,
                    "metadata": {
                        "filename": rec["pdf"],
                        "parser": "pymupdf",
                        "has_table": False,
                        "page": page_num_1,
                    },
                }
                for text, page_num_1 in pages
            ]

            records.append({
                "state": rec["state"],
                "city": rec["city"],
                "year": rec["year"],
                "expense": expense_label,
                "budget": budget,
                "chunks": chunks,
            })

        doc.close()

    return records


def default_output_name(mode, split, n_distractors):
    """Generate a default output filename."""
    suffix = f"_d{n_distractors}" if n_distractors > 0 else ""
    if mode == "train":
        return TRAINING_DIR / f"training_data{suffix}.jsonl"
    else:
        return TRAINING_DIR / f"test_chunks_{split}{suffix}.json"


def main():
    parser = argparse.ArgumentParser(description="Generate training or test data from pymupdf_split.json")
    parser.add_argument("mode", choices=["train", "test"], help="train = JSONL for fine-tuning, test = JSON chunk cache")
    parser.add_argument("--split", choices=["train", "val", "full"], default=None,
                        help="Which split to use (default: train for training, val for test)")
    parser.add_argument("--distractors", type=int, default=0, help="Number of random distractor pages (0 = target only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output path (auto-generated if not specified)")
    args = parser.parse_args()

    # Default split
    if args.split is None:
        args.split = "train" if args.mode == "train" else "val"

    rng = random.Random(args.seed)

    with open(TRAINING_DIR / "pymupdf_split.json") as f:
        split = json.load(f)

    if args.split == "full":
        records = split["training"] + split["validation"]
    elif args.split == "train":
        records = split["training"]
    else:
        records = split["validation"]

    output_path = args.output or default_output_name(args.mode, args.split, args.distractors)

    dist_label = f"{args.distractors} distractors" if args.distractors > 0 else "no distractors"
    print(f"Mode: {args.mode}, split: {args.split} ({len(records)} cities), {dist_label}")

    if args.mode == "train":
        examples, errors = build_training(records, args.distractors, rng)
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        avg_chars = sum(len(ex["messages"][1]["content"]) for ex in examples) // max(len(examples), 1)
        print(f"{len(examples)} examples, {errors} errors, ~{avg_chars} avg chars/prompt")
    else:
        cache_records = build_test_cache(records, args.distractors, rng)
        with open(output_path, "w") as f:
            json.dump(cache_records, f, indent=2)
        avg_chunks = sum(len(r["chunks"]) for r in cache_records) / max(len(cache_records), 1)
        print(f"{len(cache_records)} records, ~{avg_chunks:.1f} avg chunks/record")

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
