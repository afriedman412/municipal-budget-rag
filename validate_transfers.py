"""Validate ground truth: does each GF budget value include or exclude interfund transfers?

Uses a local LLM (Ollama) to classify each record by reading the chunk(s)
that contain the expected budget number.

Usage:
  ollama pull mistral
  python validate_transfers.py
  python validate_transfers.py --cache training/test_chunks_full.json
  python validate_transfers.py --llm-url http://localhost:11434/v1  # default (Ollama)
  python validate_transfers.py --limit 10  # test on first 10 records
"""

import argparse
import json
import re
import sys
import time

from openai import OpenAI


def progress_bar(current, total, counts, width=30):
    frac = current / total if total else 0
    filled = int(width * frac)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    parts = []
    for label in ["INCLUDES_TRANSFERS", "EXCLUDES_TRANSFERS", "AMBIGUOUS", "NO_CONTEXT", "NO_CHUNKS"]:
        short = {"INCLUDES_TRANSFERS": "INC", "EXCLUDES_TRANSFERS": "EXC",
                 "AMBIGUOUS": "AMB", "NO_CONTEXT": "NC", "NO_CHUNKS": "SKIP"}[label]
        n = counts.get(label, 0)
        if n:
            parts.append(f"{short}:{n}")
    stats = " ".join(parts)
    sys.stderr.write(f"\r  {bar} {current}/{total}  {stats}  ")
    sys.stderr.flush()


SYSTEM_PROMPT = """You are analyzing municipal budget documents. Given a text excerpt from a budget document, determine whether the General Fund total expenditure figure includes or excludes interfund transfers.

Respond with exactly one of:
- INCLUDES_TRANSFERS — if the figure clearly includes transfers (e.g., "Total Expenditures and Transfers Out")
- EXCLUDES_TRANSFERS — if the figure clearly excludes transfers (e.g., "Total Expenditures" with a separate "Transfers Out" line below)
- AMBIGUOUS — if you cannot determine from the context whether transfers are included
- NO_CONTEXT — if the chunk doesn't contain enough budget structure to tell

Do not explain. Just respond with one of the four labels."""

USER_PROMPT = """The expected General Fund total expenditure for {city}, {state} in {year} is ${budget:,.0f}.

Here is the budget text excerpt that contains this number:

---
{chunk_text}
---

Does this figure include or exclude interfund transfers?"""


def find_matching_chunks(record):
    """Find chunks that contain the expected budget number."""
    budget = record["budget"]
    formatted = f"{budget:,.0f}"
    plain = str(int(budget))

    matching = []
    for chunk in record["chunks"]:
        text = chunk["text"]
        if formatted in text or plain in text:
            matching.append(chunk)
    return matching


def classify_record(client, model, record, matching_chunks):
    """Ask LLM to classify whether the GF figure includes transfers."""
    # Use the first matching chunk (usually the most relevant)
    chunk_text = matching_chunks[0]["text"]

    # Trim chunk if very long (keep context around the number)
    if len(chunk_text) > 3000:
        chunk_text = chunk_text[:3000] + "\n[...truncated]"

    prompt = USER_PROMPT.format(
        city=record["city"],
        state=record["state"],
        year=record["year"],
        budget=record["budget"],
        chunk_text=chunk_text,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=20,
    )

    raw = response.choices[0].message.content.strip()
    # Normalize response
    upper = raw.upper().replace(" ", "_")
    for label in ["INCLUDES_TRANSFERS", "EXCLUDES_TRANSFERS", "AMBIGUOUS", "NO_CONTEXT"]:
        if label in upper:
            return label
    return f"UNKNOWN({raw})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="training/test_chunks_full.json",
                   help="Cached chunks file")
    p.add_argument("--llm-url", default="http://localhost:11434/v1",
                   help="LLM API base URL (default: Ollama)")
    p.add_argument("--model", default="mistral",
                   help="Model name (default: mistral)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit to first N GF records")
    p.add_argument("--output", default="training/transfers_audit.json",
                   help="Output file")
    args = p.parse_args()

    data = json.load(open(args.cache))
    gf_records = [r for r in data if r["expense"] == "General Fund"]
    print(f"Loaded {len(gf_records)} General Fund records")

    if args.limit:
        gf_records = gf_records[:args.limit]
        print(f"Limited to {args.limit} records")

    client = OpenAI(base_url=args.llm_url, api_key="not-needed")

    results = []
    counts = {"INCLUDES_TRANSFERS": 0, "EXCLUDES_TRANSFERS": 0,
              "AMBIGUOUS": 0, "NO_CONTEXT": 0, "NO_CHUNKS": 0}

    for i, rec in enumerate(gf_records):
        matching = find_matching_chunks(rec)
        label_key = f"{rec['state']} {rec['city']} {rec['year']}"

        if not matching:
            label = "NO_CHUNKS"
        else:
            label = classify_record(client, args.model, rec, matching)
        counts[label] = counts.get(label, 0) + 1
        progress_bar(i + 1, len(gf_records), counts)

        results.append({
            "state": rec["state"],
            "city": rec["city"],
            "year": rec["year"],
            "budget": rec["budget"],
            "transfer_status": label,
            "n_matching_chunks": len(matching),
        })

    # Summary
    sys.stderr.write("\r" + " " * 80 + "\r")
    sys.stderr.flush()
    print("=" * 50)
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count:
            print(f"  {label:<22s} {count:4d}  ({100*count/len(gf_records):.0f}%)")
    print(f"  {'TOTAL':<22s} {len(gf_records):4d}")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
