"""Recall@k sweep: how many retrieved chunks needed to find the target value?"""

import argparse
import json
from pathlib import Path

GOLD_DIR = Path("training")
K_VALUES = [1, 2, 3, 5, 10, 15, 20, 30, 40]


def first_hit_index(chunks, expected_int):
    """Return 0-based index of first chunk containing the expected value, or -1."""
    formatted = f"{expected_int:,}"
    plain = str(expected_int)
    for i, c in enumerate(chunks):
        text = c["text"]
        if formatted in text or plain in text:
            return i
    return -1


def run_sweep(parser):
    path = GOLD_DIR / f"gold_chunks_{parser}.json"
    if not path.exists():
        print(f"  {path} not found, skipping")
        return

    with open(path) as f:
        records = json.load(f)

    hits = []  # (first_hit_index, record_info) for found records
    misses = []  # record_info for not-found records

    for rec in records:
        expected_int = int(rec["budget"])
        idx = first_hit_index(rec["chunks"], expected_int)
        info = f"  {rec['city']} {rec['state'].upper()} {rec['year']} {rec['expense']} (${expected_int:,})"
        n_chunks = len(rec["chunks"])

        if idx >= 0:
            hits.append((idx, info, n_chunks))
        else:
            misses.append((info, n_chunks))

    total = len(records)
    print(f"\nParser: {parser} ({total} records, up to {max(len(r['chunks']) for r in records)} chunks cached)")
    print("-" * 50)

    for k in K_VALUES:
        found = sum(1 for hit_idx, _, _ in hits if hit_idx < k)
        print(f"  k={k:<3}  {found:>3}/{total}  ({100*found/total:.0f}%)")

    if misses:
        print(f"\nNot found in cached chunks ({len(misses)}):")
        for info, n in misses:
            print(f"  {info}  [{n} chunks searched]")
    else:
        print(f"\nAll records found!")

    # Show distribution of first-hit positions
    if hits:
        print(f"\nFirst-hit position distribution:")
        buckets = {}
        for idx, _, _ in hits:
            bucket = idx  # 0-based
            buckets[bucket] = buckets.get(bucket, 0) + 1
        for pos in sorted(buckets):
            print(f"  chunk {pos+1}: {buckets[pos]} records")


def main():
    parser = argparse.ArgumentParser(description="Retrieval recall@k sweep")
    parser.add_argument("--parser", nargs="*", default=["aryn", "pymupdf"],
                        help="Parsers to sweep (default: aryn pymupdf)")
    args = parser.parse_args()

    for p in args.parser:
        run_sweep(p)
        print()


if __name__ == "__main__":
    main()
