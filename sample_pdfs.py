"""Sample PDFs from S3 that have ground truth in the validation table.

Randomly selects N city/year combos from validation records that have pdf_page,
finds matching PDFs in S3, and downloads them locally.

Usage:
  python sample_pdfs.py --n 100                    # Download 100 random PDFs
  python sample_pdfs.py --n 50 --seed 99           # Reproducible sample
  python sample_pdfs.py --n 100 --dry-run          # List only
"""

import argparse
import random
import sqlite3

import boto3

from config import PDF_DIR

S3_BUCKET = "budget-pdf-depot"
S3_REGION = "us-east-2"


def find_s3_key(state, city, year, s3_files):
    """Find matching S3 key for a city/year."""
    prefix = (
        f"{state.lower()}_"
        f"{city.lower().replace(' ', '_')}_"
        f"{str(year)[2:]}"
    )
    matches = [k for k in s3_files if k.startswith(prefix)]
    return matches[0] if matches else None


def list_s3_files():
    """List all PDFs in the S3 bucket."""
    s3 = boto3.client("s3", region_name=S3_REGION)
    files = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET):
        for obj in page.get("Contents", []):
            files.append(obj["Key"])
    return files


def download_file(key, local_path):
    """Download a single file from S3."""
    s3 = boto3.client("s3", region_name=S3_REGION)
    s3.download_file(S3_BUCKET, key, str(local_path))


def main():
    parser = argparse.ArgumentParser(
        description="Sample PDFs from S3 with ground truth",
    )
    parser.add_argument("--n", type=int, required=True,
                        help="Number of PDFs to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="List without downloading")
    parser.add_argument("--expense",
                        choices=["General Fund", "Police"],
                        help="Filter to specific expense type")
    args = parser.parse_args()

    random.seed(args.seed)
    PDF_DIR.mkdir(exist_ok=True)

    # Get distinct city/year combos with ground truth
    conn = sqlite3.connect("pipeline_state.db")
    conn.row_factory = sqlite3.Row

    query = """
        SELECT DISTINCT state, city, year
        FROM validation
        WHERE pdf_page IS NOT NULL AND pdf_page > 0
    """
    if args.expense:
        query += f" AND expense = '{args.expense}'"

    combos = [dict(r) for r in conn.execute(query).fetchall()]
    print(f"Found {len(combos)} city/year combos with ground truth")

    if args.n > len(combos):
        print(f"Requested {args.n} but only "
              f"{len(combos)} available, using all")
        sample = combos
    else:
        sample = random.sample(combos, args.n)

    # List S3 files
    print("Listing S3 bucket...")
    s3_files = list_s3_files()
    print(f"Found {len(s3_files)} files in S3")

    # Match sample to S3 keys
    to_download = []
    not_found = []
    for combo in sample:
        key = find_s3_key(
            combo["state"], combo["city"],
            combo["year"], s3_files,
        )
        if key:
            to_download.append((combo, key))
        else:
            not_found.append(combo)

    print(f"\nMatched: {len(to_download)}, "
          f"not found in S3: {len(not_found)}")
    if not_found:
        for c in not_found[:5]:
            print(f"  Missing: {c['state']} {c['city']} {c['year']}")
        if len(not_found) > 5:
            print(f"  ... and {len(not_found) - 5} more")

    if args.dry_run:
        print(f"\nWould download {len(to_download)} PDFs:")
        for combo, key in sorted(to_download, key=lambda x: x[1]):
            print(f"  {key}")
        return

    # Download
    print(f"\nDownloading {len(to_download)} PDFs...")
    downloaded = 0
    errors = 0
    for i, (combo, key) in enumerate(to_download):
        local_path = PDF_DIR / key
        if local_path.exists():
            downloaded += 1
            continue
        try:
            download_file(key, local_path)
            downloaded += 1
        except Exception as e:
            errors += 1
            print(f"  Error: {key}: {e}")

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(to_download)} done...")

    print(f"\nDone: {downloaded} downloaded, {errors} errors")


if __name__ == "__main__":
    main()
