"""Build a gold chunk cache from PDF pages directly.

No ChromaDB, no Aryn, no OpenAI, no embeddings needed.
Parses PDFs page-by-page, builds a cache compatible with test_llm_extraction.py.

Usage:
  python build_page_cache.py --parser marker
  python build_page_cache.py --parser pymupdf
  python build_page_cache.py --parser marker --test-only
"""

import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path


PDF_DIR = Path("pdfs_2026")
TRAINING_DIR = Path("training")


def find_pdf(state, city, year):
    state = state.lower()
    city = city.lower().replace(" ", "_")
    yr = str(year)[2:]
    for pdf in PDF_DIR.glob("*.pdf"):
        if pdf.name.startswith(f"{state}_{city}_{yr}"):
            return pdf
    return None


def parse_page_pymupdf(pdf_path, page_idx):
    import fitz
    doc = fitz.open(str(pdf_path))
    try:
        if page_idx < 0 or page_idx >= len(doc):
            return None
        text = doc[page_idx].get_text()
        return re.sub(r'\n{3,}', '\n\n', text).strip()
    finally:
        doc.close()


_marker_converter = None


def parse_page_marker(pdf_path, page_idx):
    """Parse a single page with Marker. Caches the converter across calls."""
    global _marker_converter
    if _marker_converter is None:
        from marker.converters.pdf import PdfConverter
        from marker.config.parser import ConfigParser
        from marker.models import create_model_dict
        config_parser = ConfigParser({"output_format": "markdown"})
        _marker_converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

    from marker.output import text_from_rendered
    rendered = _marker_converter(str(pdf_path))
    full_text, _, _ = text_from_rendered(rendered)

    # Split on page breaks
    if "\n\n---\n\n" in full_text:
        pages = full_text.split("\n\n---\n\n")
    else:
        pages = [full_text]

    if page_idx < 0 or page_idx >= len(pages):
        return None
    return pages[page_idx].strip() or None


# Cache entire PDF parse results to avoid re-parsing for each page
_pdf_cache = {}


def parse_page(pdf_path, page_idx, parser):
    """Parse a single page, with per-PDF caching for Marker."""
    if parser == "pymupdf":
        return parse_page_pymupdf(pdf_path, page_idx)
    elif parser == "marker":
        key = str(pdf_path)
        if key not in _pdf_cache:
            # Parse entire PDF once, cache all pages
            _pdf_cache[key] = _parse_all_pages_marker(pdf_path)
        pages = _pdf_cache[key]
        if page_idx < 0 or page_idx >= len(pages):
            return None
        return pages[page_idx]
    else:
        # Fallback to pipeline parser (needs pipeline deps)
        from pipeline.parsers import parse_page_text
        return parse_page_text(pdf_path, page_idx, parser=parser)


def _parse_all_pages_marker(pdf_path):
    """Parse all pages of a PDF with Marker. Returns list of page texts."""
    global _marker_converter
    if _marker_converter is None:
        from marker.converters.pdf import PdfConverter
        from marker.config.parser import ConfigParser
        from marker.models import create_model_dict
        config_parser = ConfigParser({"output_format": "markdown"})
        _marker_converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

    from marker.output import text_from_rendered
    rendered = _marker_converter(str(pdf_path))
    full_text, _, _ = text_from_rendered(rendered)

    if "\n\n---\n\n" in full_text:
        pages = [p.strip() for p in full_text.split("\n\n---\n\n")]
    else:
        pages = [full_text.strip()]
    return pages


def main():
    parser = argparse.ArgumentParser(
        description="Build gold chunk cache from PDF pages (no ChromaDB)",
    )
    parser.add_argument(
        "--parser", "-p", default="pymupdf",
        help="Parser to use (pymupdf, marker)",
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
    TRAINING_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect("pipeline_state.db")
    conn.row_factory = sqlite3.Row

    # Load test set keys if --test-only
    test_keys = None
    if args.test_only:
        test_file = TRAINING_DIR / "test_budgets.json"
        if test_file.exists():
            with open(test_file) as f:
                test_keys = {
                    (r["state"].lower(), r["city"].lower().replace(" ", "_"), r["year"])
                    for r in json.load(f)
                }
            print(f"Test set: {len(test_keys)} city/year combos")
        else:
            print(f"Warning: {test_file} not found, processing all records")

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

    for i, r in enumerate(rows):
        state, city, year = r["state"], r["city"], r["year"]
        expense = r["expense"]
        target_page = r["pdf_page"] - 1  # 0-indexed

        if test_keys is not None:
            key = (state.lower(), city.lower().replace(" ", "_"), year)
            if key not in test_keys:
                skipped_test += 1
                continue

        pdf_path = find_pdf(state, city, year)
        if not pdf_path:
            skipped_no_pdf += 1
            continue

        try:
            target_text = parse_page(pdf_path, target_page, args.parser)
            if not target_text:
                errors += 1
                continue

            # Build chunks: target page + surrounding context pages
            chunks = [{
                "text": target_text,
                "metadata": {
                    "filename": pdf_path.name,
                    "state": state.lower(),
                    "city": city.lower().replace(" ", "_"),
                    "year": year,
                    "page": target_page,
                    "parser": args.parser,
                    "has_table": "|" in target_text,
                },
            }]

            for offset in range(1, args.context_pages + 1):
                for page_idx in [target_page - offset, target_page + offset]:
                    if page_idx < 0:
                        continue
                    text = parse_page(pdf_path, page_idx, args.parser)
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

    # Clear PDF cache to free memory
    _pdf_cache.clear()

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
