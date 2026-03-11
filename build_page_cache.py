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
import logging
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _check_mupdf_warnings(pdf_path):
    """Log any MuPDF warnings that accumulated."""
    import fitz
    warnings = fitz.TOOLS.mupdf_warnings()
    if warnings:
        logger.warning("MuPDF [%s]: %s", Path(pdf_path).name, warnings)
        fitz.TOOLS.mupdf_warnings(reset=True)


def parse_page_pymupdf(pdf_path, page_idx):
    import fitz
    fitz.TOOLS.mupdf_warnings(reset=True)
    doc = fitz.open(str(pdf_path))
    try:
        if page_idx < 0 or page_idx >= len(doc):
            return None
        text = doc[page_idx].get_text()
        _check_mupdf_warnings(pdf_path)
        return re.sub(r'\n{3,}', '\n\n', text).strip()
    finally:
        doc.close()


_marker_converter = None


def _get_marker_converter():
    """Lazily initialize the Marker converter (caches across calls)."""
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
    return _marker_converter


def _extract_pages_to_temp_pdf(pdf_path, page_indices):
    """Use PyMuPDF to extract specific pages into a temp PDF. Returns (temp_path, page_map).
    page_map maps original page index -> index in temp PDF."""
    import fitz
    fitz.TOOLS.mupdf_warnings(reset=True)
    doc = fitz.open(str(pdf_path))
    try:
        # Filter to valid pages and deduplicate, preserving order
        valid = []
        seen = set()
        for idx in page_indices:
            if 0 <= idx < len(doc) and idx not in seen:
                valid.append(idx)
                seen.add(idx)

        if not valid:
            return None, {}

        new_doc = fitz.open()
        page_map = {}  # original_idx -> temp_idx
        for temp_idx, orig_idx in enumerate(valid):
            new_doc.insert_pdf(doc, from_page=orig_idx, to_page=orig_idx)
            page_map[orig_idx] = temp_idx

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        new_doc.save(tmp.name)
        new_doc.close()
        _check_mupdf_warnings(pdf_path)
        return tmp.name, page_map
    finally:
        doc.close()


def _parse_marker_pages(pdf_path, page_indices):
    """Parse specific pages with Marker by extracting them one at a time.
    Returns dict mapping original page index -> parsed text.

    We parse each page individually because Marker's page separators (---)
    are unreliable when processing multi-page PDFs — it may merge pages
    or skip separators, breaking the page mapping."""
    converter = _get_marker_converter()
    from marker.output import text_from_rendered

    result = {}
    for orig_idx in page_indices:
        tmp_path, page_map = _extract_pages_to_temp_pdf(
            pdf_path, [orig_idx],
        )
        if tmp_path is None:
            continue

        try:
            rendered = converter(tmp_path)
            full_text, _, _ = text_from_rendered(rendered)
            text = full_text.strip()
            if text:
                result[orig_idx] = text
        except Exception as e:
            logger.warning(
                "Marker failed on %s page %d: %s",
                Path(pdf_path).name, orig_idx, e,
            )
        finally:
            os.unlink(tmp_path)

    return result


# Cache parsed pages per PDF (key: pdf_path, value: {page_idx: text})
_pdf_cache = {}


def parse_pages(pdf_path, page_indices, parser):
    """Parse multiple pages from a PDF. Returns dict of {page_idx: text}.
    For Marker, extracts only needed pages into a temp PDF first."""
    if parser == "pymupdf":
        result = {}
        for idx in page_indices:
            text = parse_page_pymupdf(pdf_path, idx)
            if text:
                result[idx] = text
        return result
    elif parser == "marker":
        key = str(pdf_path)
        if key not in _pdf_cache:
            _pdf_cache[key] = {}
        # Find pages we haven't cached yet
        needed = [idx for idx in page_indices if idx not in _pdf_cache[key]]
        if needed:
            parsed = _parse_marker_pages(pdf_path, needed)
            _pdf_cache[key].update(parsed)
        return {idx: _pdf_cache[key][idx] for idx in page_indices if idx in _pdf_cache[key]}
    else:
        from pipeline.parsers import parse_page_text
        result = {}
        for idx in page_indices:
            text = parse_page_text(pdf_path, idx, parser=parser)
            if text:
                result[idx] = text
        return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

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
            # Compute all needed page indices upfront
            needed_pages = [target_page]
            for offset in range(1, args.context_pages + 1):
                for page_idx in [target_page - offset, target_page + offset]:
                    if page_idx >= 0:
                        needed_pages.append(page_idx)

            # Parse all needed pages in one batch
            parsed = parse_pages(pdf_path, needed_pages, args.parser)

            target_text = parsed.get(target_page)
            if not target_text:
                errors += 1
                continue

            # Build chunks: target page first, then context pages
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

            for page_idx in needed_pages[1:]:
                text = parsed.get(page_idx)
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
