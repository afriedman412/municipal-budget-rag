"""Test a PDF parser on the focused parser test set.

Parses each PDF, checks if the expected budget number appears in the full text.
This tests parser quality independent of the LLM — if the number isn't in the
parsed text, no model can extract it.

Usage:
  python test_parser.py --parser marker pdfs_2026/
  python test_parser.py --parser pymupdf pdfs_2026/
  python test_parser.py --parser reducto pdfs_2026/
  python test_parser.py --parser llamaparse pdfs_2026/
  python test_parser.py --parser unstructured pdfs_2026/
  python test_parser.py --parser marker pdfs_2026/ --pages 50  # only first 50 pages

Env vars for API parsers:
  REDUCTO_API_KEY, LLAMA_CLOUD_API_KEY, UNSTRUCTURED_KEY, DOCSUMO_API_KEY
"""

import argparse
import json
import os
import sys
import time


def parse_marker(pdf_path, max_pages=None):
    """Parse PDF with Marker (local, GPU). pip install marker-pdf"""
    from marker.converters.pdf import PdfConverter
    from marker.config.parser import ConfigParser
    from marker.models import create_model_dict

    config_parser = ConfigParser({"output_format": "markdown"})
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )
    result = converter(str(pdf_path))

    pages = []
    for i, page in enumerate(result.children):
        if max_pages and i >= max_pages:
            break
        # Each page has children blocks — extract text from each
        parts = []
        for block in page.children:
            html = getattr(block, "html", "") or ""
            # Strip HTML tags, keep content
            import re
            text = re.sub(r"<[^>]+>", "", html)
            if text.strip():
                parts.append(text.strip())
        pages.append((i + 1, "\n".join(parts)))
    return pages


def parse_pymupdf(pdf_path, max_pages=None):
    """Parse PDF with PyMuPDF (local, free). pip install pymupdf"""
    import pymupdf

    pages = []
    doc = pymupdf.open(pdf_path)
    for i, page in enumerate(doc):
        if max_pages and i >= max_pages:
            break
        pages.append((i + 1, page.get_text()))
    doc.close()
    return pages


def parse_reducto(pdf_path, max_pages=None):
    """Parse PDF with Reducto API. pip install reductoai"""
    from pathlib import Path
    from reducto import Reducto

    client = Reducto()  # reads REDUCTO_API_KEY from env
    upload = client.upload(file=Path(pdf_path))
    result = client.parse.run(
        input=upload,
        options={"table_output_format": "markdown"},
    )

    pages = []
    for chunk in result.result.chunks:
        pg = chunk.blocks[0].bbox.page if chunk.blocks else 0
        pages.append((pg + 1, chunk.content))
    return pages


def parse_llamaparse(pdf_path, max_pages=None):
    """Parse PDF with LlamaParse API. pip install llama-parse"""
    from llama_parse import LlamaParse

    parser = LlamaParse(
        result_type="markdown",
        num_workers=1,
    )  # reads LLAMA_CLOUD_API_KEY from env
    documents = parser.load_data(pdf_path)

    # LlamaParse returns documents, not pages — concatenate
    pages = []
    for i, doc in enumerate(documents):
        pages.append((i + 1, doc.text))
    return pages


def parse_unstructured(pdf_path, max_pages=None):
    """Parse PDF with Unstructured API. pip install unstructured-client"""
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import shared

    client = UnstructuredClient(
        api_key_auth=os.environ["UNSTRUCTURED_KEY"],
    )
    with open(pdf_path, "rb") as f:
        response = client.general.partition(
            request=shared.PartitionParameters(
                files=shared.Files(
                    content=f.read(),
                    file_name=os.path.basename(pdf_path),
                ),
                strategy="hi_res",
            )
        )

    # Group elements by page
    page_texts = {}
    for el in response.elements:
        pg = el.metadata.page_number or 1
        page_texts.setdefault(pg, []).append(el.text or "")

    pages = [(pg, "\n".join(texts)) for pg, texts in sorted(page_texts.items())]
    return pages


def parse_docsumo(pdf_path, max_pages=None):
    """Parse PDF with Docsumo API. pip install requests"""
    import requests

    api_key = os.environ["DOCSUMO_API_KEY"]
    url = "https://app.docsumo.com/api/v1/eevee/apikey/upload/"
    headers = {"X-API-KEY": api_key}
    with open(pdf_path, "rb") as f:
        resp = requests.post(url, headers=headers,
                             files={"files": f},
                             data={"type": "auto"})
    resp.raise_for_status()
    data = resp.json()

    # Extract text from response
    pages = []
    if "data" in data and "pages" in data["data"]:
        for i, page in enumerate(data["data"]["pages"]):
            text = page.get("text", "")
            pages.append((i + 1, text))
    else:
        # Fallback: dump all text
        pages.append((1, json.dumps(data.get("data", {}))))
    return pages


PARSERS = {
    "marker": parse_marker,
    "pymupdf": parse_pymupdf,
    "reducto": parse_reducto,
    "llamaparse": parse_llamaparse,
    "unstructured": parse_unstructured,
    "docsumo": parse_docsumo,
}


def check_number_in_text(text, expected):
    """Check if expected number appears in text (comma-formatted or plain)."""
    formatted = f"{expected:,}"
    plain = str(expected)
    return formatted in text or plain in text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdf_dir", help="Directory containing PDFs")
    p.add_argument("--parser", required=True, choices=list(PARSERS.keys()))
    p.add_argument("--test-set", default="parser_test_set.json")
    p.add_argument("--pages", type=int, default=None,
                   help="Max pages to parse per PDF (saves time on large docs)")
    args = p.parse_args()

    test_set = json.load(open(args.test_set))
    parse_fn = PARSERS[args.parser]

    print(f"Parser: {args.parser}")
    print(f"Test set: {len(test_set)} records")
    if args.pages:
        print(f"Max pages: {args.pages}")
    print("=" * 70)

    found = 0
    total = 0

    for rec in test_set:
        pdf_path = os.path.join(args.pdf_dir, rec["pdf"])
        if not os.path.exists(pdf_path):
            print(f"  SKIP  {rec['pdf']} — not found")
            continue

        total += 1
        expected = int(rec["expected"])
        label = f"{rec['state']} {rec['city']} {rec['year']} {rec['expense']}"

        t0 = time.time()
        pages = parse_fn(pdf_path, max_pages=args.pages)
        elapsed = time.time() - t0

        # Check full text
        full_text = "\n".join(text for _, text in pages)
        in_text = check_number_in_text(full_text, expected)

        # Find which page(s) contain the number
        found_pages = []
        if in_text:
            for pg_num, text in pages:
                if check_number_in_text(text, expected):
                    found_pages.append(pg_num)

        if in_text:
            found += 1
            pg_str = ",".join(str(p) for p in found_pages[:3])
            flag = "FOUND"
        else:
            flag = "MISS"
            pg_str = "-"

        cat = rec.get("category", "")
        print(f"  {flag:<6} {label:<40} ${expected:>14,}  pg:{pg_str:<10} {elapsed:.1f}s  [{cat}]")

    print("=" * 70)
    print(f"Found: {found}/{total}  ({100*found/total:.0f}%)" if total else "No PDFs found.")


if __name__ == "__main__":
    main()
