"""Test Docling parser on test set PDFs.
Parses each PDF, checks if expected budget numbers appear in extracted text.
Compare results to Aryn (10/12) and PyMuPDF (7/12).

Usage:
  python test_docling.py              # Parse all 6 test PDFs
  python test_docling.py --dump       # Also save full extracted text to docling_output/
"""

import argparse
import json
import os
import time

from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def find_pdf(city, state, year, pdf_dir="pdfs_2026"):
    """Find PDF matching city/state/year."""
    prefix = f"{state}_{city}_{str(year)[-2:]}"
    for f in os.listdir(pdf_dir):
        if f.startswith(prefix) and f.endswith(".pdf"):
            return os.path.join(pdf_dir, f)
    return None


def check_number_in_text(text, expected):
    """Check if expected budget number appears in text (formatted or plain)."""
    expected_int = int(expected)
    formatted = f"{expected_int:,}"
    plain = str(expected_int)
    return formatted in text or plain in text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", action="store_true",
                        help="Save full extracted text to docling_output/")
    args = parser.parse_args()

    with open("test_budgets.json") as f:
        test_set = json.load(f)

    with open("gold_validation_set.json") as f:
        gold = json.load(f)

    # Build lookup: (state, city, year, expense) -> budget
    gold_lookup = {}
    for g in gold:
        key = (g["state"].lower(), g["city"].lower().replace(" ", "_"), g["year"], g["expense"])
        gold_lookup[key] = g["budget"]

    accel = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CUDA)
    pdf_options = PdfPipelineOptions(do_ocr=False, accelerator_options=accel)
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pdf_options)}
    )

    if args.dump:
        os.makedirs("docling_output", exist_ok=True)

    print(f"{'State':<6} {'City':<20} {'Year':<6} {'Expense':<15} {'Expected':<20} {'Found?':<8} {'Time'}")
    print("-" * 90)

    found_count = 0
    total = 0

    for t in test_set:
        city, state, year = t["city"], t["state"], t["year"]
        pdf_path = find_pdf(city, state, year)
        if not pdf_path:
            print(f"{state.upper():<6} {city:<20} {year:<6} {'--':<15} {'NO PDF':<20} {'--':<8}")
            continue

        # Parse once per PDF
        print(f"  Parsing {pdf_path}...", end=" ", flush=True)
        start = time.time()
        result = converter.convert(pdf_path)
        elapsed = time.time() - start
        print(f"{elapsed:.0f}s")

        doc = result.document
        full_text = doc.export_to_markdown()

        if args.dump:
            out_name = f"docling_output/{state}_{city}_{year}.md"
            with open(out_name, "w") as f:
                f.write(full_text)

        # Check both expense types
        for expense in ["General Fund", "Police"]:
            key = (state, city, year, expense)
            if key not in gold_lookup:
                continue

            expected = gold_lookup[key]
            expected_str = f"${expected:,.0f}"
            found = check_number_in_text(full_text, expected)
            if found:
                found_count += 1
            total += 1
            flag = "Y" if found else "N"

            print(f"{state.upper():<6} {city:<20} {year:<6} {expense:<15} {expected_str:<20} {flag:<8} {elapsed:.0f}s")

    print("-" * 90)
    if total:
        print(f"Found in text: {found_count}/{total} ({100*found_count/total:.0f}%)")
        print(f"\nBaseline comparison:")
        print(f"  Aryn:    10/12 (83%)")
        print(f"  PyMuPDF:  7/12 (58%)")
        print(f"  Docling: {found_count}/{total} ({100*found_count/total:.0f}%)")


if __name__ == "__main__":
    main()
