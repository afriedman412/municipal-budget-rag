#!/usr/bin/env python3
"""Find which PDF is causing the segfault by testing each one directly."""

import fitz
from pathlib import Path

PDF_FOLDER = "/Users/user/Documents/code/sycamore_scrap/municipal-budget-rag/pdfs_2026/selects"

def test_pdf(pdf_path: Path) -> bool:
    """Test a single PDF. Returns True if OK."""
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
        doc.close()
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    pdf_folder = Path(PDF_FOLDER)
    pdf_files = sorted(pdf_folder.glob("*.pdf"))

    print(f"Testing {len(pdf_files)} PDFs (will crash on bad PDF)...\n")

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Testing {pdf_path.name}...", flush=True)
        if not test_pdf(pdf_path):
            print(f"  ^ FAILED (but didn't crash)")

    print("\nAll PDFs passed without crashing!")

if __name__ == "__main__":
    main()
