"""Shared parser types and factory."""

import random
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    s3_key: str
    filename: str
    elements: list[dict]
    text: str  # Combined text for embedding
    # Metadata from filename
    state: str | None
    city: str | None
    year: int | None
    # Per-page raw text (0-indexed by physical page number)
    pages: list[str] = field(default_factory=list)


def parse_filename(filename: str) -> tuple[str | None, str | None, int | None]:
    """Extract state, city, year from filename.

    Format: SS_city_name_YY[_suffix].pdf
    Example: ca_san_diego_22.pdf -> ('ca', 'san_diego', 22)
    """
    name = filename.lower().replace(".pdf", "")
    parts = name.split("_")

    if len(parts) < 3:
        return None, None, None

    state = parts[0] if len(parts[0]) == 2 else None

    # Find year (2-digit number 16-30)
    year = None
    year_idx = None
    for i, part in enumerate(parts[1:], 1):
        if re.match(r"^\d{2}$", part):
            y = int(part)
            if 16 <= y <= 30:
                year = 2000 + y
                year_idx = i
                break

    # City is everything between state and year
    city = "_".join(parts[1:year_idx]) if year_idx and year_idx > 1 else None

    return state, city, year


def parse_page_text(pdf_path, page_idx, parser="pymupdf"):
    """Extract text from a single PDF page using the specified parser.

    Args:
        pdf_path: Path to PDF file
        page_idx: 0-based page index
        parser: Parser to use ('pymupdf', 'pdfplumber', or 'aryn')

    Returns:
        Extracted text string, or None if page is out of range.
    """
    if parser == "marker":
        return _marker_page_text(pdf_path, page_idx)
    elif parser == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_idx < 0 or page_idx >= len(pdf.pages):
                return None
            return _pdfplumber_page_text(pdf.pages[page_idx])
    elif parser == "aryn":
        return _aryn_page_text(pdf_path, page_idx)
    else:
        import fitz
        doc = fitz.open(str(pdf_path))
        try:
            if page_idx < 0 or page_idx >= len(doc):
                return None
            text = doc[page_idx].get_text()
            return re.sub(r'\n{3,}', '\n\n', text).strip()
        finally:
            doc.close()


def _marker_page_text(pdf_path, page_idx):
    """Extract text from a single PDF page using Marker."""
    from .marker_parser import _parse_pdf
    _, pages = _parse_pdf(Path(pdf_path))
    if page_idx < 0 or page_idx >= len(pages):
        return None
    return pages[page_idx].strip() or None


def _aryn_page_text(pdf_path, page_idx, max_retries=5):
    """Extract text from a single PDF page using Aryn DocParse.

    Uses selected_pages to parse only the requested page.
    Reads ARYN_API_KEY from environment.
    Retries on 429 (rate limit) with exponential backoff.
    """
    import os
    import time
    from aryn_sdk.partition import partition_file

    api_key = os.environ.get("ARYN_API_KEY")
    if not api_key:
        raise ValueError("ARYN_API_KEY environment variable required for aryn parser")

    for attempt in range(max_retries):
        try:
            with open(str(pdf_path), "rb") as f:
                data = partition_file(
                    f,
                    aryn_api_key=api_key,
                    extract_table_structure=True,
                    selected_pages=[[page_idx + 1]],  # Aryn uses 1-based page numbers
                )
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt + random.random()
                time.sleep(wait)
                continue
            raise

    # Convert elements to text using aryn.py's logic (lazy import to avoid circular)
    from .aryn import _elements_to_text
    elements = data.get("elements", [])
    text = _elements_to_text(elements)
    return text.strip() if text else None


def _pdfplumber_page_text(page):
    """Extract text from a pdfplumber page, rendering tables as markdown."""
    table_settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
    }
    tables = page.find_tables(table_settings=table_settings)

    if not tables:
        text = page.extract_text() or ""
        return text.strip()

    # Clamp table bboxes to page bounds and extract text outside tables
    px0, py0, px1, py1 = page.bbox
    filtered_page = page
    for t in tables:
        bx0, by0, bx1, by1 = t.bbox
        clamped = (max(bx0, px0), max(by0, py0), min(bx1, px1), min(by1, py1))
        if clamped[0] < clamped[2] and clamped[1] < clamped[3]:
            filtered_page = filtered_page.outside_bbox(clamped)
    plain_text = filtered_page.extract_text() or ""

    # Render tables as markdown
    table_markdowns = []
    for table in tables:
        rows = table.extract()
        if not rows:
            continue
        md_lines = []
        for i, row in enumerate(rows):
            cells = [str(c or "").strip() for c in row]
            md_lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                md_lines.append("| " + " | ".join("---" for _ in cells) + " |")
        table_markdowns.append("\n".join(md_lines))

    # Combine: plain text + tables
    parts = [plain_text.strip()] if plain_text.strip() else []
    parts.extend(table_markdowns)
    return "\n\n".join(parts)


def get_parser(name: str, config):
    """Get a parser client by name."""
    if name == "aryn":
        from .aryn import ArynClient
        return ArynClient(config)
    elif name == "pymupdf":
        from .pymupdf import PyMuPDFClient
        return PyMuPDFClient(config)
    elif name == "pdfplumber":
        from .pdfplumber_parser import PdfPlumberClient
        return PdfPlumberClient(config)
    elif name == "llamaparse":
        from .llamaparse import LlamaParseClient
        return LlamaParseClient(config)
    elif name == "unstructured":
        from .unstructured import UnstructuredClient_
        return UnstructuredClient_(config)
    elif name == "marker":
        from .marker_parser import MarkerClient
        return MarkerClient(config)
    else:
        raise ValueError(f"Unknown parser: {name}. Use 'aryn', 'pymupdf', 'pdfplumber', 'llamaparse', 'unstructured', or 'marker'.")
