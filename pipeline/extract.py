"""PDF text extraction with metadata parsing."""

import re
from dataclasses import dataclass, asdict
from multiprocessing import Process, Queue
from queue import Empty
from pathlib import Path

import fitz  # pymupdf


@dataclass
class ExtractedPage:
    page_num: int
    text: str


@dataclass
class ExtractedDocument:
    s3_key: str
    filename: str
    pages: list[ExtractedPage]
    # Parsed metadata
    city: str
    state: str
    year: int

    @property
    def page_count(self) -> int:
        return len(self.pages)


def extract_pdf(pdf_path: Path, s3_key: str) -> ExtractedDocument:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Local path to PDF file
        s3_key: Original S3 key (for metadata parsing)

    Returns:
        ExtractedDocument with pages and metadata
    """
    doc = fitz.open(str(pdf_path))
    pages = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            # Clean up whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()

            if text:  # Skip empty pages
                pages.append(ExtractedPage(
                    page_num=page_num + 1,
                    text=text
                ))
    finally:
        doc.close()

    # Parse metadata from filename
    filename = Path(s3_key).name
    metadata = parse_filename(filename)

    return ExtractedDocument(
        s3_key=s3_key,
        filename=filename,
        pages=pages,
        city=metadata["city"],
        state=metadata["state"],
        year=metadata["year"],
    )


def parse_filename(filename: str) -> dict:
    """
    Parse metadata from filename.

    Expected format: SS_city_name_YY[_budget_type].pdf
        - SS: two-letter state abbreviation
        - city_name: city name with underscores (e.g., san_antonio)
        - YY: two-digit year
        - budget_type: optional suffix (e.g., proposed, capital, operating)

    Examples:
        az_phoenix_23.pdf -> {"state": "AZ", "city": "Phoenix", "year": 2023}
        tx_san_antonio_24_proposed.pdf -> {"state": "TX", "city": "San Antonio", "year": 2024, "budget_type": "proposed"}
        ca_los_angeles_22_capital.pdf -> {"state": "CA", "city": "Los Angeles", "year": 2022, "budget_type": "capital"}
    """
    stem = Path(filename).stem.lower()
    parts = stem.split("_")

    metadata = {"city": "Unknown", "state": "XX", "year": 0, "budget_type": None}

    if len(parts) < 3:
        return metadata

    # First part: state (must be 2 letters)
    if len(parts[0]) == 2 and parts[0].isalpha():
        metadata["state"] = parts[0].upper()
    else:
        return metadata

    # Find the year (two-digit number) - scan from position 2 onwards
    year_idx = None
    for i in range(2, len(parts)):
        if len(parts[i]) == 2 and parts[i].isdigit():
            year_val = int(parts[i])
            # Reasonable year range: 00-50 -> 2000-2050, 51-99 -> 1951-1999
            metadata["year"] = 2000 + year_val if year_val < 51 else 1900 + year_val
            year_idx = i
            break

    if year_idx is None:
        # No year found, treat everything after state as city
        metadata["city"] = " ".join(parts[1:]).title()
        return metadata

    # City: everything between state and year
    city_parts = parts[1:year_idx]
    metadata["city"] = " ".join(city_parts).title()

    # Budget type: everything after year (if any)
    if year_idx < len(parts) - 1:
        budget_type_parts = parts[year_idx + 1:]
        metadata["budget_type"] = " ".join(budget_type_parts).title()

    return metadata


def chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    """
    Split text into chunks that fit within character limit.
    Tries to break at paragraph boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for \n\n

        if current_len + para_len > max_chars and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


@dataclass
class ExtractResult:
    """Result of extraction attempt."""
    success: bool
    document: ExtractedDocument | None
    error: str | None


def _extract_worker(pdf_path_str: str, s3_key: str, result_queue: Queue):
    """Worker function that runs in subprocess. Crashes here don't kill parent."""
    try:
        doc = extract_pdf(Path(pdf_path_str), s3_key)
        # Serialize the document for queue transfer
        result_queue.put({
            "success": True,
            "document": {
                "s3_key": doc.s3_key,
                "filename": doc.filename,
                "pages": [{"page_num": p.page_num, "text": p.text} for p in doc.pages],
                "city": doc.city,
                "state": doc.state,
                "year": doc.year,
            },
            "error": None,
        })
    except Exception as e:
        result_queue.put({
            "success": False,
            "document": None,
            "error": f"{type(e).__name__}: {str(e)}",
        })


def extract_pdf_safe(pdf_path: Path, s3_key: str, timeout: int = 300) -> ExtractResult:
    """
    Extract PDF in a subprocess so segfaults don't kill the main process.

    Args:
        pdf_path: Local path to PDF file
        s3_key: Original S3 key
        timeout: Timeout in seconds (default 5 minutes for large PDFs)

    Returns:
        ExtractResult with success status and document or error
    """
    result_queue = Queue()

    proc = Process(target=_extract_worker, args=(str(pdf_path), s3_key, result_queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        # Timed out
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        return ExtractResult(
            success=False,
            document=None,
            error="Extraction timeout - PDF may be corrupted or very large"
        )

    if proc.exitcode != 0:
        # Crashed (segfault, etc.)
        return ExtractResult(
            success=False,
            document=None,
            error=f"Extraction crashed (exit code {proc.exitcode}) - PDF causes segfault"
        )

    # Get result from queue
    try:
        result_dict = result_queue.get_nowait()
        if result_dict["success"]:
            # Reconstruct the document from serialized form
            doc_data = result_dict["document"]
            pages = [ExtractedPage(page_num=p["page_num"], text=p["text"])
                     for p in doc_data["pages"]]
            document = ExtractedDocument(
                s3_key=doc_data["s3_key"],
                filename=doc_data["filename"],
                pages=pages,
                city=doc_data["city"],
                state=doc_data["state"],
                year=doc_data["year"],
            )
            return ExtractResult(success=True, document=document, error=None)
        else:
            return ExtractResult(
                success=False,
                document=None,
                error=result_dict["error"]
            )
    except Empty:
        return ExtractResult(
            success=False,
            document=None,
            error="No result from extraction worker - unknown error"
        )
