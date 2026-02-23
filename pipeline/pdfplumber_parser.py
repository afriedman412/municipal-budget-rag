"""pdfplumber-based PDF extraction with table-aware parsing."""

import asyncio
from pathlib import Path

import pdfplumber

from .config import Config
from .parsers import ParsedDocument, parse_filename, _pdfplumber_page_text


def _extract_text(path: Path) -> tuple[str, list[str]]:
    """Extract all text from a PDF using pdfplumber.

    Tables are detected with text-based strategy and rendered as markdown.
    Returns (combined_text, per_page_texts).
    """
    with pdfplumber.open(str(path)) as pdf:
        page_texts = []
        for page in pdf.pages:
            text = _pdfplumber_page_text(page)
            page_texts.append(text)
        combined = "\n\n".join(t for t in page_texts if t)
        return combined, page_texts


class PdfPlumberClient:
    def __init__(self, config: Config):
        self.config = config

    async def parse(
        self, s3_key: str, local_path: Path
    ) -> ParsedDocument:
        """Parse a PDF with pdfplumber (runs in thread to avoid blocking)."""
        text, pages = await asyncio.to_thread(_extract_text, local_path)
        state, city, year = parse_filename(local_path.name)
        return ParsedDocument(
            s3_key=s3_key,
            filename=local_path.name,
            elements=[],
            text=text,
            state=state,
            city=city,
            year=year,
            pages=pages,
        )

    async def parse_batch(
        self, items: list[tuple[str, Path]]
    ) -> list[tuple[ParsedDocument | None, str | None]]:
        """Parse multiple PDFs concurrently."""
        async def parse_one(s3_key, path):
            try:
                doc = await self.parse(s3_key, path)
                return (doc, None)
            except Exception as e:
                return (None, f"{type(e).__name__}: {e}")

        return await asyncio.gather(
            *[parse_one(key, path) for key, path in items]
        )
