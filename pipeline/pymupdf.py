"""PyMuPDF-based PDF text extraction."""

import asyncio
import re
from pathlib import Path

import fitz

from .aryn import ParsedDocument, _parse_filename
from .config import Config


def _extract_text(path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(str(path))
    try:
        texts = []
        for page in doc:
            text = page.get_text()
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()
            if text:
                texts.append(text)
        return "\n\n".join(texts)
    finally:
        doc.close()


class PyMuPDFClient:
    def __init__(self, config: Config):
        self.config = config

    async def parse(
        self, s3_key: str, local_path: Path
    ) -> ParsedDocument:
        """Parse a PDF with PyMuPDF (runs in thread to avoid blocking)."""
        text = await asyncio.to_thread(_extract_text, local_path)
        state, city, year = _parse_filename(local_path.name)
        return ParsedDocument(
            s3_key=s3_key,
            filename=local_path.name,
            elements=[],
            text=text,
            state=state,
            city=city,
            year=year,
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
