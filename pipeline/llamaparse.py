"""LlamaParse API-based PDF parsing.

Requires: pip install llama-parse
Env var: LLAMA_CLOUD_API_KEY
"""

import asyncio
import os
from pathlib import Path

from .config import Config
from .parsers import ParsedDocument, parse_filename


def _parse_pdf(path: Path) -> tuple[str, list[str]]:
    """Parse PDF with LlamaParse (sync, runs in thread).

    Returns (combined_text, per_page_texts).
    """
    from llama_parse import LlamaParse

    parser = LlamaParse(
        result_type="markdown",
        num_workers=1,
    )  # reads LLAMA_CLOUD_API_KEY from env
    documents = parser.load_data(str(path))

    page_texts = [doc.text for doc in documents]
    combined = "\n\n".join(t for t in page_texts if t)
    return combined, page_texts


class LlamaParseClient:
    def __init__(self, config: Config):
        self.config = config
        if not os.environ.get("LLAMA_CLOUD_API_KEY"):
            raise ValueError("LLAMA_CLOUD_API_KEY environment variable required")

    async def parse(
        self, s3_key: str, local_path: Path
    ) -> ParsedDocument:
        """Parse a PDF with LlamaParse (runs in thread)."""
        text, pages = await asyncio.to_thread(_parse_pdf, local_path)
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
        """Parse multiple PDFs sequentially (API rate limits)."""
        results = []
        for s3_key, path in items:
            try:
                doc = await self.parse(s3_key, path)
                results.append((doc, None))
            except Exception as e:
                results.append((None, f"{type(e).__name__}: {e}"))
        return results
