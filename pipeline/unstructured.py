"""Unstructured API-based PDF parsing.

Requires: pip install unstructured-client
Env var: UNSTRUCTURED_KEY
"""

import asyncio
import os
from pathlib import Path

from .config import Config
from .parsers import ParsedDocument, parse_filename


def _parse_pdf(path: Path) -> tuple[str, list[str]]:
    """Parse PDF with Unstructured API (sync, runs in thread).

    Returns (combined_text, per_page_texts).
    """
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import operations, shared

    client = UnstructuredClient(
        api_key_auth=os.environ["UNSTRUCTURED_KEY"],
    )
    with open(path, "rb") as f:
        req = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(
                    content=f.read(),
                    file_name=path.name,
                ),
                strategy="hi_res",
                split_pdf_page=True,
                split_pdf_allow_failed=True,
            )
        )
        response = client.general.partition(request=req)

    # Group elements by page (SDK returns dicts or objects depending on version)
    page_map: dict[int, list[str]] = {}
    for el in response.elements:
        if isinstance(el, dict):
            pg = el.get("metadata", {}).get("page_number") or 1
            text = el.get("text") or ""
        else:
            pg = el.metadata.page_number or 1
            text = el.text or ""
        if text.strip():
            page_map.setdefault(pg, []).append(text)

    # Build per-page list (0-indexed)
    if not page_map:
        return "", []

    max_page = max(page_map.keys())
    page_texts = []
    for i in range(1, max_page + 1):
        page_texts.append("\n".join(page_map.get(i, [])))

    combined = "\n\n".join(t for t in page_texts if t)
    return combined, page_texts


class UnstructuredClient_:
    def __init__(self, config: Config):
        self.config = config
        if not os.environ.get("UNSTRUCTURED_KEY"):
            raise ValueError("UNSTRUCTURED_KEY environment variable required")

    async def parse(
        self, s3_key: str, local_path: Path
    ) -> ParsedDocument:
        """Parse a PDF with Unstructured (runs in thread)."""
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
