"""Marker-based PDF parsing (local, GPU-accelerated)."""

import asyncio
import re
from pathlib import Path

from .config import Config
from .parsers import ParsedDocument, parse_filename


def _parse_pdf(path: Path) -> tuple[str, list[str]]:
    """Parse a PDF with Marker. Returns (combined_text, per_page_texts)."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser

    config = {"output_format": "json"}
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )

    rendered = converter(str(path))

    # JSON output is a list of pages, each with children blocks containing html
    page_texts = []
    for page in rendered.children:
        parts = []
        for block in page.children:
            html = getattr(block, "html", "") or ""
            # Convert HTML tables to markdown-ish text, strip other tags
            text = _html_to_text(html)
            if text.strip():
                parts.append(text.strip())
        page_texts.append("\n\n".join(parts))

    combined = "\n\n".join(t for t in page_texts if t)
    return combined, page_texts


def _html_to_text(html: str) -> str:
    """Simple HTML to text conversion, preserving table structure."""
    # Handle tables: convert to markdown
    if "<table" in html:
        return _html_table_to_markdown(html)
    # Strip HTML tags for regular text
    text = re.sub(r"<[^>]+>", "", html)
    return text.strip()


def _html_table_to_markdown(html: str) -> str:
    """Convert an HTML table to markdown format."""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    if not rows:
        return re.sub(r"<[^>]+>", "", html)

    md_lines = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        md_lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md_lines.append("| " + " | ".join("---" for _ in cells) + " |")

    return "\n".join(md_lines)


class MarkerClient:
    def __init__(self, config: Config):
        self.config = config

    async def parse(self, s3_key: str, local_path: Path) -> ParsedDocument:
        """Parse a PDF with Marker (runs in thread)."""
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
        """Parse multiple PDFs sequentially (Marker is GPU-heavy)."""
        results = []
        for s3_key, path in items:
            try:
                doc = await self.parse(s3_key, path)
                results.append((doc, None))
            except Exception as e:
                results.append((None, f"{type(e).__name__}: {e}"))
        return results
