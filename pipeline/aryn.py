"""Async Aryn document parsing."""

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from aryn_sdk.partition import (
    partition_file,
    partition_file_async_submit,
    partition_file_async_result,
)

from .config import Config


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


def _parse_filename(filename: str) -> tuple[str | None, str | None, int | None]:
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


def _table_to_markdown(el: dict) -> str | None:
    """Convert a structured table element to markdown."""
    table_data = el.get("table")
    if not table_data or not table_data.get("cells"):
        # Fall back to text_representation
        return el.get("text_representation", "").strip() or None

    cells = table_data["cells"]

    # Find table dimensions
    max_row = max(max(c["rows"]) for c in cells) + 1
    max_col = max(max(c["cols"]) for c in cells) + 1

    # Build grid
    grid = [["" for _ in range(max_col)] for _ in range(max_row)]
    header_rows = set()

    for cell in cells:
        content = cell.get("content", "").strip()
        for r in cell["rows"]:
            for c in cell["cols"]:
                grid[r][c] = content
        if cell.get("is_header"):
            header_rows.update(cell["rows"])

    # Render as markdown table
    lines = []
    for i, row in enumerate(grid):
        line = "| " + " | ".join(row) + " |"
        lines.append(line)
        # Add separator after header row(s)
        if i in header_rows and (i + 1 not in header_rows):
            sep = "| " + " | ".join("---" for _ in row) + " |"
            lines.append(sep)

    # If no header was detected, add separator after first row
    if not header_rows and len(lines) > 0:
        sep = "| " + " | ".join("---" for _ in grid[0]) + " |"
        lines.insert(1, sep)

    return "\n".join(lines)


def _elements_to_text(elements: list[dict]) -> str:
    """Convert Aryn elements to text for embedding.

    Tables are rendered as markdown tables to preserve structure.
    """
    texts = []
    for el in elements:
        el_type = el.get("type", "")

        # Skip headers/footers
        if el_type in ("Page-header", "Page-footer"):
            continue

        # Render tables as markdown
        if el_type == "table":
            md = _table_to_markdown(el)
            if md:
                texts.append(md)
            continue

        text = el.get("text_representation", "").strip()
        if text:
            texts.append(text)

    return "\n\n".join(texts)


def _parse_sync(path: Path, api_key: str) -> dict:
    """Synchronous Aryn parsing (runs in thread)."""
    with open(path, "rb") as f:
        return partition_file(
            f,
            aryn_api_key=api_key,
            extract_table_structure=True,
        )


def _submit_async(path: Path, api_key: str) -> str:
    """Submit a PDF for async parsing. Returns task ID."""
    with open(path, "rb") as f:
        result = partition_file_async_submit(
            f,
            aryn_api_key=api_key,
            extract_table_structure=True,
        )
        # SDK returns {'task_id': 'aryn:t-...'}, extract the string
        if isinstance(result, dict):
            return result["task_id"]
        return result


def _poll_async(task_id: str, api_key: str):
    """Poll for async result. Returns data or None."""
    result = partition_file_async_result(
        task_id, aryn_api_key=api_key
    )
    if result.get("task_status") == "done":
        return result.get("result")
    return None


class ArynClient:
    def __init__(self, config: Config):
        self.config = config
        self.use_async = config.aryn_async

    async def parse(
        self, s3_key: str, local_path: Path
    ) -> ParsedDocument:
        """Parse a PDF with Aryn (sync API via thread)."""
        data = await asyncio.to_thread(
            _parse_sync, local_path, self.config.aryn_api_key
        )
        return self._to_document(s3_key, local_path, data)

    async def parse_batch(
        self, items: list[tuple[str, Path]]
    ) -> list[tuple[ParsedDocument | None, str | None]]:
        """Parse multiple PDFs.

        Uses async API (PAYG) if configured, otherwise
        sequential sync API (free tier).

        Returns: [(parsed_doc or None, error or None), ...]
        """
        if self.use_async:
            return await self._parse_batch_async(items)
        return await self._parse_batch_sequential(items)

    async def _parse_batch_sequential(
        self, items: list[tuple[str, Path]]
    ) -> list[tuple[ParsedDocument | None, str | None]]:
        """Sequential parsing for free tier."""
        results = []
        for s3_key, path in items:
            try:
                doc = await self.parse(s3_key, path)
                results.append((doc, None))
            except Exception as e:
                results.append((None, f"{type(e).__name__}: {e}"))
        return results

    async def _parse_batch_async(
        self, items: list[tuple[str, Path]]
    ) -> list[tuple[ParsedDocument | None, str | None]]:
        """Parallel parsing via Aryn async API (PAYG)."""
        api_key = self.config.aryn_api_key

        # Submit all jobs
        tasks = {}
        errors = {}
        for s3_key, path in items:
            try:
                task_id = await asyncio.to_thread(
                    _submit_async, path, api_key
                )
                tasks[s3_key] = (task_id, path)
            except Exception as e:
                errors[s3_key] = f"{type(e).__name__}: {e}"

        # Poll for results
        results_map = {}
        poll_interval = 5
        while tasks:
            await asyncio.sleep(poll_interval)
            done_keys = []
            for s3_key, (task_id, path) in tasks.items():
                try:
                    result = await asyncio.to_thread(
                        _poll_async, task_id, api_key
                    )
                    if result is not None:
                        doc = self._to_document(
                            s3_key, path, result
                        )
                        results_map[s3_key] = (doc, None)
                        done_keys.append(s3_key)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    results_map[s3_key] = (None, err)
                    done_keys.append(s3_key)
            for k in done_keys:
                del tasks[k]

        # Combine in original order
        results = []
        for s3_key, path in items:
            if s3_key in errors:
                results.append((None, errors[s3_key]))
            elif s3_key in results_map:
                results.append(results_map[s3_key])
            else:
                results.append((None, "Unknown error"))
        return results

    def _to_document(
        self, s3_key: str, path: Path, data: dict
    ) -> ParsedDocument:
        elements = data.get("elements", [])
        text = _elements_to_text(elements)
        st, city, year = _parse_filename(path.name)
        return ParsedDocument(
            s3_key=s3_key,
            filename=path.name,
            elements=elements,
            text=text,
            state=st,
            city=city,
            year=year,
        )
