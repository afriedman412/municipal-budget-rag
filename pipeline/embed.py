"""Async OpenAI embeddings."""

import asyncio
import hashlib
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from .config import Config
from .aryn import ParsedDocument


@dataclass
class Chunk:
    chunk_id: str
    text: str
    embedding: list[float] | None
    # Metadata
    s3_key: str
    filename: str
    state: str | None
    city: str | None
    year: int | None
    chunk_index: int
    # Content tags (set at chunking time)
    has_table: bool = False
    has_summary: bool = False
    parser: str = ""


def _chunk_text(text: str, max_chars: int = 4000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars

        # Try to break at paragraph or sentence
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = text.rfind(sep, start + max_chars // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


_SUMMARY_PATTERN = re.compile(
    r"TOTAL\s+(EXPENDITURE|EXPENSE|REVENUE|BUDGET)"
    r"|GENERAL\s+FUND"
    r"|ALL\s+FUNDS\s+SUMMARY",
    re.IGNORECASE,
)


def _tag_chunk(text: str) -> tuple[bool, bool]:
    """Detect if chunk has tables or summary data."""
    has_table = "| " in text and " | " in text
    has_summary = bool(_SUMMARY_PATTERN.search(text))
    return has_table, has_summary


def document_to_chunks(doc: ParsedDocument, parser: str = "") -> list[Chunk]:
    """Convert a parsed document to chunks for embedding."""
    text_chunks = _chunk_text(doc.text)
    chunks = []

    for i, text in enumerate(text_chunks):
        # Deterministic ID from content
        content_hash = hashlib.md5(
            f"{doc.s3_key}:{i}:{text[:100]}".encode()
        ).hexdigest()[:12]
        chunk_id = f"{doc.filename}_{i}_{content_hash}"
        has_table, has_summary = _tag_chunk(text)

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=text,
            embedding=None,
            s3_key=doc.s3_key,
            filename=doc.filename,
            state=doc.state,
            city=doc.city,
            year=doc.year,
            chunk_index=i,
            has_table=has_table,
            has_summary=has_summary,
            parser=parser,
        ))

    return chunks


class EmbeddingClient:
    def __init__(self, config: Config):
        self.config = config
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.embed_model
        self.dimensions = config.embed_dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []

        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]

    async def embed_chunks(self, chunks: list[Chunk], batch_size: int = 50) -> list[Chunk]:
        """Embed all chunks, batching API calls."""
        if not chunks:
            return []

        # Process in batches
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text[:2500] for c in batch]  # truncate to stay under 8192 token limit
            embeddings = await self.embed_texts(texts)
            all_embeddings.extend(embeddings)

        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk.embedding = embedding

        return chunks

    async def embed_document(self, doc: ParsedDocument) -> list[Chunk]:
        """Convert document to chunks and embed them."""
        chunks = document_to_chunks(doc)
        return await self.embed_chunks(chunks)
