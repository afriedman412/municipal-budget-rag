"""Embedding generation with batching and async support."""

import asyncio
import logging
from dataclasses import dataclass
import hashlib

from openai import AsyncOpenAI, RateLimitError, APITimeoutError

from .config import PipelineConfig
from .extract import ExtractedDocument, chunk_text

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of text ready for embedding."""
    chunk_id: str
    text: str
    s3_key: str
    filename: str
    page_num: int
    chunk_idx: int
    city: str
    state: str
    year: int


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding."""
    chunk: DocumentChunk
    embedding: list[float]


def create_chunk_id(filename: str, page: int, chunk_idx: int) -> str:
    """Create a deterministic unique ID for a chunk."""
    content = f"{filename}:p{page}:c{chunk_idx}"
    return hashlib.md5(content.encode()).hexdigest()


def document_to_chunks(doc: ExtractedDocument) -> list[DocumentChunk]:
    """Convert an extracted document to embeddable chunks."""
    chunks = []

    for page in doc.pages:
        text_chunks = chunk_text(page.text)

        for chunk_idx, text in enumerate(text_chunks):
            chunk_id = create_chunk_id(doc.filename, page.page_num, chunk_idx)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=text,
                s3_key=doc.s3_key,
                filename=doc.filename,
                page_num=page.page_num,
                chunk_idx=chunk_idx,
                city=doc.city,
                state=doc.state,
                year=doc.year,
            ))

    return chunks


class EmbeddingClient:
    """Async client for generating embeddings with batching."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = AsyncOpenAI()
        self.semaphore = asyncio.Semaphore(config.embed_concurrency)

    async def embed_batch(
        self,
        texts: list[str],
        max_retries: int = 5
    ) -> list[list[float]]:
        """Embed a batch of texts with rate limiting and retry."""
        # Truncate texts to avoid token limit (~8191 tokens for text-embedding-3-small)
        # Conservative: ~3 chars per token for budget docs with numbers
        truncated = [t[:24000] for t in texts]

        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    response = await self.client.embeddings.create(
                        model=self.config.embedding_model,
                        input=truncated
                    )
                    return [item.embedding for item in response.data]

                except RateLimitError as e:
                    wait_time = min(2 ** attempt * 1.0, 60.0)  # Exponential backoff, max 60s
                    logger.warning(f"Rate limited, waiting {wait_time:.1f}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)

                except APITimeoutError as e:
                    wait_time = 2 ** attempt * 0.5
                    logger.warning(f"Timeout, retrying in {wait_time:.1f}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)

            # Final attempt - let it raise
            response = await self.client.embeddings.create(
                model=self.config.embedding_model,
                input=truncated
            )
            return [item.embedding for item in response.data]

    async def embed_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int | None = None
    ) -> list[EmbeddedChunk]:
        """Embed all chunks with batching and concurrency control."""
        batch_size = batch_size or self.config.embed_batch_size
        results = []

        # Create batches
        batches = [
            chunks[i:i + batch_size]
            for i in range(0, len(chunks), batch_size)
        ]

        # Process batches concurrently (limited by semaphore)
        async def process_batch(batch: list[DocumentChunk]) -> list[EmbeddedChunk]:
            texts = [c.text for c in batch]
            embeddings = await self.embed_batch(texts)
            return [
                EmbeddedChunk(chunk=chunk, embedding=emb)
                for chunk, emb in zip(batch, embeddings)
            ]

        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                raise result
            results.extend(result)

        return results
