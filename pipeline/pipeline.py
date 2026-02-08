"""Async pipeline orchestrator: S3 → Aryn → OpenAI → ChromaDB."""

import asyncio
import logging

from .config import Config
from .s3 import S3Client
from .aryn import ArynClient
from .embed import EmbeddingClient, document_to_chunks
from .chroma import ChromaClient
from .state import StateDB

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.s3 = S3Client(config)
        self.aryn = ArynClient(config)
        self.embedder = EmbeddingClient(config)
        self.chroma = ChromaClient(config)
        self.state = StateDB(config.state_db)

    async def discover(self) -> int:
        """Discover PDFs in S3 and register as jobs."""
        logger.info(f"Listing s3://{self.config.s3_bucket}/{self.config.s3_prefix}")
        keys = await self.s3.list_pdfs()
        logger.info(f"Found {len(keys)} PDFs")

        added = self.state.register_jobs(keys)
        logger.info(f"Registered {added} new jobs")
        return added

    async def process_one(self, s3_key: str) -> bool:
        """Process a single PDF through the full pipeline."""
        self.state.mark_processing(s3_key)

        try:
            # Download
            local_path = await self.s3.download(s3_key)

            # Parse with Aryn
            doc = await self.aryn.parse(s3_key, local_path)

            # Embed
            chunks = await self.embedder.embed_document(doc)

            # Index
            if chunks:
                self.chroma.add_chunks(chunks)

            # Cleanup
            self.s3.cleanup(local_path)

            self.state.mark_done(s3_key)
            return True

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.error(f"Failed {s3_key}: {error}")
            self.state.mark_failed(s3_key, error)
            return False

    async def process_batch(self, keys: list[str]) -> tuple[int, int]:
        """Process a batch of PDFs concurrently.

        Returns: (success_count, fail_count)
        """
        # Mark all as processing
        for key in keys:
            self.state.mark_processing(key)

        # Download all
        download_results = await self.s3.download_batch(keys)

        # Filter successful downloads
        to_parse = []
        for s3_key, path, error in download_results:
            if error:
                self.state.mark_failed(s3_key, f"Download: {error}")
            else:
                to_parse.append((s3_key, path))

        if not to_parse:
            return 0, len(keys)

        # Parse all with Aryn
        parse_results = await self.aryn.parse_batch(to_parse)

        # Collect successful parses
        to_embed = []
        paths_to_cleanup = []
        for (s3_key, path), (doc, error) in zip(to_parse, parse_results):
            paths_to_cleanup.append(path)
            if error:
                self.state.mark_failed(s3_key, f"Parse: {error}")
            else:
                to_embed.append(doc)

        if not to_embed:
            for path in paths_to_cleanup:
                self.s3.cleanup(path)
            return 0, len(keys)

        # Embed all documents
        all_chunks = []
        for doc in to_embed:
            chunks = document_to_chunks(doc)
            all_chunks.extend(chunks)

        if all_chunks:
            try:
                await self.embedder.embed_chunks(all_chunks)
                self.chroma.add_chunks(all_chunks)
            except Exception as e:
                error = f"Embed/Index: {type(e).__name__}: {e}"
                for doc in to_embed:
                    self.state.mark_failed(doc.s3_key, error)
                for path in paths_to_cleanup:
                    self.s3.cleanup(path)
                return 0, len(keys)

        # Mark successful
        success = 0
        for doc in to_embed:
            self.state.mark_done(doc.s3_key)
            success += 1

        # Cleanup
        for path in paths_to_cleanup:
            self.s3.cleanup(path)

        failed = len(keys) - success
        return success, failed

    async def run(self, batch_size: int | None = None, limit: int | None = None):
        """Run the pipeline on all pending jobs."""
        batch_size = batch_size or self.config.batch_size

        # Discover new PDFs
        await self.discover()

        stats = self.state.get_stats()
        pending = stats.get("pending", 0)

        if pending == 0:
            logger.info("No pending jobs")
            return stats

        if limit:
            pending = min(pending, limit)

        logger.info(f"Processing {pending} documents in batches of {batch_size}")

        total_success = 0
        total_failed = 0
        processed = 0

        while processed < pending:
            # Get next batch (respect overall limit)
            remaining = pending - processed
            keys = self.state.get_pending(limit=min(batch_size, remaining))
            if not keys:
                break

            # Process batch
            success, failed = await self.process_batch(keys)
            total_success += success
            total_failed += failed
            processed += len(keys)

            logger.info(
                f"Progress: {processed}/{pending} "
                f"(+{success} done, +{failed} failed)"
            )

        logger.info(f"Complete: {total_success} succeeded, {total_failed} failed")
        return self.state.get_stats()
