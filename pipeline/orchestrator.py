"""
Main pipeline orchestrator.

Coordinates: S3 → Extract → Embed → ChromaDB
With parallel processing, fault tolerance, and progress tracking.
"""

import asyncio
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Iterator

from tqdm import tqdm

from .config import PipelineConfig
from .state import StateDB, JobStatus
from .s3 import S3Client
from .extract import extract_pdf, extract_pdf_safe, ExtractedDocument
from .embed import EmbeddingClient, document_to_chunks, EmbeddedChunk
from .chroma import ChromaClient

logger = logging.getLogger(__name__)


@dataclass
class ExtractedJob:
    """A successfully extracted document ready for embedding."""
    s3_key: str
    document: ExtractedDocument


def _extract_single_pdf(args: tuple) -> tuple[str, ExtractedDocument | None, str | None]:
    """
    Worker function for parallel PDF extraction.
    Uses subprocess isolation so segfaults don't kill the worker pool.
    Returns: (s3_key, extracted_doc or None, error_message or None)
    """
    s3_key, local_path, skip_pages = args
    result = extract_pdf_safe(Path(local_path), s3_key, skip_pages=skip_pages)
    if result.success:
        return (s3_key, result.document, None)
    else:
        return (s3_key, None, result.error)


class Pipeline:
    """
    Main pipeline for processing PDFs from S3 to ChromaDB.

    Architecture:
    - Producer thread: downloads + extracts PDFs in parallel (CPU-bound)
    - Consumer: embeds + indexes async (I/O-bound)
    - Queue connects them for pipeline parallelism
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state = StateDB(config.state_db_path)
        self.s3 = S3Client(config)
        self.chroma = ChromaClient(config)
        self.embedder = EmbeddingClient(config)

    def discover_pdfs(self) -> int:
        """Discover PDFs in S3 and register them as jobs."""
        logger.info(f"Listing PDFs in s3://{self.config.s3_bucket}/{self.config.s3_prefix}")
        keys = self.s3.list_pdfs()
        logger.info(f"Found {len(keys)} PDFs in S3")

        added = self.state.register_jobs(keys)
        logger.info(f"Registered {added} new jobs ({len(keys) - added} already tracked)")
        return added

    def _extraction_producer(
        self,
        queue: Queue,
        batch_size: int = 50
    ):
        """
        Producer: Download and extract PDFs, push to queue.
        Runs in a separate thread, uses ProcessPoolExecutor for CPU-bound extraction.
        """
        while True:
            pending = self.state.get_pending(JobStatus.EXTRACTING, limit=batch_size)
            if not pending:
                break

            # Mark all as extracting and build lookup for skip_pages
            job_lookup = {}
            for job in pending:
                self.state.update_status(job.s3_key, JobStatus.EXTRACTING)
                job_lookup[job.s3_key] = job

            # Parallel download
            s3_keys = [job.s3_key for job in pending]
            download_results = self.s3.download_batch(s3_keys)

            # Separate successes and failures
            work_items = []
            for s3_key, local_path, error in download_results:
                if error:
                    self.state.update_status(
                        s3_key,
                        JobStatus.FAILED,
                        error_message=f"Download failed: {error}",
                        stage_failed="download"
                    )
                else:
                    skip_pages = job_lookup[s3_key].skip_pages
                    work_items.append((s3_key, str(local_path), skip_pages))

            if not work_items:
                continue

            # Parallel extraction
            with ProcessPoolExecutor(max_workers=self.config.pdf_workers) as executor:
                results = list(executor.map(_extract_single_pdf, work_items))

            for s3_key, doc, error in results:
                local_path = self.config.temp_dir / Path(s3_key).name

                if error:
                    self.state.update_status(
                        s3_key,
                        JobStatus.FAILED,
                        error_message=error,
                        stage_failed="extract"
                    )
                else:
                    # Update state and push to embedding queue
                    self.state.update_status(
                        s3_key,
                        JobStatus.EXTRACTED,
                        metadata={
                            "city": doc.city,
                            "state": doc.state,
                            "year": doc.year,
                            "page_count": doc.page_count,
                        }
                    )
                    queue.put(ExtractedJob(s3_key=s3_key, document=doc))

                # Cleanup local file
                self.s3.cleanup_local(local_path)

        # Signal done
        queue.put(None)

    async def _embedding_consumer(
        self,
        queue: Queue,
        batch_size: int = 10
    ) -> tuple[int, int]:
        """
        Consumer: Pull extracted docs from queue, embed and index.
        Batches multiple documents for efficient embedding.
        """
        success = 0
        failed = 0

        batch: list[ExtractedJob] = []

        while True:
            # Collect a batch (with timeout to avoid blocking forever)
            try:
                item = queue.get(timeout=1.0)
            except Empty:
                # Process partial batch if we have items and queue seems done
                if batch:
                    s, f = await self._embed_batch(batch)
                    success += s
                    failed += f
                    batch = []
                continue

            if item is None:
                # Producer done - process remaining batch
                if batch:
                    s, f = await self._embed_batch(batch)
                    success += s
                    failed += f
                break

            batch.append(item)

            if len(batch) >= batch_size:
                s, f = await self._embed_batch(batch)
                success += s
                failed += f
                batch = []

        return success, failed

    async def _embed_batch(self, jobs: list[ExtractedJob]) -> tuple[int, int]:
        """Embed and index a batch of extracted documents."""
        success = 0
        failed = 0

        # Collect all chunks from all documents
        all_chunks = []
        chunk_to_job = {}  # Track which job each chunk came from

        for job in jobs:
            self.state.update_status(job.s3_key, JobStatus.EMBEDDING)
            chunks = document_to_chunks(job.document)
            for chunk in chunks:
                chunk_to_job[chunk.chunk_id] = job.s3_key
            all_chunks.extend(chunks)

        if not all_chunks:
            # No text in any document
            for job in jobs:
                self.state.update_status(job.s3_key, JobStatus.DONE)
            return len(jobs), 0

        try:
            # Embed all chunks together (more efficient batching)
            embedded = await self.embedder.embed_chunks(all_chunks)

            # Index to ChromaDB
            self.chroma.add_chunks(embedded)

            # Mark all jobs done
            for job in jobs:
                self.state.update_status(job.s3_key, JobStatus.DONE)
                success += 1

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Batch embedding failed: {error_msg}")

            # Mark all jobs in batch as failed
            for job in jobs:
                self.state.update_status(
                    job.s3_key,
                    JobStatus.FAILED,
                    error_message=error_msg,
                    stage_failed="embed"
                )
                failed += 1

        return success, failed

    async def run(self, batch_size: int = 100):
        """
        Run the full pipeline with producer-consumer parallelism.

        - Producer thread: downloads + extracts PDFs (CPU-bound, parallel)
        - Consumer: embeds + indexes (I/O-bound, async batched)
        - Both run concurrently via queue
        """
        logger.info("Starting pipeline run")
        logger.info(f"Config: {self.config.pdf_workers} PDF workers, "
                    f"{self.config.embed_concurrency} embed concurrency")

        # Phase 1: Discover new PDFs
        self.discover_pdfs()

        stats = self.state.get_stats()
        pending_count = stats.get("pending", 0) + stats.get("extracted", 0)

        if pending_count == 0:
            logger.info("No pending work")
            return stats

        logger.info(f"Processing {pending_count} documents")

        # Queue for producer → consumer communication
        queue: Queue[ExtractedJob | None] = Queue(maxsize=batch_size * 2)

        # Start producer in background thread
        producer = Thread(
            target=self._extraction_producer,
            args=(queue, batch_size),
            daemon=True
        )
        producer.start()

        # Run consumer (async)
        with tqdm(total=pending_count, desc="Processing") as pbar:
            success = 0
            failed = 0

            batch: list[ExtractedJob] = []
            embed_batch_size = 10

            while True:
                try:
                    item = queue.get(timeout=0.5)
                except Empty:
                    if not producer.is_alive() and queue.empty():
                        break
                    continue

                if item is None:
                    break

                batch.append(item)
                # Show current file
                filename = item.s3_key.split('/')[-1]
                pbar.set_postfix_str(filename[:40])

                if len(batch) >= embed_batch_size:
                    s, f = await self._embed_batch(batch)
                    success += s
                    failed += f
                    pbar.update(len(batch))
                    batch = []

            # Final batch
            if batch:
                s, f = await self._embed_batch(batch)
                success += s
                failed += f
                pbar.update(len(batch))

        producer.join(timeout=5.0)

        # Summary
        stats = self.state.get_stats()
        logger.info("=" * 60)
        logger.info("Pipeline run complete")
        logger.info(f"  Processed: {success} succeeded, {failed} failed")
        logger.info(f"  State: {stats}")

        chroma_stats = self.chroma.get_stats()
        logger.info(f"  ChromaDB: {chroma_stats['total_chunks']} chunks, "
                    f"{chroma_stats['unique_documents']} documents")

        return stats

    async def run_simple(self, batch_size: int = 100):
        """
        Simpler sequential run (useful for debugging).
        Processes one document at a time: extract → embed → index.
        """
        logger.info("Starting simple sequential run")

        self.discover_pdfs()

        success = 0
        failed = 0

        while True:
            pending = self.state.get_pending(JobStatus.EXTRACTING, limit=batch_size)
            if not pending:
                break

            pbar = tqdm(pending, desc="Processing")
            for job in pbar:
                # Show current file
                filename = job.s3_key.split('/')[-1]
                pbar.set_postfix_str(filename[:40])

                try:
                    # Download
                    local_path = self.s3.download_pdf(job.s3_key)

                    # Extract (crash-safe, with page skipping if preflight identified bad pages)
                    self.state.update_status(job.s3_key, JobStatus.EXTRACTING)
                    result = extract_pdf_safe(local_path, job.s3_key, skip_pages=job.skip_pages)

                    if not result.success:
                        self.state.update_status(
                            job.s3_key,
                            JobStatus.FAILED,
                            error_message=result.error,
                            stage_failed="extract"
                        )
                        failed += 1
                        self.s3.cleanup_local(local_path)
                        continue

                    doc = result.document

                    # Embed
                    self.state.update_status(job.s3_key, JobStatus.EMBEDDING)
                    chunks = document_to_chunks(doc)

                    if chunks:
                        embedded = await self.embedder.embed_chunks(chunks)
                        self.chroma.add_chunks(embedded)

                    self.state.update_status(job.s3_key, JobStatus.DONE)
                    success += 1

                    # Cleanup
                    self.s3.cleanup_local(local_path)

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"Failed {job.s3_key}: {error_msg}")
                    self.state.update_status(
                        job.s3_key,
                        JobStatus.FAILED,
                        error_message=error_msg,
                        stage_failed="unknown"
                    )
                    failed += 1

        logger.info(f"Complete: {success} succeeded, {failed} failed")
        return self.state.get_stats()

    async def retry_failed(self):
        """Retry all failed jobs that haven't exceeded max attempts."""
        retryable = self.state.get_retryable(self.config.max_retries)
        if not retryable:
            logger.info("No retryable jobs")
            return

        logger.info(f"Resetting {len(retryable)} failed jobs for retry")

        for job in retryable:
            self.state.reset_for_retry(job.s3_key)

        # Run the pipeline again
        await self.run()
