"""
Main pipeline orchestrator.

Coordinates: S3 → Extract → Embed → ChromaDB
With parallel processing, fault tolerance, and progress tracking.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

from tqdm import tqdm

from .config import PipelineConfig
from .state import StateDB, JobStatus
from .s3 import S3Client
from .rich_monitor import RichMonitor
from .extract import extract_pdf_safe, ExtractedDocument
from .embed import EmbeddingClient, document_to_chunks
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
        logger.info(
            f"Listing PDFs in s3://{self.config.s3_bucket}/{self.config.s3_prefix}")
        keys = self.s3.list_pdfs()
        logger.info(f"Found {len(keys)} PDFs in S3")

        added = self.state.register_jobs(keys)
        logger.info(
            f"Registered {added} new jobs ({len(keys) - added} already tracked)")
        return added

    def _extraction_producer(
        self,
        out_queue,
        loop: asyncio.AbstractEventLoop,
        batch_size: int = None,
        progress_cb=None,
    ):
        """
        Producer: Download and extract PDFs in small batches.

        Uses batch_size (default: num_workers) to download a batch,
        then extract in parallel, showing progress as each completes.
        """
        def _put(item):
            loop.call_soon_threadsafe(out_queue.put_nowait, item)

        def _progress(phase, current_file="", **kwargs):
            if progress_cb:
                progress_cb(phase=phase, current_file=current_file, **kwargs)

        num_workers = self.config.pdf_workers
        if batch_size is None:
            batch_size = num_workers

        producer_extracted_docs = 0
        producer_extracted_chunks = 0

        while True:
            # Get next batch of jobs
            pending = self.state.get_pending(JobStatus.EXTRACTING, limit=batch_size)
            if not pending:
                break

            # Mark as extracting
            job_lookup = {}
            for job in pending:
                self.state.update_status(job.s3_key, JobStatus.EXTRACTING)
                job_lookup[job.s3_key] = job

            # Download batch
            s3_keys = [job.s3_key for job in pending]
            file_names = [Path(k).name for k in s3_keys]
            _progress("downloading", current_file="\n".join(file_names))

            download_results = self.s3.download_batch(s3_keys)

            # Build work items from successful downloads
            work_items = []
            for s3_key, local_path, error in download_results:
                if error:
                    self.state.update_status(
                        s3_key, JobStatus.FAILED,
                        error_message=f"Download failed: {error}",
                        stage_failed="download"
                    )
                else:
                    skip_pages = job_lookup[s3_key].skip_pages
                    work_items.append((s3_key, str(local_path), skip_pages))

            if not work_items:
                continue

            # Extract in parallel with progress
            file_names = [Path(item[0]).name for item in work_items]
            _progress(
                "extracting",
                current_file="\n".join(file_names),
                extract_total=len(work_items),
                extract_done=0,
            )

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_key = {
                    executor.submit(_extract_single_pdf, item): item[0]
                    for item in work_items
                }

                batch_done = 0
                for future in as_completed(future_to_key):
                    s3_key, doc, error = future.result()
                    local_path = self.config.temp_dir / Path(s3_key).name
                    batch_done += 1

                    if error:
                        self.state.update_status(
                            s3_key, JobStatus.FAILED,
                            error_message=error, stage_failed="extract"
                        )
                    else:
                        self.state.update_status(
                            s3_key, JobStatus.EXTRACTED,
                            metadata={
                                "city": doc.city, "state": doc.state,
                                "year": doc.year, "page_count": doc.page_count,
                            }
                        )
                        producer_extracted_docs += 1
                        producer_extracted_chunks += len(document_to_chunks(doc))
                        _put(ExtractedJob(s3_key=s3_key, document=doc))

                    # Update progress
                    _progress(
                        "extracting",
                        current_file="\n".join(file_names),
                        extract_total=len(work_items),
                        extract_done=batch_done,
                        extracted_docs=producer_extracted_docs,
                        extracted_chunks=producer_extracted_chunks,
                    )

                    self.s3.cleanup_local(local_path)

        logger.info(f"Producer: complete, extracted {producer_extracted_docs} docs")
        _progress("done", current_file="")
        _put(None)

    async def _embedding_consumer(
        self,
        queue: asyncio.Queue,
        batch_size: int = 10
    ) -> tuple[int, int]:
        """
        Consumer: Pull extracted docs from an asyncio queue, embed and index.

        Note: This is not currently used by `run()` (which has a progress bar and
        explicit concurrency controls), but it's kept as a simpler reusable consumer.
        """
        success = 0
        failed = 0
        batch: list[ExtractedJob] = []

        while True:
            item = await queue.get()

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

    async def run(self, batch_size: int = None, use_rich: bool = False):
        """
        Run the full pipeline with real producer-consumer parallelism.

        - Producer thread: downloads + extracts PDFs (CPU-bound, parallel)
        - Consumer: embeds + indexes (I/O-bound, async, batched)
        - Async queue connects them without blocking the event loop

        Args:
            batch_size: Number of PDFs to download/extract per batch (defaults to pdf_workers)
            use_rich: Use Rich monitor instead of tqdm progress bar
        """
        logger.info("Starting pipeline run")
        logger.info(
            f"Config: {self.config.pdf_workers} PDF workers, "
            f"{self.config.embed_concurrency} embed concurrency"
        )

        # Phase 1: Discover new PDFs
        self.discover_pdfs()

        stats = self.state.get_stats()
        pending_count = stats.get("pending", 0) + stats.get("extracted", 0)

        if pending_count == 0:
            logger.info("No pending work")
            return stats

        logger.info(f"Processing {pending_count} documents")

        # Default batch size to worker count (download only what we can extract)
        if batch_size is None:
            batch_size = self.config.pdf_workers

        loop = asyncio.get_running_loop()
        out_queue: asyncio.Queue[ExtractedJob |
                                 None] = asyncio.Queue(maxsize=batch_size * 2)

        # Tracking for Rich monitor
        extracted_docs = 0
        extracted_chunks = 0
        embedded_chunks = 0
        batches_started = 0
        batches_done = 0

        # Embed/index concurrently (bounded by embed_concurrency)
        sem = asyncio.Semaphore(max(1, int(self.config.embed_concurrency)))
        embed_batch_size = 10

        async def _process_batch(jobs: list[ExtractedJob]) -> tuple[int, int, int, int]:
            # returns (success, failed, jobs_processed, chunks_embedded)
            async with sem:
                s, f = await self._embed_batch(jobs)
                # Count chunks from successful jobs
                chunk_count = sum(
                    len(document_to_chunks(job.document))
                    for job in jobs
                )
                return s, f, len(jobs), chunk_count

        success = 0
        failed = 0
        in_flight: set[asyncio.Task] = set()
        batch: list[ExtractedJob] = []

        # Set up progress display (Rich monitor or tqdm)
        monitor = None
        pbar = None
        # Shared state for producer → monitor communication
        producer_phase = {"phase": "starting", "current_file": "", "download_count": 0, "download_total": 0}

        if use_rich:
            monitor = RichMonitor(refresh_hz=4)
            monitor.start()
            # Suppress info logging when Rich monitor is active (it interferes with display)
            logging.getLogger(__name__).setLevel(logging.WARNING)
        else:
            pbar = tqdm(total=pending_count, desc="Processing")

        def _producer_progress(phase, current_file="", **kwargs):
            """Callback for producer to update monitor with phase/file info."""
            producer_phase["phase"] = phase
            producer_phase["current_file"] = current_file
            producer_phase.update(kwargs)
            if monitor:
                monitor.update({
                    "phase": phase,
                    "current_file": current_file,
                    **kwargs
                })

        def _update_monitor(current_file=""):
            if monitor:
                # Use producer's extraction counts (real-time) + consumer's embedding counts
                monitor.update({
                    "extracted_docs": producer_phase.get("extracted_docs", 0) or extracted_docs,
                    "extracted_chunks": producer_phase.get("extracted_chunks", 0) or extracted_chunks,
                    "embedded_docs": success,
                    "embedded_chunks": embedded_chunks,
                    "batches_started": batches_started,
                    "batches_done": batches_done,
                    "queue_depth": out_queue.qsize(),
                    "producer_done": not producer.is_alive(),
                    "errors": failed,
                    "phase": producer_phase["phase"],
                    "current_file": current_file or producer_phase["current_file"],
                    "download_count": producer_phase.get("download_count", 0),
                    "download_total": producer_phase.get("download_total", 0),
                    "extract_done": producer_phase.get("extract_done", 0),
                    "extract_total": producer_phase.get("extract_total", 0),
                })

        # Start producer in background thread (after callbacks are defined)
        producer = Thread(
            target=self._extraction_producer,
            args=(out_queue, loop, batch_size, _producer_progress),
            daemon=True
        )
        producer.start()

        try:
            while True:
                item = await out_queue.get()

                if item is None:
                    break

                batch.append(item)
                extracted_docs += 1
                extracted_chunks += len(document_to_chunks(item.document))

                # Show current file
                filename = item.s3_key.split('/')[-1]
                if pbar:
                    pbar.set_postfix_str(filename[:40])
                _update_monitor(current_file=filename)

                if len(batch) >= embed_batch_size:
                    jobs = batch
                    batch = []
                    batches_started += 1
                    in_flight.add(asyncio.create_task(_process_batch(jobs)))
                    _update_monitor()

                # Reap completed tasks without blocking
                if in_flight:
                    done, pending = await asyncio.wait(
                        in_flight,
                        timeout=0,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    in_flight = pending
                    for t in done:
                        s, f, n, chunks = t.result()
                        success += s
                        failed += f
                        embedded_chunks += chunks
                        batches_done += 1
                        if pbar:
                            pbar.update(n)
                        _update_monitor()

            # Final partial batch
            if batch:
                batches_started += 1
                in_flight.add(asyncio.create_task(_process_batch(batch)))
                batch = []
                _update_monitor()

            # Drain remaining in-flight embedding/indexing tasks
            while in_flight:
                done, pending = await asyncio.wait(
                    in_flight,
                    return_when=asyncio.FIRST_COMPLETED
                )
                in_flight = pending
                for t in done:
                    s, f, n, chunks = t.result()
                    success += s
                    failed += f
                    embedded_chunks += chunks
                    batches_done += 1
                    if pbar:
                        pbar.update(n)
                    _update_monitor()

        finally:
            if pbar:
                pbar.close()
            if monitor:
                # Final update before stopping
                monitor.update({
                    "extracted_docs": extracted_docs,
                    "extracted_chunks": extracted_chunks,
                    "embedded_docs": success,
                    "embedded_chunks": embedded_chunks,
                    "batches_started": batches_started,
                    "batches_done": batches_done,
                    "queue_depth": 0,
                    "producer_done": True,
                    "errors": failed,
                    "phase": "done",
                })
                monitor.stop()

        producer.join(timeout=5.0)

        # Summary
        stats = self.state.get_stats()
        logger.info("=" * 60)
        logger.info("Pipeline run complete")
        logger.info(f"  Processed: {success} succeeded, {failed} failed")
        logger.info(f"  State: {stats}")

        chroma_stats = self.chroma.get_stats()
        logger.info(
            f"  ChromaDB: {chroma_stats['total_chunks']} chunks, "
            f"{chroma_stats['unique_documents']} documents"
        )

        return stats

    async def run_simple(self, batch_size: int = 100, use_rich: bool = False):
        """
        Simpler sequential run (useful for debugging).
        Processes one document at a time: extract → embed → index.

        Args:
            batch_size: Number of PDFs to fetch per batch
            use_rich: Use Rich monitor instead of tqdm progress bar
        """
        logger.info("Starting simple sequential run")

        self.discover_pdfs()

        stats = self.state.get_stats()
        total_pending = stats.get("pending", 0) + stats.get("extracted", 0)

        if total_pending == 0:
            logger.info("No pending work")
            return stats

        success = 0
        failed = 0
        extracted_chunks = 0
        embedded_chunks = 0

        # Set up progress display
        monitor = None
        if use_rich:
            monitor = RichMonitor(refresh_hz=4)
            monitor.start()
            # Suppress info logging when Rich monitor is active (it interferes with display)
            logging.getLogger(__name__).setLevel(logging.WARNING)

        def _update_monitor(phase="processing", current_file=""):
            if monitor:
                monitor.update({
                    "extracted_docs": success + failed,
                    "extracted_chunks": extracted_chunks,
                    "embedded_docs": success,
                    "embedded_chunks": embedded_chunks,
                    "batches_started": 1,
                    "batches_done": 1 if (success + failed) > 0 else 0,
                    "queue_depth": 0,
                    "producer_done": False,
                    "errors": failed,
                    "phase": phase,
                    "current_file": current_file,
                })

        try:
            while True:
                pending = self.state.get_pending(
                    JobStatus.EXTRACTING, limit=batch_size)
                if not pending:
                    break

                if use_rich:
                    job_iter = pending
                else:
                    pbar = tqdm(pending, desc="Processing")
                    job_iter = pbar

                for job in job_iter:
                    # Show current file
                    filename = job.s3_key.split('/')[-1]
                    if not use_rich:
                        pbar.set_postfix_str(filename[:40])

                    _update_monitor(phase="downloading", current_file=filename)

                    try:
                        # Download
                        logger.info(f"Downloading {job.s3_key}")
                        local_path = self.s3.download_pdf(job.s3_key)

                        # Extract (crash-safe, with page skipping if preflight identified bad pages)
                        _update_monitor(phase="extracting", current_file=filename)
                        self.state.update_status(job.s3_key, JobStatus.EXTRACTING)
                        result = extract_pdf_safe(
                            local_path, job.s3_key, skip_pages=job.skip_pages)

                        if not result.success:
                            self.state.update_status(
                                job.s3_key,
                                JobStatus.FAILED,
                                error_message=result.error,
                                stage_failed="extract"
                            )
                            failed += 1
                            _update_monitor(phase="failed", current_file=filename)
                            self.s3.cleanup_local(local_path)
                            continue

                        doc = result.document
                        extracted_chunks += len(document_to_chunks(doc))

                        # Embed
                        _update_monitor(phase="embedding", current_file=filename)
                        self.state.update_status(job.s3_key, JobStatus.EMBEDDING)
                        chunks = document_to_chunks(doc)

                        if chunks:
                            embedded = await self.embedder.embed_chunks(chunks)
                            self.chroma.add_chunks(embedded)
                            embedded_chunks += len(chunks)

                        self.state.update_status(job.s3_key, JobStatus.DONE)
                        success += 1
                        _update_monitor(phase="done", current_file=filename)

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
                        _update_monitor(phase="error", current_file=filename)

        finally:
            if monitor:
                monitor.update({
                    "extracted_docs": success + failed,
                    "extracted_chunks": extracted_chunks,
                    "embedded_docs": success,
                    "embedded_chunks": embedded_chunks,
                    "batches_started": 1,
                    "batches_done": 1,
                    "queue_depth": 0,
                    "producer_done": True,
                    "errors": failed,
                    "phase": "done",
                })
                monitor.stop()

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
