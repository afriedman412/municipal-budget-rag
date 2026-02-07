# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF ingestion pipeline for municipal budget documents. Downloads PDFs from S3, extracts text, generates embeddings via OpenAI, and indexes to ChromaDB for RAG queries.

## Common Commands

```bash
# Run the pipeline
python -m pipeline run                    # Full pipeline (parallel producer-consumer)
python -m pipeline run --simple           # Sequential mode (for debugging)
python -m pipeline run -f "filename.pdf"  # Run a single file
python -m pipeline run --preflight        # Run preflight checks first
python -m pipeline run --reset            # Reset stuck jobs before running
python -m pipeline run --rich             # Use Rich live dashboard instead of tqdm
python -m pipeline run -m                 # Same as --rich (shortcut)
python -m pipeline run --web              # Write metrics for web monitor (see below)

# Web-based monitoring (recommended for EC2/remote)
python -m pipeline monitor                # Start web dashboard on port 8000
python -m pipeline run --web              # Run pipeline with metrics output
# Then open http://localhost:8000 or http://<ec2-ip>:8000

# Pipeline management
python -m pipeline status                 # Show job counts by status
python -m pipeline failures               # Show failed jobs and error summary
python -m pipeline failures -v            # Show full error messages
python -m pipeline preflight              # Quick preflight (first 5 pages)
python -m pipeline preflight --thorough   # Thorough preflight (all pages, catches more)
python -m pipeline skip <filename>        # Mark a PDF as failed/skipped
python -m pipeline reset-stuck            # Reset jobs stuck in intermediate states
python -m pipeline retry                  # Retry failed jobs

# Environment setup
make venv                                 # Create virtual environment
make install                              # Install dependencies
pip install fastapi uvicorn               # Required for web monitor
```

## Architecture

### Pipeline Flow
```
S3 (PDFs) → Download → Extract (PyMuPDF) → Embed (OpenAI) → ChromaDB
```

### Key Components (`pipeline/`)

- **cli.py**: Typer-based CLI with commands for running pipeline, checking status, managing failures
- **orchestrator.py**: Main `Pipeline` class with producer-consumer architecture
  - Producer thread: downloads batches + extracts in parallel (ThreadPoolExecutor)
  - Uses ThreadPoolExecutor (not ProcessPoolExecutor) because `extract_pdf_safe` spawns subprocesses
  - Uses `as_completed()` for real-time progress as extractions finish
  - Batch size defaults to worker count (PDF_WORKERS env var)
  - Consumer: embeds + indexes async (I/O-bound, batched)
  - asyncio.Queue connects them (use `loop.call_soon_threadsafe()` for producer → consumer)
- **rich_monitor.py**: Optional Rich-based live dashboard for monitoring pipeline progress
  - Shows: phase (downloading/extracting), current files, extracted/embedded counts, rates, batch progress, queue depth, errors
  - Extraction progress: shows files being processed (multiline) and counts (e.g., "extracting (3/8)")
  - Thread-safe callback-based updates, auto-disables if not TTY
- **preflight.py**: Pre-flight checks to identify problematic PDFs before extraction
  - Runs checks in subprocess so segfaults don't kill main process
  - Catches: corrupted files, encryption, RichMedia annotations, huge documents
- **extract.py**: PDF text extraction using PyMuPDF (fitz), filename metadata parsing
- **embed.py**: OpenAI embedding generation with batching, rate limiting, retries
- **chroma.py**: ChromaDB client (supports cloud, self-hosted, or local)
- **state.py**: SQLite-based job state tracking (pending → extracting → embedding → done/failed)
- **config.py**: Environment-based configuration (`PipelineConfig.from_env()`)
- **s3.py**: S3 operations (list, download, batch download)

### Filename Convention
PDFs follow: `SS_city_name_YY[_budget_type].pdf`
- SS: state abbreviation (e.g., `az`, `tx`)
- city_name: underscored (e.g., `san_antonio`)
- YY: two-digit year
- Example: `tx_san_antonio_24_proposed.pdf` → Texas, San Antonio, 2024, proposed budget

### Environment Variables
```
S3_BUCKET         # Required: S3 bucket name
S3_PREFIX         # S3 prefix for PDFs
CHROMA_HOST       # ChromaDB host (empty for local)
CHROMA_API_KEY    # ChromaDB API key (for cloud)
OPENAI_API_KEY    # For embeddings
PDF_WORKERS       # Parallel PDF extraction processes (default: 8)
```

### EC2 Instance
```bash
# Start the instance
aws ec2 start-instances --instance-ids $(aws ec2 describe-instances --query 'Reservations[0].Instances[0].InstanceId' --output text)

# Get the IP
aws ec2 describe-instances --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# SSH in (instance name: budget-rag-ingest)
ssh -i ~/.ssh/<your-key>.pem ubuntu@<ip>
```

## Current Work

Working on making the pipeline more robust and generalizable:
- Built preflight functionality to scan PDFs for errors before full ingestion
- Added crash-resistant preflight and extraction (runs in subprocess to survive segfaults)
- Thorough preflight mode (`--thorough`) tests all pages and records which pages fail
- Page-level skipping: if specific pages crash, skip just those pages, not the whole file
- Single file mode (`run -f`) for testing/debugging individual files with page-level tqdm
- Fixed ChromaDB batch size limit (was failing on docs with >5461 chunks)
- Fixed async queue issue (switched from threading.Queue to asyncio.Queue)
- Added Rich live dashboard (`--rich` / `-m` flag) for real-time pipeline monitoring
- Rich monitor shows download/extract phases with current file names and real-time metrics
- Changed producer from ProcessPoolExecutor to ThreadPoolExecutor (avoids nested process deadlocks)

### Known Issues (In Progress)
- **al_gadsden_23.pdf** causes segfault during embedding (not extraction) — unclear why
- Some PDFs pass preflight but still crash during actual extraction/embedding
- Need to investigate: crash might be in a different code path than preflight tests
- Rich Live display may stack multiple tables instead of updating in-place (terminal detection issue)

### Next Steps
1. Re-run `python -m pipeline preflight --thorough` to catch problem files
2. Investigate why some crashes aren't being caught by subprocess isolation
3. Consider if preflight needs to test embedding path too, not just extraction
4. Reconcile document counts: 903 done + ~745 pending + ~100 failed vs 1755 in S3

### Debugging Tips
- Single file with progress: `python -m pipeline run -f "filename.pdf"`
- Use Rich dashboard for monitoring: `python -m pipeline run -m`
- If ChromaDB segfaults: `rm -rf chroma_data/` and restart
- Check status: `python -m pipeline status`
- Check failures: `python -m pipeline failures -v`
