# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF ingestion pipeline for municipal budget documents. Downloads PDFs from S3, extracts text, generates embeddings via OpenAI, and indexes to ChromaDB for RAG queries.

## Common Commands

```bash
# Run the pipeline
python -m pipeline run                    # Full pipeline (parallel producer-consumer)
python -m pipeline run --simple           # Sequential mode (for debugging)
python -m pipeline run --preflight        # Run preflight checks first
python -m pipeline run --reset            # Reset stuck jobs before running

# Pipeline management
python -m pipeline status                 # Show job counts by status
python -m pipeline failures               # Show failed jobs and error summary
python -m pipeline failures -v            # Show full error messages
python -m pipeline preflight              # Run preflight checks on all pending PDFs
python -m pipeline skip <filename>        # Mark a PDF as failed/skipped
python -m pipeline reset-stuck            # Reset jobs stuck in intermediate states
python -m pipeline retry                  # Retry failed jobs

# Environment setup
make venv                                 # Create virtual environment
make install                              # Install dependencies
```

## Architecture

### Pipeline Flow
```
S3 (PDFs) → Download → Extract (PyMuPDF) → Embed (OpenAI) → ChromaDB
```

### Key Components (`pipeline/`)

- **cli.py**: Typer-based CLI with commands for running pipeline, checking status, managing failures
- **orchestrator.py**: Main `Pipeline` class with producer-consumer architecture
  - Producer thread: downloads + extracts PDFs in parallel (CPU-bound, ProcessPoolExecutor)
  - Consumer: embeds + indexes async (I/O-bound, batched)
  - Queue connects them for pipeline parallelism
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
- Added crash-resistant preflight (runs in subprocess to survive segfaults)
- Improved CLI with file names in tqdm progress bar
- Goal: make this useful for others who want to ingest PDFs into a RAG system
