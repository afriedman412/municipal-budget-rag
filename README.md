# Municipal Budget RAG Pipeline

PDF ingestion pipeline for municipal budget documents. Downloads PDFs from S3, extracts text, generates embeddings via OpenAI, and indexes to ChromaDB for RAG queries.

## Setup

```bash
# Create virtual environment and install dependencies
make venv
make install

# Configure environment variables (copy .env.example or create .env)
S3_BUCKET=your-bucket-name
S3_PREFIX=budgets/pdfs/
OPENAI_API_KEY=sk-...
CHROMA_HOST=              # leave empty for local ChromaDB
CHROMA_API_KEY=           # for cloud ChromaDB
```

## CLI Commands

### Run Pipeline

```bash
# Full pipeline (parallel producer-consumer architecture)
python -m pipeline run

# Sequential mode (useful for debugging)
python -m pipeline run --simple

# Run a single file (prompts if file is marked failed)
python -m pipeline run -f "ca_phoenix_23.pdf"

# Run preflight checks before processing
python -m pipeline run --preflight

# Reset stuck jobs before running
python -m pipeline run --reset

# Combine flags
python -m pipeline run --preflight --reset

# Adjust parallelism
python -m pipeline run --batch-size 50 --workers 4
```

### Preflight Checks

Scan PDFs for problems before processing. Catches corrupted files, encryption, problematic annotations, etc.

```bash
# Quick preflight (checks first 5 pages per PDF)
python -m pipeline preflight

# Thorough preflight (tests ALL pages - slower but catches more crashers)
# Records which specific pages fail so extraction can skip just those pages
python -m pipeline preflight --thorough

# Limit number of PDFs to check
python -m pipeline preflight --limit 100

# Adjust timeout per PDF
python -m pipeline preflight --timeout 120
```

**Page-level skipping:** In thorough mode, if specific pages fail but the rest of the PDF is OK, those pages are recorded and skipped during extraction. You get partial documents instead of losing the whole file.

### Status & Monitoring

```bash
# Show job counts by status
python -m pipeline status

# Show failed jobs and error summary
python -m pipeline failures

# Show full error messages
python -m pipeline failures -v

# Limit number of failures shown
python -m pipeline failures --limit 50
```

### Managing Jobs

```bash
# Discover PDFs in S3 without processing
python -m pipeline discover

# Skip a problematic PDF (mark as failed)
python -m pipeline skip "filename.pdf"
python -m pipeline skip "filename.pdf" --reason "Causes segfault"

# Reset jobs stuck in intermediate states
python -m pipeline reset-stuck

# Retry failed jobs
python -m pipeline retry

# Reset ALL failed jobs to pending
python -m pipeline reset-failed
```

## Pipeline Architecture

```
S3 (PDFs) → Download → Extract (PyMuPDF) → Embed (OpenAI) → ChromaDB
```

**Producer-Consumer Model:**
- Producer thread: downloads + extracts PDFs in parallel (CPU-bound, ProcessPoolExecutor)
- Consumer: embeds + indexes async (I/O-bound, batched API calls)
- Queue connects them for pipeline parallelism

**Crash Resistance:**
- Preflight runs in subprocess to survive segfaults
- Extraction is subprocess-isolated so bad PDFs don't kill the pipeline
- Page-level skipping: bad pages are skipped, rest of document is extracted
- Failed jobs are tracked and can be retried

## PDF Filename Convention

Files should follow: `SS_city_name_YY[_budget_type].pdf`

- `SS`: two-letter state abbreviation
- `city_name`: city name with underscores
- `YY`: two-digit year
- `budget_type`: optional (e.g., proposed, adopted, capital)

Examples:
- `az_phoenix_23.pdf` → Arizona, Phoenix, 2023
- `tx_san_antonio_24_proposed.pdf` → Texas, San Antonio, 2024, proposed budget
- `ca_los_angeles_22_capital.pdf` → California, Los Angeles, 2022, capital budget

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `S3_BUCKET` | S3 bucket name (required) | — |
| `S3_PREFIX` | S3 prefix for PDFs | `""` |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | — |
| `CHROMA_HOST` | ChromaDB host (empty for local) | `""` |
| `CHROMA_API_KEY` | ChromaDB API key (for cloud) | `""` |
| `CHROMA_COLLECTION` | ChromaDB collection name | `municipal-budgets` |
| `PDF_WORKERS` | Parallel PDF extraction processes | `8` |
| `EMBED_BATCH_SIZE` | Texts per embedding API call | `50` |
| `EMBED_CONCURRENCY` | Concurrent embedding requests | `10` |
