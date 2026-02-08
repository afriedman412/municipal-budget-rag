# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF ingestion pipeline for municipal budget documents. Downloads PDFs from S3, extracts text, generates embeddings via OpenAI, and indexes to ChromaDB for RAG queries.

## Common Commands

```bash
# Run the pipeline
python -m pipeline run                    # Run pipeline on pending jobs
python -m pipeline run --limit 5          # Process only N documents
python -m pipeline run -v                 # Verbose logging

# Pipeline management
python -m pipeline status                 # Show job counts by status
python -m pipeline discover               # Discover PDFs in S3 without processing
python -m pipeline failures               # Show failed jobs and error summary
python -m pipeline failures -v            # Show full error messages
python -m pipeline retry                  # Reset failed jobs to pending
python -m pipeline reset                  # Reset stuck processing jobs to pending
python -m pipeline stats                  # Show ChromaDB statistics

# Environment setup
make venv                                 # Create virtual environment
make install                              # Install dependencies
```

## Architecture

### Pipeline Flow
```
S3 (PDFs) → Download → Parse (Aryn DocParse) → Embed (OpenAI) → ChromaDB
```

### Key Components (`pipeline/`)

- **cli.py**: Typer-based CLI with commands for running pipeline, checking status, managing failures
- **pipeline.py**: Main `Pipeline` class with async batch processing
  - Discovers PDFs in S3, registers as jobs
  - Processes in batches: download → parse → embed → index
  - All async using asyncio
- **aryn.py**: Aryn DocParse client for PDF parsing
  - Wraps sync SDK with `asyncio.to_thread()`
  - Extracts text, tables, and structure from PDFs
  - Parses filename metadata (state, city, year)
- **embed.py**: OpenAI embedding generation with batching
  - Chunks documents with overlap
  - Batches API calls (100 chunks per request)
- **chroma.py**: ChromaDB client (supports cloud, self-hosted, or local)
- **state.py**: SQLite-based job state tracking (pending → processing → done/failed)
- **config.py**: Environment-based configuration (`Config.from_env()`)
- **s3.py**: Async S3 operations (list, download, batch download)

### Old Architecture (in `pipeline_old/`)
The previous PyMuPDF-based extraction code is preserved in `pipeline_old/` for reference.

### Filename Convention
PDFs follow: `SS_city_name_YY[_budget_type].pdf`
- SS: state abbreviation (e.g., `az`, `tx`)
- city_name: underscored (e.g., `san_antonio`)
- YY: two-digit year
- Example: `tx_san_antonio_24_proposed.pdf` → Texas, San Antonio, 2024, proposed budget

### Environment Variables
```
S3_BUCKET         # Required: S3 bucket name
S3_PREFIX         # S3 prefix for PDFs (e.g., "de" for Delaware files)
ARYN_API_KEY      # Required: Aryn DocParse API key
OPENAI_API_KEY    # Required: For embeddings
CHROMA_HOST       # ChromaDB host (empty for local)
CHROMA_API_KEY    # ChromaDB API key (for cloud)
CHROMA_COLLECTION # Collection name (default: municipal-budgets)
BATCH_SIZE        # Documents per batch (default: 10)
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

## Current State

### Data
- **~103 PDFs ingested** across ~50 states, both parsers
- **170,577 chunks** in ChromaDB (local persistent, `chroma_data/`)
- **Dual parser**: Aryn DocParse (cloud) + PyMuPDF (local), both stored with `parser` metadata field
- **Ground truth DB**: SQLite `pipeline_state.db` → `validation` table with 6,126 records across 485 cities
  - Schema: `(state TEXT, city TEXT, year INTEGER, expense TEXT, budget_type TEXT, budget REAL, doc_page TEXT, pdf_page INTEGER)`

### Validation (`validate_retrieval.py`)
- Iterates ALL rows from `validation` table (no hardcoded city list)
- Pre-caches 3 query embeddings (General Fund, Police, Education)
- Tests both semantic-only and keyword-filtered retrieval
- **Current results: 84% retrieval rate** (81/97 records where chunks contain the expected value)
- Currently uses **string-matching** on raw chunks — NOT full RAG

### What's Missing: LLM Extraction Step
The validation currently checks if the expected number appears in retrieved chunks (string match).
The intended full RAG loop is:
1. Retrieve chunks from ChromaDB (have this)
2. Send chunks to LLM → extract definitive budget number (MISSING)
3. Compare LLM answer to ground truth (have the comparison logic)

**Plan**: Use a local LLM (Ollama or vLLM) to avoid OpenAI costs:
- Ollama: `brew install ollama && ollama pull llama3.1:8b` — uses Metal on Apple Silicon
- vLLM: For GPU compute instance — `vllm serve meta-llama/Llama-3.1-8B-Instruct`
- Both expose OpenAI-compatible API, so code uses same `openai` Python client with different `base_url`

### Key Scripts
- `process_local.py` — Process local PDFs: parse → embed → index. Supports `--parser {aryn,pymupdf}`
- `validate_retrieval.py` — Run retrieval validation against ground truth

### Aryn API Notes

**Free Tier Limitations:**
- 10,000 pages/month
- 1,000 documents max storage
- **Sequential processing only** - one request at a time (429 error if concurrent)
- No async API access (`partition_file_async_submit` returns 403)

**PAYG Tier ($2/1000 pages):**
- Unlimited pages
- Async API available for concurrent processing
- Use `partition_file_async_submit` / `partition_file_async_result` for batch processing

**SDK Functions:**
- `partition_file(file, aryn_api_key=...)` - sync parsing
- `partition_file_async_submit(file, ...)` - submit for async (PAYG only)
- `partition_file_async_result(task_id, ...)` - poll for result
- `selected_pages` param can process specific page ranges

**Docs:** https://docs.aryn.ai/docparse/aryn_sdk

### Next Steps
1. Build LLM extraction step into validation (local Ollama or vLLM on GPU)
2. Add `--llm-url` flag to `validate_retrieval.py` for flexible backend
3. Investigate the 16% retrieval misses (mostly scanned/image-heavy PDFs that parsed poorly)

### Debugging Tips
- Check status: `python -m pipeline status`
- Check failures: `python -m pipeline failures -v`
- Reset stuck jobs: `python -m pipeline reset`
- Retry failed: `python -m pipeline retry`
