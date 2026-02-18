# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF ingestion pipeline for municipal budget documents. Downloads PDFs from S3, extracts text, generates embeddings via OpenAI, and indexes to ChromaDB for RAG queries. Includes LLM-based budget number extraction and fine-tuning infrastructure.

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

# Local parsing (both parsers into separate collections)
python process_local.py pdfs_2026/tx_el_paso_20.pdf --parser aryn
python process_local.py pdfs_2026/tx_el_paso_20.pdf --parser pymupdf

# Gold set & chunk caching (run locally with ChromaDB)
python build_gold_set.py                  # Generates gold_validation_set.json, gold_query_embeddings.json,
                                          # gold_chunks_aryn.json, gold_chunks_pymupdf.json

# Training data generation (run locally with PDFs + pipeline_state.db)
python build_training_data.py             # 500 examples (default, stratified GF + Police)
python build_training_data.py --n 1000    # More examples

# LLM extraction testing (run on VM with GPU)
python test_llm_extraction.py --llm-url http://localhost:8000/v1 --model mistralai/Mistral-7B-Instruct-v0.3 --n-chunks 10
python test_llm_extraction.py --cache gold_chunks_aryn.json   # Test specific parser
python test_llm_extraction.py --city vallejo ca 2021          # Test specific city
python test_llm_extraction.py -v                               # Show chunks sent to LLM

# Fine-tuning (run on VM with GPU)
python finetune.py --data training_data.jsonl --epochs 3      # LoRA fine-tune Mistral 7B
vllm serve budget-mistral-lora-merged                          # Serve fine-tuned model

# Environment setup
make venv                                 # Create virtual environment
make install                              # Install dependencies
```

## Architecture

### Pipeline Flow
```
S3 (PDFs) → Download → Parse (Aryn/PyMuPDF) → Embed (OpenAI) → ChromaDB
                                                                    ↓
Gold set: ChromaDB → cached chunks (JSON) → LLM extraction → compare to ground truth
                                                                    ↓
Fine-tuning: validation table + PDFs → training JSONL → LoRA fine-tune → serve with vLLM
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
ARYN_API_KEY      # Required: Aryn DocParse API key (PAYG tier for parallel processing)
OPENAI_API_KEY    # Required: For embeddings
CHROMA_HOST       # ChromaDB host (empty for local)
CHROMA_API_KEY    # ChromaDB API key (for cloud)
CHROMA_COLLECTION # Collection name (default: municipal-budgets)
BATCH_SIZE        # Documents per batch (default: 10)
```

### Dual Parser Architecture
Two separate ChromaDB collections, one per parser:
- `budgets-aryn` — Aryn DocParse (cloud API, better table extraction)
- `budgets-pymupdf` — PyMuPDF (local, free, simpler text extraction)

This allows independent evaluation of each parser's impact on extraction accuracy. The `process_local.py` script routes to the correct collection based on `--parser` flag.

### GCP VM (GPU Inference & Fine-tuning)
```bash
# Instance: muni-rag-420, zone us-central1-a
# Machine: g2-standard-8, L4 GPU (24GB VRAM)
# Image: deep learning common-cu128-ubuntu-2204-nvidia-570

# Start VM
gcloud compute instances start muni-rag-420 --zone=us-central1-a

# SSH in
gcloud compute ssh muni-rag-420 --zone=us-central1-a

# Serve base Mistral 7B
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --port 8000 --max-model-len 16384

# Serve fine-tuned model
vllm serve budget-mistral-lora-merged --port 8000 --max-model-len 16384

# Install fine-tuning deps
pip install "unsloth[cu128-torch250] @ https://unsloth.ai/whl/0.15.3/cu128-torch250/unsloth-0.15.3-py3-none-any.whl" datasets trl
```

Code transfer to VM via GitHub + deploy keys.

## Current State

### Data
- **~103 PDFs ingested** across ~50 states
- **Separate ChromaDB collections**: `budgets-aryn` and `budgets-pymupdf` (local persistent, `chroma_data/`)
- **Ground truth DB**: SQLite `pipeline_state.db` → `validation` table with 6,126 records across 485 cities
  - Schema: `(state TEXT, city TEXT, year INTEGER, expense TEXT, budget_type TEXT, budget REAL, doc_page TEXT, pdf_page INTEGER)`
- **209 gold records** in cached chunk files, pre-retrieved top 20 chunks each

### Test Set
6 cities defined in `test_budgets.json` (each tested for General Fund + Police = 12 records):
- Vallejo CA 2021, El Paso TX 2020, Corvallis OR 2018
- Peekskill NY 2023, Carrollton GA 2022, Paducah KY 2022

### LLM Extraction Results (209 gold records, gold_chunks_cache.json)

| Model | Chunks | Match | In Chunks | Extraction Rate |
|-------|--------|-------|-----------|-----------------|
| Base Mistral 7B | 10 | 63/209 (30%) | 139/209 (67%) | 63/139 (45%) |
| **Fine-tuned** | **5** | **79/209 (38%)** | 126/209 (60%) | **79/126 (63%)** |
| **Fine-tuned** | **10** | **84/209 (40%)** | 139/209 (67%) | **84/139 (60%)** |

- Fine-tuning improved extraction rate from 45% → 60% (same chunks)
- Fine-tuned model outputs clean `$X,XXX,XXX` format (no more chatty responses)
- More chunks improves retrieval but slightly hurts extraction (noise tradeoff)
- **Error breakdown** (base, 10 chunks): wrong_scope 45, not_in_chunks 70, close_but_wrong 18, wrong_fund 7
- **By expense type**: GF extraction 54%, Police extraction 38% — Police suffers more from wrong_scope (sub-department line items)
- Test results saved as JSON in `runs/` directory for dashboard comparison

### Fine-tuning Pipeline
1. **`build_training_data.py`** — Generates training examples from validation table + local PDFs
   - Uses `pdf_page` column to find the page with the correct answer
   - Adds 4 random distractor pages per example (simulates noisy retrieval)
   - Stratified sampling: balanced General Fund + Police
   - Generated 456 examples (231 GF + 225 Police), median ~2,700 tokens
2. **`finetune.py`** — LoRA fine-tuning with unsloth + trl
   - Mistral 7B in 4-bit, LoRA r=16 on all attention + MLP layers
   - 3 epochs, batch 2, grad accum 4, lr 2e-4, cosine schedule
   - Saves LoRA adapter + merged 16-bit model for vLLM serving
3. **`training_data.jsonl`** — Chat-format training data (system/user/assistant messages)

### Key Scripts
- `process_local.py` — Parse local PDFs → embed → index into parser-specific ChromaDB collection
- `validate_retrieval.py` — Run retrieval validation against ground truth (string-match, no LLM)
- `build_gold_set.py` — Build gold validation set + cache chunks from both parser collections
- `test_llm_extraction.py` — Test LLM extraction on cached chunks (runs on VM)
- `build_training_data.py` — Generate fine-tuning JSONL from validation table + PDFs
- `finetune.py` — LoRA fine-tune Mistral 7B on budget extraction
- `test_docling.py` — Test Docling parser on test PDFs (number-in-text check)
- `analyze_results.py` — Analyze/compare test run JSON files (GF vs Police, error categories, diffs)
- `dashboard.html` — Browser-based dashboard for comparing test runs (load JSON files from `runs/`)
- `test_budgets.json` — Defines the 6-city test set
- `start_vm.sh` — Start GCP VM + launch vLLM + SSH session

### Cached Data Files
- `gold_validation_set.json` — 209 gold records (state, city, year, expense, budget_type, budget)
- `gold_query_embeddings.json` — 3 pre-computed query embeddings (General Fund, Police, Education)
- `gold_chunks_aryn.json` — Cached top-20 chunks per gold record from Aryn collection
- `gold_chunks_pymupdf.json` — Cached top-20 chunks per gold record from PyMuPDF collection
- `gold_chunks_cache.json` — Legacy combined cache (both parsers mixed)
- `training_data.jsonl` — 456 fine-tuning examples in chat format

### Aryn API Notes

**PAYG Tier ($2/1000 pages)** — currently active:
- Unlimited pages
- Async API available for concurrent/parallel processing
- Use `partition_file_async_submit` / `partition_file_async_result` for batch processing

**SDK Functions:**
- `partition_file(file, aryn_api_key=...)` - sync parsing
- `partition_file_async_submit(file, ...)` - submit for async
- `partition_file_async_result(task_id, ...)` - poll for result
- `selected_pages` param can process specific page ranges

**Docs:** https://docs.aryn.ai/docparse/aryn_sdk

### Known Issues
- Dual parser chunks may duplicate content, wasting chunk slots (mitigated by separate collections)
- Biennial budgets (e.g., 2018-2019) cause confusion — avoid using cities with biennial/mid-biennial budgets in test set
- Some gold truth values may be wrong (e.g., Fairbanks Police case)
- Peekskill NY: Aryn parser failed to produce usable chunks
- Prompt changes alone didn't improve extraction accuracy — fine-tuning is the path forward

### Fine-tuning Status
- LoRA fine-tuning completed (3 epochs, ~2 hours on L4 GPU, 456 examples)
- Merged model saved on VM at `budget-mistral-lora-merged/` (~14GB)
- **Fine-tuning works**: extraction rate 45% → 60% on 209 records (10 chunks)
- Model outputs clean dollar amounts, no more chatty/verbose responses
- VM venvs: `venv-vllm` (vLLM serving), `venv-finetune` (unsloth + openai + training)
- Fixed bf16/fp16 mismatch: L4 loads in bfloat16, so `bf16=True` in SFTConfig (not fp16)
- Serve fine-tuned model: `vllm serve ~/budget-mistral-lora-merged --port 8000 --max-model-len 32768`

### Retrieval Analysis (2026-02-17)
- **75% ceiling**: with 40 chunks, only 164/219 records have the exact answer in chunks
- **55 misses** broken down:
  - 6 parser failures (<10 chunks): Mountain Home, Henderson, Peekskill
  - 7 biennial budget confusion: Dothan, Davis, Birmingham
  - 44 number in document but parser mangled it (millions prefix found, exact value not)
  - 5 no trace of number at all
- **Aryn vs PyMuPDF** (12 test records): Aryn 10/12 (83%), PyMuPDF 7/12 (58%)
- **Chunk count doesn't matter**: answer is in top 5 or not at all
- **Key insight**: 44/55 misses are table extraction quality — the number is there but garbled

### Docling Parser Test (2026-02-17)
- Installed Docling locally and on VM (`venv-finetune`)
- `test_docling.py` — parses PDFs with Docling, checks if expected numbers appear in full text
- **Result: too slow** — even on L4 GPU (2GB VRAM used, CUDA confirmed), 10+ minutes per large PDF
- Docling uses ML-based layout detection (CPU-bound rendering + GPU inference) — not practical for production
- Still waiting for full results to see accuracy vs Aryn/PyMuPDF

### Parser Speed/Cost/Quality Comparison
| Parser | Speed | Cost | Table Quality |
|--------|-------|------|--------------|
| Aryn | Fast (cloud) | $2/1000 pages | Good (83% hit rate) |
| Docling | Very slow | Free (GPU time) | TBD |
| PyMuPDF | Instant | Free | Poor (58% hit rate) |
| pdfplumber | Fast | Free | Moderate (untested) |

### Next Steps
1. **Find optimal chunk count** — testing 5/10/15 to find sweet spot (retrieval vs noise tradeoff)
2. **Scale up training data** — more examples, especially Police wrong_scope cases
3. **Test per-parser chunks** — run fine-tuned model on `gold_chunks_aryn.json` vs `gold_chunks_pymupdf.json`
4. **Improve retrieval** — 33% of records still missing from chunks (parser quality issue)
5. **Future**: consider pdfplumber as fast+free parser with table support

### Debugging Tips
- Check status: `python -m pipeline status`
- Check failures: `python -m pipeline failures -v`
- Reset stuck jobs: `python -m pipeline reset`
- Retry failed: `python -m pipeline retry`
- Inspect chunks sent to LLM: `python test_llm_extraction.py -v --city <city> <state> <year>`
