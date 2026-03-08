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
python build_gold_set.py                  # Generates training/gold_*.json files

# Training data generation (run locally with PDFs + pipeline_state.db)
python build_training_data.py             # 500 examples (default, stratified GF + Police)
python build_training_data.py --n 1000    # More examples

# LLM extraction testing (run on VM with GPU)
python test_llm_extraction.py --llm-url http://localhost:8000/v1 --model mistralai/Mistral-7B-Instruct-v0.3 --n-chunks 10
python test_llm_extraction.py --cache training/gold_chunks_aryn.json   # Test specific parser
python test_llm_extraction.py --city vallejo ca 2021          # Test specific city
python test_llm_extraction.py -v                               # Show chunks sent to LLM

# Fine-tuning (run on VM with GPU)
python finetune.py --data training/training_data.jsonl --epochs 3  # LoRA fine-tune Mistral 7B
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
- **ChromaDB collections** (local persistent, `chroma_data/`): `budgets-aryn` (9390), `budgets-pymupdf` (10075), `budgets-llamaparse` (1779), `budgets-unstructured` (1067)
- **Ground truth DB**: SQLite `pipeline_state.db` → `validation` table with 6,126 records across 485 cities
  - Schema: `(state TEXT, city TEXT, year INTEGER, expense TEXT, budget_type TEXT, budget REAL, doc_page TEXT, pdf_page INTEGER)`
- **60 validation records** in test chunk cache files, per parser (test split)

### Test Set
60 validation records across multiple cities, tested per parser and with/without distractors.
6-city parser test set defined in `parser_test_set.json` for quick parser quality checks.

### V7 Results (fine-tuned Mistral 7B, 5 samples majority vote, new prompt)

| Suite | Match | Close | Miss |
|-------|-------|-------|------|
| pymupdf/clean | 53/60 (88%) | 4 | 3 |
| pymupdf/d4 | 45/60 (75%) | 4 | 11 |
| aryn/clean | 53/60 (88%) | 4 | 3 |
| aryn/d4 | 49/60 (81%) | 4 | 7 |

- Aryn is best overall (81% on noisy d4 chunks vs 75% for pymupdf)
- pymupdf and aryn tied at 88% on clean chunks
- Model is very deterministic at temp=0.3 (multi-sample confirms)
- 5 epochs is the sweet spot (10 epochs showed no improvement in V6)

### Prompt Rules
System prompt includes these extraction rules (in both `test_llm_extraction.py` and `build_training_data.py`):
- Return EXPENDITURES only, ADOPTED/APPROVED budget, BUDGETED/APPROPRIATED amount
- **For General Fund, return GF total only — not total/all-funds/combined budget**
- For Police, return General Fund police expenditure
- **Exclude interfund transfers if both values shown**
- Return only `$X,XXX,XXX` format or "NOT FOUND"

### Transfer Ambiguity Analysis
- 719 General Fund records audited: 65% AMBIGUOUS, 30% EXCLUDES_TRANSFERS, 5% INCLUDES_TRANSFERS
- Convention is clearly "excludes transfers" (214:35 ratio)
- Only ~5-10 records where transfers actually cause CLOSE errors (Cincinnati, San Rafael, Eau Claire)
- Not a major lever — wrong_scope and parser quality are bigger issues

### Fine-tuning Pipeline
1. **`build_training_data.py`** — Generates training examples from validation table + local PDFs
   - Uses `pdf_page` column to find the page with the correct answer
   - Configurable distractor pages per example (simulates noisy retrieval)
   - Stratified sampling: balanced General Fund + Police
2. **`finetune.py`** — LoRA fine-tuning with unsloth + trl
   - Mistral 7B in 4-bit, LoRA on all attention + MLP layers
   - Key params: epochs, lr, lora-r, lora-alpha, warmup, packing
   - Saves LoRA adapter + merged 16-bit model for vLLM serving
3. **`training/training_data_d4.jsonl`** — Current training data (4 distractors)

### Fine-tuning History
| Version | Epochs | LR | LoRA r | Distractors | aryn clean | aryn d4 |
|---------|--------|----|--------|-------------|------------|---------|
| V6 | 5 | 2e-4 | 16 | 2 | 87% | 78% |
| V7 | 5 | 1e-4 | 32 | 4 | 88% | 81% |

### Key Scripts
- `process_local.py` — Parse local PDFs → embed → index into parser-specific ChromaDB collection
- `build_gold_set.py` — Build gold validation set + cache chunks from all parser collections
- `test_llm_extraction.py` — Test LLM extraction on cached chunks (runs on VM). Supports `--samples N` for majority vote
- `run_test_suite.py` — Run all 6 test suites (pymupdf/aryn/pdfplumber x clean/d4), print summary grid
- `build_training_data.py` — Generate fine-tuning JSONL from validation table + PDFs
- `finetune.py` — LoRA fine-tune Mistral 7B on budget extraction
- `test_parser.py` — Quick parser quality test (number-in-text check) for multiple parsers
- `validate_transfers.py` — Audit ground truth for transfer inclusion/exclusion
- `analyze_results.py` — Analyze/compare test run JSON files
- `dashboard.html` — Browser-based dashboard for comparing test runs (load JSON files from `runs/`)
- `config.py` — Central config: test PDFs, parser list, paths, embedding params
- `start_vm.sh` — Start GCP VM + launch vLLM + SSH session

### Parser Comparison (tested on 6-city parser test set)

| Parser | Speed | Cost | Test Set Quality | Verdict |
|--------|-------|------|-----------------|---------|
| **Aryn** | Fast (cloud) | $2/1000 pages | **88% (best)** | **Use this** |
| PyMuPDF | Instant | Free | 88% clean, 75% d4 | Good free option |
| LlamaParse | Fast (cloud) | Free tier 1K pg/day | 3/6 (50%) | Not worth it |
| Unstructured | Very slow | Free tier | 2/6 (33%) | Too slow + errors |
| Docling | Very slow | Free (GPU) | TBD | Too slow |
| pdfplumber | Fast | Free | ~73% clean | Moderate |

### Known Issues
- Biennial budgets (e.g., 2018-2019) cause confusion — avoid in test set
- Some gold truth values may be wrong (e.g., transfer inclusion inconsistency)
- Corvallis OR 2018 GF: stubborn miss across all models/parsers
- Milwaukee WI, Eau Claire WI: consistently regress between model versions
- Wrong_scope (~30 records) is the biggest error category — model picks All Funds instead of GF

### GCP VM
- Instance: muni-rag-420, zone us-central1-a, g2-standard-8, L4 GPU (24GB VRAM)
- ~$0.90/hr — stop when done
- VM venvs: `venv-vllm` (vLLM serving), `venv-finetune` (unsloth + openai + training)
- Models on VM: `budget-mistral-lora-v3/v4/v5/v6/v7-merged`
- Serve: `vllm serve budget-mistral-lora-v7-merged --port 8000 --max-model-len 32768`
- Code transfer via GitHub + deploy keys

### Next Steps
1. **Retrain with updated prompt** — regenerate training data with GF-scope and transfer-exclusion rules
2. **Address wrong_scope in training data** — ensure examples include pages with both All Funds and GF totals

### Debugging Tips
- Check status: `python -m pipeline status`
- Check failures: `python -m pipeline failures -v`
- Reset stuck jobs: `python -m pipeline reset`
- Retry failed: `python -m pipeline retry`
- Inspect chunks sent to LLM: `python test_llm_extraction.py -v --city <city> <state> <year>`
