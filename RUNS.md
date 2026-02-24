# Run Tracker

## V1 — Base Mistral (no fine-tuning)
- **Date**: 2026-02-10
- **Model**: mistralai/Mistral-7B-Instruct-v0.3 (base, no LoRA)
- **Training**: none
- **Test set**: gold_chunks_cache.json (108 records, mixed parsers)
- **Inference params**: n-chunks=20, NO max_tokens, temperature=0
- **Results**:
  - Aryn: 42/108 (39%), in-chunks 75/108
  - PyMuPDF: 29/108 (27%), in-chunks 66/108

## V2 — First fine-tune (mixed training data)
- **Date**: 2026-02-19
- **Model**: budget-mistral-lora-merged (LoRA on Mistral 7B)
- **Training**:
  - 517 examples (411 PyMuPDF page-based + 106 Aryn chunk-based)
  - 4 distractor pages per PDF example
  - Chunk headers: `[Page N]` for PDF, `[Chunk N]` for cache
  - 3 epochs, lr=2e-4, batch=2, grad_accum=4, max_seq_len=8192
  - Final loss: 0.985 (high)
- **Test set**: gold_chunks_aryn.json / gold_chunks_pymupdf.json (108 records each)
- **Inference params**: n-chunks=5, max_tokens=50, temperature=0
- **Results**:
  - Aryn: 36/108 (33%), in-chunks 69/108
  - PyMuPDF: 26/108 (24%), in-chunks 64/108
- **Notes**: Regression from V1. NOT a fair comparison — different n-chunks, V1 had no max_tokens (long rambling outputs may have stumbled onto correct values). Training data had format mismatch (pages vs chunks, mixed parsers).

## V3 — PyMuPDF-only, no distractors
- **Date**: 2026-02-20
- **Model**: budget-mistral-lora-v3-merged (LoRA on Mistral 7B)
- **Training**:
  - 206 examples (103 cities x 2 expenses, PyMuPDF pages only)
  - No distractors — single target page per example
  - Chunk headers: `[Chunk 1]`
  - 10 epochs, lr=2e-4, batch=2, grad_accum=4, max_seq_len=8192
  - Train/val split: 103 train / 30 validation (from pymupdf_split.json)
  - All records verified: value appears on target page via PyMuPDF
  - Loss curve: 1.22 → 0.98 → 0.89 (at epoch 2, continued to epoch 10)
- **Test set**: single target page per record (n-chunks=1)
- **Inference params**: n-chunks=1, max_tokens=50, temperature=0
- **Baseline (base Mistral, same test)**: 131/266 (49%)
- **Results**:
  - Full (266 records, includes training): **226/266 (85%)**
  - Validation (60 held-out): **40/60 (67%)**
- **Takeaway**: Fine-tuning works. 49% → 67% on held-out data. Some overfitting (85% vs 67%). Model learned to read budget tables. Next: add distractors and test with real retrieval chunks.

## V3 adversarial — V3 model tested with distractors
- **Date**: 2026-02-21
- **Model**: budget-mistral-lora-v3-merged (same as V3)
- **Training**: same as V3 (no distractors in training)
- **Test set**: target page + 4 random distractor pages per record
- **Inference params**: n-chunks=5, max_tokens=50, temperature=0
- **Results**:
  - Full (266 records): **177/266 (67%)**
  - Validation (60 held-out): **38/60 (63%)**
- **Takeaway**: Distractors drop accuracy (85% → 67% full, 67% → 63% val). Model still performs reasonably — only 4% drop on held-out data suggests it learned to find the right page, not just memorize.

## V4 — PyMuPDF with distractors in training
- **Date**: 2026-02-21
- **Model**: budget-mistral-lora-v4-merged (LoRA on Mistral 7B)
- **Training**:
  - 206 examples (103 cities x 2 expenses, PyMuPDF pages only)
  - 4 random distractor pages per example (~13,156 avg chars/prompt)
  - 10 epochs, lr=2e-4, batch=1→2, grad_accum=8→4, max_seq_len=8192
  - **Note**: Training crashed at epoch ~4.5 (OOM with batch=2). Resumed with batch=1, grad_accum=8. Resume loaded checkpoint from epoch 4, but trainer_state.json overrode batch size back to 2. Cosine LR schedule was disrupted.
  - Loss: started ~2.5, dropped to ~0.6 before crash; resumed and completed to epoch 10
- **Test set**: validation only (60 held-out records)
- **Inference params**: n-chunks=1 or 5, max_tokens=50, temperature=0
- **Results**:
  - Validation clean (1 page): **40/60 (67%)** — same as V3
  - Validation adversarial (5 pages): **34/60 (57%)** — worse than V3 adversarial (63%)
- **Takeaway**: Distractors in training didn't help (and hurt on adversarial). Likely due to messy training (crash + resume with batch size mismatch disrupted LR schedule). A clean re-run might give different results, but more training data is probably the bigger lever.

## V5 — More training data (6.7x), no distractors
- **Date**: 2026-02-21
- **Model**: budget-mistral-lora-v5-merged (LoRA on Mistral 7B)
- **Training**:
  - 1,378 examples (689 cities x 2 expenses, PyMuPDF pages only)
  - No distractors — single target page per example (~3,184 avg chars/prompt)
  - 5 epochs, lr=2e-4, batch=2, grad_accum=4, max_seq_len=8192
  - Train/val split: 689 train / 30 validation (same 30 val cities as V3)
  - Interrupted at epoch 2.3 (Ctrl+C), resumed from epoch 2 checkpoint
  - Loss curve: 1.23 → 0.70 (epoch 2.3)
- **Inference params**: n-chunks=1 or 5, max_tokens=50, temperature=0
- **Results**:
  - Validation clean (1 page): **44/60 (73%)**
  - Validation adversarial (5 pages): **44/60 (73%)**
  - Full clean (999 records, limit): **920/999 (92%)**
  - Full adversarial (1,438 records): **1,146/1,438 (80%)**
- **Takeaway**: More data is the biggest lever. Val clean 67% → 73%, val adversarial 63% → 73%. Distractors no longer hurt on held-out data (73% both ways) — the model generalized to handle noise without distractor training. Overfitting gap similar (92% full vs 73% val) but absolute scores up across the board.

## V6 (5 epochs) — Distractor training (2 per example)
- **Date**: 2026-02-23
- **Model**: budget-mistral-lora-v6-merged (LoRA on Mistral 7B)
- **Training**:
  - 1,378 examples (689 cities x 2 expenses, PyMuPDF pages only)
  - 2 random distractor pages per example (~7,695 avg chars/prompt)
  - 5 epochs, lr=2e-4, batch=2, grad_accum=4, max_seq_len=8192
  - Train/val split: 689 train / 30 validation (same 30 val cities as V3-V5)
  - Loss at epoch 5: ~0.603 (still falling)
- **Inference params**: n-chunks=1 or 5, max_tokens=50, temperature=0
- **Results** (val only, 60 records, 3 parsers x clean/d4):
  - Aryn clean: **52/60 (87%)**, Aryn d4: **48/60 (80%)**
  - PyMuPDF clean: **50/60 (83%)**, PyMuPDF d4: **44/60 (73%)**
  - pdfplumber clean: **41/60 (68%)**, pdfplumber d4: **36/60 (60%)**
- **V5 → V6 comparison**:
  - Aryn: 88→87% clean (-1), 78→80% d4 (+2) — flat
  - PyMuPDF: 78→83% clean (+5), 65→73% d4 (+8) — big improvement
  - pdfplumber: 70→68% clean (-2), 62→60% d4 (-2) — slightly worse
- **Takeaway**: Distractor training helps the parser it was trained on (pymupdf). PyMuPDF d4 gap narrowed from 15% to 10%. Aryn slightly improved on d4. pdfplumber slightly regressed. Loss still falling at epoch 5 — continuing to 10 epochs.

## V6 (10 epochs) — *training in progress*

---

## Controlled variables to track per run
| Variable | V1 | V2 | V3 | V4 | V5 | V6 (5ep) |
|---|---|---|---|---|---|---|
| Base model | Mistral 7B v0.3 | Mistral 7B v0.3 | Mistral 7B v0.3 | Mistral 7B v0.3 | Mistral 7B v0.3 | Mistral 7B v0.3 |
| Fine-tuned | no | yes | yes | yes | yes | yes |
| Training examples | - | 517 | 206 | 206 | 1,378 | 1,378 |
| Training source | - | mixed (PyMuPDF + Aryn) | PyMuPDF pages only | PyMuPDF pages only | PyMuPDF pages only | PyMuPDF pages only |
| Distractors (train) | - | 4 per example | 0 | 4 random pages | 0 | 2 random pages |
| Epochs | - | 3 | 10 | 10 (crashed) | 5 (resumed from 2) | 5 |
| Final loss | - | 0.985 | ~0.34 | ~0.6 (messy) | ~0.70 (epoch 2.3) | ~0.603 |
| n-chunks (test) | 20 | 5 | 1 (target page) | 1 or 5 | 1 or 5 | 1 or 5 |
| max_tokens | none | 50 | 50 | 50 | 50 | 50 |
| Test records | 108 | 108 | 60 (val) + 266 (full) | 60 (val only) | 60 (val) + 1,438 (full) | 60 (val, 3 parsers) |
| Val clean (pymupdf) | - | - | 67% | 67% | 73% | **83%** |
| Val adversarial (pymupdf) | - | - | 63% | 57% | 73% | **73%** |
| Full clean | - | - | 85% | - | 92% | - |
| Full adversarial | - | - | 67% | - | 80% | - |

## Error analysis (V5, 60 val records, all 6 parser/distractor combos)
Spot-checked all 78 misses across 6 runs. Categories:
1. **Wrong scope (~30)**: total vs GF, all-funds vs GF, with/without transfers, revenue vs expenditure
2. **Wrong year/column (~17)**: pulling different FY from multi-year tables, especially with distractors
3. **Proposed vs adopted (~8)**: recommended/proposed instead of council-adopted
4. **Way off / wrong page (~8)**: completely wrong number (school budget, public works line item, etc.)
5. **Parser misread (~7)**: small digit errors from OCR (misread 7→5, 721→700, lost leading 1)
6. **Ambiguous gold truth (~8+)**: genuinely unclear which number is correct (transfers, operating vs total)

## Key takeaways so far
1. **Fine-tuning works**: 49% → 67% → 73% → 83% on held-out pymupdf clean data (base → V3 → V5 → V6)
2. **More data is the biggest lever**: 206 → 1,378 examples gave +6% val clean, +10% val adversarial
3. **Distractor training helps robustness**: V6 pymupdf d4 65→73% (+8pp), clean 78→83% (+5pp)
4. **Aryn is the best parser**: 87% clean / 80% d4 — consistently outperforms pymupdf and pdfplumber
5. **Wrong scope is the #1 error**: model finds a related number at wrong aggregation level (total vs GF, with/without transfers)
6. **Many "misses" are debatable**: ambiguous gold truth (should transfers be included? operating vs final total?)
7. **Next**: test V6 at 10 epochs, consider full-set evaluation
