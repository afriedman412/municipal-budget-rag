"""Test retrieval quality on Marker blocks vs pages.

Parses test PDFs with Marker into blocks, embeds them, and checks
whether embedding-based retrieval finds the block containing the
ground truth budget number.

Usage:
  python test_retrieval.py                          # Default: marker blocks
  python test_retrieval.py --granularity page        # Compare with page-level
  python test_retrieval.py --top-k 5                 # Check top-5 retrieval
  python test_retrieval.py --parser pymupdf          # Use pymupdf pages
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
from openai import OpenAI

from config import PDF_DIR, EMBED_MODEL, EMBED_DIMENSIONS

TRAINING_DIR = Path("training")


def find_pdf(state, city, year):
    state = state.lower()
    city = city.lower().replace(" ", "_")
    yr = str(year)[2:]
    for pdf in PDF_DIR.glob("*.pdf"):
        if pdf.name.startswith(f"{state}_{city}_{yr}"):
            return pdf
    return None


def _pdf_to_query(pdf_path):
    """Extract (state, city, year) from PDF filename for DB lookup.
    Returns args tuple for sqlite query. Uses LIKE for case-insensitive city match."""
    from pipeline.parsers import parse_filename
    state, city, year = parse_filename(pdf_path.name)
    if not state or not city or not year:
        return None
    # DB stores 'CA', 'San Diego' — filename gives 'ca', 'san_diego'
    city_pattern = city.replace("_", " ")  # 'san diego'
    return (state.upper(), city_pattern, year)


def format_budget(amount):
    """Format budget as $X,XXX,XXX for string matching."""
    return f"${amount:,.0f}"


def parse_marker_blocks(pdf_path):
    """Parse a PDF with Marker, returning individual blocks with metadata.
    Returns list of {text, page, block_idx, block_type}."""
    from build_page_cache import _get_marker_converter, _html_to_text

    converter = _get_marker_converter()
    rendered = converter(str(pdf_path))

    blocks = []
    for page_idx, page in enumerate(rendered.children):
        for block_idx, block in enumerate(page.children):
            html = getattr(block, "html", "") or ""
            if not html.strip():
                continue
            text = _html_to_text(html)
            if not text.strip():
                continue
            block_type = getattr(block, "block_type", "unknown")
            blocks.append({
                "text": text.strip(),
                "page": page_idx,
                "block_idx": block_idx,
                "block_type": str(block_type),
            })
    return blocks


def parse_marker_pages(pdf_path):
    """Parse a PDF with Marker, returning per-page text.
    Returns list of {text, page}."""
    from build_page_cache import _get_marker_converter, _html_to_text

    converter = _get_marker_converter()
    rendered = converter(str(pdf_path))

    pages = []
    for page_idx, page in enumerate(rendered.children):
        parts = []
        for block in page.children:
            html = getattr(block, "html", "") or ""
            text = _html_to_text(html) if html.strip() else ""
            if text.strip():
                parts.append(text.strip())
        page_text = "\n\n".join(parts)
        if page_text.strip():
            pages.append({"text": page_text, "page": page_idx})
    return pages


def parse_pymupdf_pages(pdf_path):
    """Parse a PDF with PyMuPDF, returning per-page text."""
    import fitz
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        if text:
            pages.append({"text": text, "page": i})
    doc.close()
    return pages


def embed_texts(client, texts, batch_size=50):
    """Embed a list of texts, returning numpy array of embeddings."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=[t[:2500] for t in batch],
            dimensions=EMBED_DIMENSIONS,
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return np.array(all_embeddings)


def cosine_similarity(query_emb, doc_embs):
    """Compute cosine similarity between query and all doc embeddings."""
    query_norm = query_emb / np.linalg.norm(query_emb)
    doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    return doc_norms @ query_norm


def budget_in_text(budget, text):
    """Check if budget amount appears in text (various formats)."""
    formatted = format_budget(budget)
    if formatted in text:
        return True
    # Also check without $ sign
    plain = f"{budget:,.0f}"
    if plain in text:
        return True
    # Check millions format (e.g., 42.2M or 42,201,781)
    return False


def main():
    parser = argparse.ArgumentParser(description="Test retrieval on Marker blocks")
    parser.add_argument("--granularity", choices=["block", "page"], default="block",
                        help="Chunk granularity (default: block)")
    parser.add_argument("--parser", choices=["marker", "pymupdf"], default="marker",
                        help="Parser to use (default: marker)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of chunks to retrieve (default: 3)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N PDFs (0 = all)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Force page granularity for pymupdf
    if args.parser == "pymupdf":
        args.granularity = "page"

    conn = sqlite3.connect("pipeline_state.db")
    conn.row_factory = sqlite3.Row
    client = OpenAI()

    # Find all local PDFs that have validation records with pdf_page
    from collections import defaultdict
    local_pdfs = list(PDF_DIR.glob("*.pdf"))
    print(f"Local PDFs: {len(local_pdfs)}")

    by_pdf = defaultdict(list)
    skipped = 0
    for pdf_path in local_pdfs:
        query_args = _pdf_to_query(pdf_path)
        if not query_args:
            skipped += 1
            continue
        state, city_pattern, year = query_args
        rows = conn.execute("""
            SELECT state, city, year, expense, budget_type, budget, pdf_page
            FROM validation
            WHERE state = ? AND LOWER(city) = ? AND year = ?
            AND pdf_page IS NOT NULL AND pdf_page > 0
        """, (state, city_pattern, year)).fetchall()
        if not rows:
            skipped += 1
            continue
        for row in rows:
            rec = dict(row)
            rec["pdf_page"] = rec["pdf_page"] - 1  # 0-indexed
            by_pdf[str(pdf_path)].append((rec, pdf_path))

    matched_pdfs = len(by_pdf)
    total_records = sum(len(v) for v in by_pdf.values())
    print(f"PDFs with validation records: {matched_pdfs}")
    print(f"Total retrieval queries: {total_records}")
    if not by_pdf:
        print("No matching PDFs found. Check pdfs_2026/ and pipeline_state.db.")
        return

    # Process each PDF
    results = {"hit": 0, "miss": 0, "total": 0}
    block_type_stats = defaultdict(int)
    pdf_items = sorted(by_pdf.items())
    if args.limit:
        pdf_items = pdf_items[:args.limit]
        print(f"Limited to {args.limit} PDFs")

    for pdf_i, (pdf_key, records) in enumerate(pdf_items):
        pdf_path = records[0][1]
        pdf_name = pdf_path.name

        # Parse into chunks
        if args.parser == "marker":
            if args.granularity == "block":
                chunks = parse_marker_blocks(pdf_path)
            else:
                chunks = parse_marker_pages(pdf_path)
        else:
            chunks = parse_pymupdf_pages(pdf_path)

        n_chunks = len(chunks)
        print(f"[{pdf_i+1}/{len(pdf_items)}] {pdf_name}: {n_chunks} chunks")

        if not chunks:
            continue

        # Embed all chunks for this PDF
        texts = [c["text"] for c in chunks]
        doc_embs = embed_texts(client, texts)

        # Test retrieval for each record in this PDF
        for rec, _ in records:
            expense = rec["expense"]
            budget = rec["budget"]
            target_page = rec["pdf_page"]
            query = f"{expense} expenditure budget total for {rec['city']} {rec['year']}"

            # Embed query
            q_emb = embed_texts(client, [query])[0]
            sims = cosine_similarity(q_emb, doc_embs)
            top_indices = np.argsort(sims)[::-1][:args.top_k]

            # Check if any top-k chunk contains the budget number
            hit = False
            for idx in top_indices:
                if budget_in_text(budget, chunks[idx]["text"]):
                    hit = True
                    break

            results["total"] += 1
            if hit:
                results["hit"] += 1
            else:
                results["miss"] += 1

            if args.verbose or not hit:
                status = "HIT" if hit else "MISS"
                print(f"  [{status}] {pdf_name} | {expense} | {format_budget(budget)} | target page {target_page}")
                if args.verbose:
                    for rank, idx in enumerate(top_indices):
                        c = chunks[idx]
                        has_answer = budget_in_text(budget, c["text"])
                        marker = " <-- ANSWER" if has_answer else ""
                        page_info = f"p{c['page']}"
                        type_info = f" [{c.get('block_type', 'page')}]" if 'block_type' in c else ""
                        print(f"    #{rank+1} (sim={sims[idx]:.3f}) {page_info}{type_info} "
                              f"({len(c['text'])} chars){marker}")
                        if has_answer or rank == 0:
                            preview = c["text"][:200].replace("\n", " ")
                            print(f"       {preview}...")

            # Track block types of target chunks
            if args.granularity == "block":
                for c in chunks:
                    if c["page"] == target_page and budget_in_text(budget, c["text"]):
                        block_type_stats[c.get("block_type", "unknown")] += 1

    # Summary
    hit_rate = results["hit"] / results["total"] * 100 if results["total"] else 0
    print(f"\n{'='*60}")
    print(f"Parser: {args.parser}, Granularity: {args.granularity}, Top-k: {args.top_k}")
    print(f"Hit: {results['hit']}/{results['total']} ({hit_rate:.0f}%)")
    print(f"Miss: {results['miss']}/{results['total']}")

    if block_type_stats:
        print(f"\nTarget block types:")
        for bt, count in sorted(block_type_stats.items(), key=lambda x: -x[1]):
            print(f"  {bt}: {count}")


if __name__ == "__main__":
    main()
