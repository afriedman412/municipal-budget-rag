#!/usr/bin/env python3
"""
Ingest budget PDFs into a vector database for RAG.

Extracts text from PDFs, chunks by page, embeds with OpenAI,
and stores in either Pinecone or ChromaDB with metadata.

Usage:
    python ingest_pdfs.py

Configure VECTOR_DB and other settings below, then run.

Required env vars:
    OPENAI_API_KEY
    PINECONE_API_KEY (only if using Pinecone)
"""

import hashlib
import os
import re
from pathlib import Path

import chromadb
import fitz  # pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Vector database: "chroma" or "pinecone"
VECTOR_DB = "chroma"

# PDF source folder
PDF_FOLDER = "/Users/user/Documents/code/sycamore_scrap/municipal-budget-rag/pdfs_2026/selects"

# ChromaDB settings (used if VECTOR_DB = "chroma")
CHROMA_PATH = "./chroma_data"
CHROMA_COLLECTION_NAME = "municipal-budgets"

# Pinecone settings (used if VECTOR_DB = "pinecone")
PINECONE_INDEX_NAME = "municipal-budgets"

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Chunking settings
CHUNK_BY = "page"  # "page" or "tokens"
MAX_CHUNK_TOKENS = 500  # Only used if CHUNK_BY = "tokens"

# PDFs to skip (known to crash or hang PyMuPDF)
SKIP_PDFS = {
    "tx_texarkana_23.pdf",  # Crashes PyMuPDF (segfault)
    "tx_mesquite_21.pdf",   # Hangs/timeout during extraction
}
# =============================================================================


def extract_text_by_page(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF."""
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        if text:  # Skip empty pages
            pages.append({
                "page": page_num + 1,
                "text": text
            })

    doc.close()
    return pages


def parse_filename(filename: str) -> dict:
    """
    Parse metadata from filename.
    Expected format: city_state_year.pdf (e.g., aberdeen_md_22.pdf)
    """
    stem = Path(filename).stem.lower()
    # Remove common suffixes like _proposed, _approved, _pt2, etc.
    stem = re.sub(r'_(proposed|approved|appr|mayor|pt\d+|copy)$', '', stem)
    parts = stem.split("_")

    metadata = {"raw_filename": filename}

    if len(parts) >= 3:
        # Find the state (2-letter code) - usually second to last
        state_idx = None
        for i, part in enumerate(parts):
            if len(part) == 2 and part.isalpha():
                state_idx = i
                break

        if state_idx is not None:
            metadata["state"] = parts[state_idx].upper()
            metadata["city"] = " ".join(parts[:state_idx]).title()
            # Year is everything after state
            year_parts = parts[state_idx + 1:]
            if year_parts:
                year_str = year_parts[0]
                # Handle formats like "21_22" or just "22"
                if len(year_str) == 2 and year_str.isdigit():
                    year_int = int(year_str)
                    metadata["year"] = 2000 + year_int
                elif len(year_str) == 4 and year_str.isdigit():
                    metadata["year"] = int(year_str)
                else:
                    metadata["year"] = 0
            else:
                metadata["year"] = 0
        else:
            metadata["state"] = "Unknown"
            metadata["city"] = " ".join(parts[:-1]).title()
            metadata["year"] = 0
    else:
        metadata["state"] = "Unknown"
        metadata["city"] = stem.replace("_", " ").title()
        metadata["year"] = 0

    return metadata


def create_chunk_id(filename: str, page: int, chunk_idx: int = 0) -> str:
    """Create a unique ID for a chunk."""
    content = f"{filename}:p{page}:c{chunk_idx}"
    return hashlib.md5(content.encode()).hexdigest()


def chunk_text(text: str, max_tokens: int = 2000) -> list[str]:
    """Split text into chunks that fit within token limit."""
    # Conservative estimate: 1 token ≈ 3 characters (budget docs have lots of numbers)
    max_chars = max_tokens * 3

    if len(text) <= max_chars:
        return [text]

    chunks = []
    words = text.split()
    current_chunk = []
    current_len = 0

    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_len + word_len > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = word_len
        else:
            current_chunk.append(word)
            current_len += word_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_embeddings(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Get embeddings for a batch of texts."""
    # Truncate texts to avoid exceeding token limit (~8191 for text-embedding-3-small)
    # Conservative estimate: 1 token ≈ 3 chars for budget docs with numbers
    truncated = [t[:24000] for t in texts]
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=truncated
    )
    return [item.embedding for item in response.data]


# =============================================================================
# Pinecone functions
# =============================================================================

def init_pinecone() -> Pinecone:
    """Initialize Pinecone and create index if needed."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created.")
    else:
        print(f"Using existing index '{PINECONE_INDEX_NAME}'")

    return pc


def is_already_ingested_pinecone(index, filename: str) -> bool:
    """Check if a PDF has already been ingested (Pinecone)."""
    chunk_id = create_chunk_id(filename, page=1, chunk_idx=0)
    try:
        result = index.fetch(ids=[chunk_id])
        return len(result.vectors) > 0
    except Exception:
        return False


def get_pinecone_stats(index) -> int:
    """Get vector count from Pinecone."""
    try:
        stats = index.describe_index_stats()
        total = stats.total_vector_count
        print(f"Pinecone index has {total:,} vectors")
        return total
    except Exception:
        return 0


# =============================================================================
# ChromaDB functions
# =============================================================================

def init_chroma():
    """Initialize ChromaDB and create/get collection."""
    chroma_path = Path(CHROMA_PATH)
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"description": "Municipal budget documents"}
    )

    print(
        f"Using ChromaDB collection '{CHROMA_COLLECTION_NAME}' at {chroma_path}")
    return client, collection


def is_already_ingested_chroma(collection, filename: str) -> bool:
    """Check if a PDF has already been ingested (ChromaDB)."""
    chunk_id = create_chunk_id(filename, page=1, chunk_idx=0)
    try:
        result = collection.get(ids=[chunk_id])
        return len(result["ids"]) > 0
    except Exception:
        return False


def get_chroma_stats(collection) -> int:
    """Get vector count from ChromaDB."""
    try:
        total = collection.count()
        print(f"ChromaDB collection has {total:,} vectors")
        return total
    except Exception:
        return 0


# =============================================================================
# Main ingestion
# =============================================================================

def ingest_pdfs(pdf_folder: str, skip_existing: bool = True):
    """Main ingestion function."""
    pdf_folder = Path(pdf_folder)

    if not pdf_folder.exists():
        print(f"Error: Folder not found: {pdf_folder}")
        return

    pdf_files = list(pdf_folder.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    if not pdf_files:
        return

    # Initialize OpenAI client
    openai_client = OpenAI()

    # Initialize vector database
    if VECTOR_DB == "pinecone":
        pc = init_pinecone()
        index = pc.Index(PINECONE_INDEX_NAME)
        get_pinecone_stats(index)
        def is_ingested(f): return is_already_ingested_pinecone(index, f)
    elif VECTOR_DB == "chroma":
        _, collection = init_chroma()
        get_chroma_stats(collection)
        def is_ingested(f): return is_already_ingested_chroma(collection, f)
    else:
        print(
            f"Error: Unknown VECTOR_DB '{VECTOR_DB}'. Use 'pinecone' or 'chroma'.")
        return

    # Track stats
    total_chunks = 0
    total_tokens_approx = 0
    skipped = 0

    # Process each PDF
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        # Skip known problematic PDFs
        if pdf_path.name in SKIP_PDFS:
            tqdm.write(f"  Skipping {pdf_path.name} (known to crash/hang)")
            skipped += 1
            continue

        # Check if already ingested
        if skip_existing and is_ingested(pdf_path.name):
            tqdm.write(f"  Skipping {pdf_path.name} (already ingested)")
            skipped += 1
            continue

        file_metadata = parse_filename(pdf_path.name)
        pages = extract_text_by_page(str(pdf_path))

        if not pages:
            tqdm.write(f"  Skipping {pdf_path.name} (no text extracted)")
            continue

        # Prepare chunks for this PDF (split long pages)
        chunks = []
        for page_data in pages:
            text_chunks = chunk_text(page_data["text"])
            for chunk_idx, text in enumerate(text_chunks):
                chunk_id = create_chunk_id(
                    pdf_path.name, page_data["page"], chunk_idx
                )
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "metadata": {
                        **file_metadata,
                        "page": page_data["page"],
                        "chunk": chunk_idx,
                        "source": pdf_path.name
                    }
                })

        # Batch embed and upsert (smaller batches to avoid token limits)
        batch_size = 10  # Keep small to avoid token limits
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]

            try:
                # Get embeddings
                embeddings = get_embeddings(texts, openai_client)

                if VECTOR_DB == "pinecone":
                    # Prepare vectors for Pinecone
                    vectors = []
                    for chunk, embedding in zip(batch, embeddings):
                        vectors.append({
                            "id": chunk["id"],
                            "values": embedding,
                            "metadata": {
                                **chunk["metadata"],
                                # Truncated for storage
                                "text": chunk["text"][:1000]
                            }
                        })
                    index.upsert(vectors=vectors)

                elif VECTOR_DB == "chroma":
                    # Prepare data for ChromaDB
                    ids = [c["id"] for c in batch]
                    documents = [c["text"] for c in batch]
                    metadatas = []
                    for c in batch:
                        # Clean metadata for ChromaDB (no None values)
                        meta = {}
                        for k, v in c["metadata"].items():
                            if v is not None and isinstance(v, (str, int, float, bool)):
                                meta[k] = v
                        metadatas.append(meta)

                    collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )

                total_chunks += len(batch)
                total_tokens_approx += sum(len(t.split()) * 1.3 for t in texts)

            except Exception as e:
                tqdm.write(f"  Error on batch: {e}")

        tqdm.write(f"  {pdf_path.name}: {len(chunks)} chunks indexed")

    print(f"\nIngestion complete!")
    print(f"  Vector DB: {VECTOR_DB}")
    print(f"  PDFs skipped (already ingested): {skipped}")
    print(f"  New chunks added: {total_chunks}")
    print(f"  Approx tokens embedded: {int(total_tokens_approx):,}")
    print(
        f"  Approx embedding cost: ${total_tokens_approx / 1_000_000 * 0.02:.4f}")


def main():
    if PDF_FOLDER == "/path/to/your/pdfs":
        print("Error: Please set PDF_FOLDER in the script configuration.")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment.")
        return

    if VECTOR_DB == "pinecone" and not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not set in environment.")
        return

    ingest_pdfs(PDF_FOLDER)


if __name__ == "__main__":
    main()
