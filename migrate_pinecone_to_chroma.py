#!/usr/bin/env python3
"""
Migrate vectors from Pinecone to ChromaDB.

Fetches all vectors (including embeddings and metadata) from Pinecone
and inserts them into a local ChromaDB instance.

Usage:
    python migrate_pinecone_to_chroma.py

Required env vars:
    PINECONE_API_KEY
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
PINECONE_INDEX_NAME = "municipal-budgets"
CHROMA_PATH = "./chroma_data"
CHROMA_COLLECTION_NAME = "municipal-budgets"

# Batch size for fetching from Pinecone
FETCH_BATCH_SIZE = 100
# =============================================================================


def migrate():
    """Migrate all vectors from Pinecone to ChromaDB."""

    # Initialize Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("Error: PINECONE_API_KEY not set in environment.")
        return

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # Get index stats
    stats = pinecone_index.describe_index_stats()
    total_vectors = stats.total_vector_count
    print(f"Pinecone index has {total_vectors:,} vectors")

    if total_vectors == 0:
        print("No vectors to migrate.")
        return

    # Initialize ChromaDB
    print(f"Initializing ChromaDB at {CHROMA_PATH}...")
    chroma_path = Path(CHROMA_PATH)
    chroma_path.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))

    # Delete existing collection if it exists (for clean migration)
    try:
        chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"Deleted existing collection '{CHROMA_COLLECTION_NAME}'")
    except Exception:
        pass  # Collection doesn't exist

    collection = chroma_client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"description": "Municipal budget documents"}
    )
    print(f"Created ChromaDB collection '{CHROMA_COLLECTION_NAME}'")

    # Migrate vectors
    print("\nMigrating vectors...")
    migrated = 0
    errors = 0

    # Use Pinecone's list operation to get all vector IDs
    # This returns a generator of ID lists
    with tqdm(total=total_vectors, desc="Migrating") as pbar:
        for id_batch in pinecone_index.list():
            if not id_batch:
                continue

            # Fetch vectors with their values and metadata
            try:
                fetched = pinecone_index.fetch(ids=list(id_batch))
            except Exception as e:
                tqdm.write(f"Error fetching batch: {e}")
                errors += len(id_batch)
                pbar.update(len(id_batch))
                continue

            if not fetched.vectors:
                pbar.update(len(id_batch))
                continue

            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for vec_id, vec_data in fetched.vectors.items():
                ids.append(vec_id)
                embeddings.append(vec_data.values)

                # Extract metadata
                meta = dict(vec_data.metadata) if vec_data.metadata else {}

                # ChromaDB stores documents separately from metadata
                # Extract text for the document field
                doc_text = meta.pop("text", "")
                documents.append(doc_text)

                # Ensure metadata values are valid types for ChromaDB
                # (str, int, float, bool only - no None values)
                clean_meta = {}
                for k, v in meta.items():
                    if v is None:
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)

                metadatas.append(clean_meta)

            # Insert into ChromaDB
            try:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                migrated += len(ids)
            except Exception as e:
                tqdm.write(f"Error inserting batch: {e}")
                errors += len(ids)

            pbar.update(len(id_batch))

    # Summary
    print("\n" + "=" * 50)
    print("MIGRATION COMPLETE")
    print("=" * 50)
    print(f"Vectors migrated: {migrated:,}")
    print(f"Errors: {errors}")
    print(f"ChromaDB path: {chroma_path.absolute()}")

    # Verify
    print("\nVerifying...")
    chroma_count = collection.count()
    print(f"ChromaDB collection count: {chroma_count:,}")

    if chroma_count == total_vectors:
        print("Migration verified successfully!")
    else:
        print(f"Warning: Count mismatch (Pinecone: {total_vectors}, Chroma: {chroma_count})")


if __name__ == "__main__":
    migrate()
