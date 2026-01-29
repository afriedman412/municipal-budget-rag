#!/usr/bin/env python3
"""Check which PDFs are already ingested in ChromaDB."""

import chromadb
from pathlib import Path

CHROMA_PATH = "./chroma_data"
CHROMA_COLLECTION_NAME = "municipal-budgets"
PDF_FOLDER = "/Users/user/Documents/code/sycamore_scrap/municipal-budget-rag/pdfs_2026/selects"


def main():
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        print(f"Collection '{CHROMA_COLLECTION_NAME}' not found.")
        return

    # Get all unique source files in the DB
    results = collection.get(include=["metadatas"])

    ingested_files = set()
    for meta in results["metadatas"]:
        if meta and "source" in meta:
            ingested_files.add(meta["source"])

    print(f"Total vectors in DB: {collection.count()}")
    print(f"Unique PDFs ingested: {len(ingested_files)}")

    # Compare with PDFs on disk
    pdf_folder = Path(PDF_FOLDER)
    all_pdfs = {p.name for p in pdf_folder.glob("*.pdf")}

    not_ingested = all_pdfs - ingested_files

    print(f"\nPDFs on disk: {len(all_pdfs)}")
    print(f"Not yet ingested: {len(not_ingested)}")

    if not_ingested and len(not_ingested) <= 20:
        print("\nMissing PDFs:")
        for name in sorted(not_ingested):
            print(f"  - {name}")
    elif not_ingested:
        print(f"\n(Too many to list - run with --list to see all)")


if __name__ == "__main__":
    main()
