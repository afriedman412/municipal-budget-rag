"""Process local PDFs through parser → OpenAI → ChromaDB (bypasses S3)."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from paths import PDF_DIR
from pipeline.config import Config
from pipeline.embed import EmbeddingClient, document_to_chunks
from pipeline.chroma import ChromaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Batch 3b: 3 more text-heavy PDFs to hit 50 total
FILES = [
    "ny_peekskill_19.pdf",
    "wa_walla_walla_19_20.pdf",
    "wa_spokane_20.pdf",
]


def get_parser(name: str, config):
    """Get a parser client by name."""
    if name == "aryn":
        from pipeline.aryn import ArynClient
        return ArynClient(config)
    elif name == "pymupdf":
        from pipeline.pymupdf import PyMuPDFClient
        return PyMuPDFClient(config)
    else:
        raise ValueError(f"Unknown parser: {name}. Use 'aryn' or 'pymupdf'.")


async def main():
    ap = argparse.ArgumentParser(description="Process local PDFs")
    ap.add_argument("--parser", choices=["aryn", "pymupdf"], default="aryn",
                    help="PDF parser to use (default: aryn)")
    ap.add_argument("files", nargs="*", help="PDF filenames (default: FILES list)")
    args = ap.parse_args()

    config = Config.from_env()
    parser = get_parser(args.parser, config)
    embedder = EmbeddingClient(config)
    chroma = ChromaClient(config)

    file_list = args.files if args.files else FILES

    # Build list of (s3_key, local_path)
    items = []
    for f in file_list:
        path = PDF_DIR / f if not Path(f).is_absolute() else Path(f)
        if not path.exists():
            logger.error(f"Missing: {path}")
            continue
        items.append((path.name, path))

    logger.info(f"Processing {len(items)} PDFs with {args.parser}...")

    # Parse all
    parse_results = await parser.parse_batch(items)

    docs = []
    for (s3_key, path), (doc, error) in zip(items, parse_results):
        if error:
            logger.error(f"PARSE FAIL {s3_key}: {error}")
        else:
            logger.info(f"PARSED {s3_key}: {len(doc.text)} chars")
            docs.append(doc)

    if not docs:
        logger.error("No documents parsed successfully")
        return

    # Embed all documents
    logger.info(f"Embedding {len(docs)} documents...")

    all_chunks = []
    for doc in docs:
        chunks = document_to_chunks(doc, parser=args.parser)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")
    await embedder.embed_chunks(all_chunks)

    # Index to ChromaDB
    logger.info("Indexing to ChromaDB...")
    chroma.add_chunks(all_chunks)

    # Summary
    logger.info(f"Done! {len(docs)}/{len(items)} docs, {len(all_chunks)} chunks indexed")
    stats = chroma.get_stats()
    logger.info(f"ChromaDB total: {stats['total_chunks']} chunks")


if __name__ == "__main__":
    asyncio.run(main())
