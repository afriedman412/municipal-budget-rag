"""Process local PDFs through parser → OpenAI → ChromaDB (bypasses S3)."""

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from config import PDF_DIR, PARSER_TEST_PDFS
from pipeline.config import Config
from pipeline.parsers import get_parser
from pipeline.embed import EmbeddingClient, document_to_chunks
from pipeline.chroma import ChromaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

async def main():
    ap = argparse.ArgumentParser(description="Process local PDFs")
    ap.add_argument("--parser", choices=["aryn", "pymupdf", "pdfplumber", "llamaparse", "unstructured"], default="aryn",
                    help="PDF parser to use (default: aryn)")
    ap.add_argument("files", nargs="*", help="PDF filenames (default: PARSER_TEST_PDFS from config.py)")
    args = ap.parse_args()

    config = Config.from_env()
    # Route to parser-specific ChromaDB collection
    config.chroma_collection = f"budgets-{args.parser}"
    logger.info(f"ChromaDB collection: {config.chroma_collection}")
    parser = get_parser(args.parser, config)
    embedder = EmbeddingClient(config)
    chroma = ChromaClient(config)

    file_list = args.files if args.files else PARSER_TEST_PDFS

    # Build list of (s3_key, local_path)
    items = []
    for f in file_list:
        path = Path(f)
        if not path.exists():
            path = PDF_DIR / f
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
