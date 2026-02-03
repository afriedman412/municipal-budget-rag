"""ChromaDB client for cloud or local storage."""

import chromadb
from chromadb.config import Settings

from .config import PipelineConfig
from .embed import EmbeddedChunk


class ChromaClient:
    """Client for storing embeddings in ChromaDB."""

    def __init__(self, config: PipelineConfig):
        self.config = config

        if config.chroma_host and config.chroma_api_key:
            # Cloud ChromaDB
            self.client = chromadb.HttpClient(
                host=config.chroma_host,
                port=443,
                ssl=True,
                headers={"Authorization": f"Bearer {config.chroma_api_key}"}
            )
        elif config.chroma_host:
            # Self-hosted ChromaDB (no auth)
            host, _, port = config.chroma_host.partition(":")
            self.client = chromadb.HttpClient(
                host=host,
                port=int(port) if port else 8000,
            )
        else:
            # Local ChromaDB (for testing)
            self.client = chromadb.PersistentClient(
                path="./chroma_data",
                settings=Settings(anonymized_telemetry=False)
            )

        self.collection = self.client.get_or_create_collection(
            name=config.chroma_collection,
            metadata={"description": "Municipal budget documents"}
        )

    def add_chunks(self, chunks: list[EmbeddedChunk], batch_size: int = 5000) -> int:
        """Add embedded chunks to the collection. Returns count added."""
        if not chunks:
            return 0

        # Batch to avoid ChromaDB's max batch size limit (5461)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            ids = [c.chunk.chunk_id for c in batch]
            embeddings = [c.embedding for c in batch]
            documents = [c.chunk.text for c in batch]
            metadatas = [
                {
                    "source": c.chunk.filename,
                    "s3_key": c.chunk.s3_key,
                    "page": c.chunk.page_num,
                    "chunk": c.chunk.chunk_idx,
                    "city": c.chunk.city,
                    "state": c.chunk.state,
                    "year": c.chunk.year,
                }
                for c in batch
            ]

            # ChromaDB handles duplicates by ID
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        return len(chunks)

    def chunk_exists(self, chunk_id: str) -> bool:
        """Check if a chunk already exists."""
        try:
            result = self.collection.get(ids=[chunk_id])
            return len(result["ids"]) > 0
        except Exception:
            return False

    def document_exists(self, filename: str) -> bool:
        """Check if any chunks from a document exist."""
        try:
            result = self.collection.get(
                where={"source": filename},
                limit=1
            )
            return len(result["ids"]) > 0
        except Exception:
            return False

    def get_stats(self) -> dict:
        """Get collection statistics."""
        count = self.collection.count()

        # Get unique sources
        try:
            all_meta = self.collection.get(include=["metadatas"])
            sources = set(m.get("source", "") for m in all_meta["metadatas"])
            states = set(m.get("state", "") for m in all_meta["metadatas"])
        except Exception:
            sources = set()
            states = set()

        return {
            "total_chunks": count,
            "unique_documents": len(sources),
            "states": sorted(states),
        }
