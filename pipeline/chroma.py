"""ChromaDB client for vector storage."""

import chromadb
from chromadb.config import Settings

from .config import Config
from .embed import Chunk


class ChromaClient:
    def __init__(self, config: Config):
        self.config = config

        if config.chroma_host:
            # Cloud or remote ChromaDB
            self.client = chromadb.HttpClient(
                host=config.chroma_host,
                headers={"Authorization": f"Bearer {config.chroma_api_key}"}
                if config.chroma_api_key else None
            )
        else:
            # Local persistent ChromaDB
            self.client = chromadb.PersistentClient(
                path="chroma_data",
                settings=Settings(anonymized_telemetry=False)
            )

        self.collection = self.client.get_or_create_collection(
            name=config.chroma_collection,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Chunk]):
        """Add embedded chunks to the collection."""
        if not chunks:
            return

        # ChromaDB batch limit is 5461
        batch_size = 5000

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            self.collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=[c.embedding for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{
                    "s3_key": c.s3_key,
                    "filename": c.filename,
                    "state": c.state or "",
                    "city": c.city or "",
                    "year": c.year or 0,
                    "chunk_index": c.chunk_index,
                    "has_table": c.has_table,
                    "has_summary": c.has_summary,
                    "parser": c.parser or "",
                } for c in batch]
            )

    def query(self, text: str, n_results: int = 10,
              where: dict | None = None):
        """Query the collection."""
        return self.collection.query(
            query_texts=[text],
            n_results=n_results,
            where=where,
        )

    def query_budget(
        self,
        query_embedding: list[float],
        city: str,
        year: int,
        n_results: int = 20,
        keywords: list[str] | None = None,
        state: str | None = None,
    ) -> dict:
        """Smart budget query: searches year AND year+1,
        retrieves more chunks, and optionally filters by
        keywords in the chunk text.

        Returns combined results from both years.
        """
        where_doc = None
        if keywords:
            if len(keywords) == 1:
                where_doc = {"$contains": keywords[0]}
            else:
                where_doc = {"$and": [
                    {"$contains": kw} for kw in keywords
                ]}

        all_docs = []
        all_meta = []
        all_dist = []

        for y in [year, year + 1]:
            conditions = [
                {"city": city.lower()},
                {"year": y},
            ]
            if state:
                conditions.append({"state": state.lower()})
            where = {"$and": conditions}

            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where,
                    where_document=where_doc,
                )
            except Exception:
                # No matching docs for this year
                continue

            docs = results.get("documents", [[]])[0]
            meta = results.get("metadatas", [[]])[0]
            dist = results.get("distances", [[]])[0]
            all_docs.extend(docs)
            all_meta.extend(meta)
            all_dist.extend(dist)

        return {
            "documents": [all_docs],
            "metadatas": [all_meta],
            "distances": [all_dist],
        }

    def get_stats(self) -> dict:
        """Get collection statistics."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection": self.config.chroma_collection,
        }
