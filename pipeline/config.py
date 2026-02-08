"""Pipeline configuration from environment variables."""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Config:
    # S3
    s3_bucket: str
    s3_prefix: str

    # API Keys
    aryn_api_key: str
    openai_api_key: str

    # ChromaDB
    chroma_host: str
    chroma_api_key: str
    chroma_collection: str

    # Processing
    batch_size: int
    aryn_async: bool  # Use async API for parallel parsing (PAYG only)
    embed_model: str
    embed_dimensions: int

    # Local
    temp_dir: Path
    state_db: Path

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            # S3
            s3_bucket=os.environ["S3_BUCKET"],
            s3_prefix=os.getenv("S3_PREFIX", ""),

            # API Keys
            aryn_api_key=os.environ["ARYN_API_KEY"],
            openai_api_key=os.environ["OPENAI_API_KEY"],

            # ChromaDB
            chroma_host=os.getenv("CHROMA_HOST", ""),
            chroma_api_key=os.getenv("CHROMA_API_KEY", ""),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "municipal-budgets"),

            # Processing
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            aryn_async=os.getenv("ARYN_ASYNC", "").lower() in ("1", "true", "yes"),
            embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
            embed_dimensions=int(os.getenv("EMBED_DIMENSIONS", "1536")),

            # Local
            temp_dir=Path(os.getenv("TEMP_DIR", "/tmp/budget_pipeline")),
            state_db=Path(os.getenv("STATE_DB", "pipeline_state.db")),
        )
