"""Pipeline configuration."""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class PipelineConfig:
    # S3 settings
    s3_bucket: str = ""
    s3_prefix: str = ""  # e.g., "budgets/pdfs/"

    # Processing settings
    pdf_workers: int = 8              # parallel PDF extraction processes
    embed_batch_size: int = 50        # texts per embedding API call
    embed_concurrency: int = 10       # concurrent embedding requests
    max_retries: int = 3              # retries per failed job

    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ChromaDB settings (cloud)
    chroma_host: str = ""             # e.g., "api.trychroma.com"
    chroma_api_key: str = ""
    chroma_collection: str = "municipal-budgets"

    # State tracking
    state_db_path: Path = field(default_factory=lambda: Path("pipeline_state.db"))

    # Local temp storage for PDFs
    temp_dir: Path = field(default_factory=lambda: Path("/tmp/pdf_pipeline"))

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load config from environment variables."""
        return cls(
            s3_bucket=os.getenv("S3_BUCKET", ""),
            s3_prefix=os.getenv("S3_PREFIX", ""),
            chroma_host=os.getenv("CHROMA_HOST", ""),
            chroma_api_key=os.getenv("CHROMA_API_KEY", ""),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "municipal-budgets"),
            pdf_workers=int(os.getenv("PDF_WORKERS", "8")),
            embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "50")),
            embed_concurrency=int(os.getenv("EMBED_CONCURRENCY", "10")),
        )
