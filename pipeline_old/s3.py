"""S3 operations for PDF pipeline."""

import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from .config import PipelineConfig


class S3Client:
    """S3 client for listing and downloading PDFs."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.download_workers = 10  # Parallel download threads

        # Set pool size larger than workers to avoid connection churn
        boto_config = Config(max_pool_connections=self.download_workers + 5)
        self.client = boto3.client("s3", config=boto_config)

        self.bucket = config.s3_bucket
        self.prefix = config.s3_prefix

        # Ensure temp directory exists
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)

    def list_pdfs(self) -> list[str]:
        """List all PDF keys in the bucket/prefix."""
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith(".pdf"):
                    keys.append(key)

        return keys

    def download_pdf(self, s3_key: str) -> Path:
        """Download a PDF to local temp storage. Returns local path."""
        # Create local path preserving some structure
        filename = Path(s3_key).name
        local_path = self.config.temp_dir / filename

        # Download if not already cached
        if not local_path.exists():
            self.client.download_file(self.bucket, s3_key, str(local_path))

        return local_path

    def download_pdf_bytes(self, s3_key: str) -> bytes:
        """Download PDF directly to memory. Use for smaller files."""
        response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
        return response["Body"].read()

    def iter_pdfs(self, keys: list[str]) -> Iterator[tuple[str, Path]]:
        """Iterate over PDFs, downloading each. Yields (s3_key, local_path)."""
        for key in keys:
            local_path = self.download_pdf(key)
            yield key, local_path

    def download_batch(self, s3_keys: list[str]) -> list[tuple[str, Path | None, str | None]]:
        """
        Download multiple PDFs in parallel.
        Returns: [(s3_key, local_path or None, error or None), ...]
        """
        results = []

        def download_one(key: str) -> tuple[str, Path | None, str | None]:
            try:
                path = self.download_pdf(key)
                return (key, path, None)
            except Exception as e:
                return (key, None, str(e))

        with ThreadPoolExecutor(max_workers=self.download_workers) as executor:
            futures = {executor.submit(download_one, key): key for key in s3_keys}
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def cleanup_local(self, local_path: Path):
        """Remove a local PDF after processing."""
        if local_path.exists():
            local_path.unlink()

    def cleanup_all(self):
        """Remove all cached PDFs."""
        for pdf in self.config.temp_dir.glob("*.pdf"):
            pdf.unlink()
