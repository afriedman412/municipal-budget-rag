"""Async S3 operations."""

import asyncio
import aioboto3
from pathlib import Path

from .config import Config


class S3Client:
    def __init__(self, config: Config):
        self.config = config
        self.session = aioboto3.Session()
        config.temp_dir.mkdir(parents=True, exist_ok=True)

    async def list_pdfs(self) -> list[str]:
        """List all PDF keys in bucket/prefix."""
        keys = []
        async with self.session.client("s3") as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self.config.s3_bucket,
                Prefix=self.config.s3_prefix
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.lower().endswith(".pdf"):
                        keys.append(key)
        return keys

    async def download(self, s3_key: str) -> Path:
        """Download a PDF to temp directory."""
        filename = Path(s3_key).name
        local_path = self.config.temp_dir / filename

        if not local_path.exists():
            async with self.session.client("s3") as s3:
                await s3.download_file(
                    self.config.s3_bucket,
                    s3_key,
                    str(local_path)
                )
        return local_path

    async def download_batch(self, keys: list[str]) -> list[tuple[str, Path | None, str | None]]:
        """Download multiple PDFs concurrently.

        Returns: [(s3_key, local_path or None, error or None), ...]
        """
        async def download_one(key: str):
            try:
                path = await self.download(key)
                return (key, path, None)
            except Exception as e:
                return (key, None, str(e))

        return await asyncio.gather(*[download_one(k) for k in keys])

    def cleanup(self, path: Path):
        """Remove a downloaded file."""
        if path.exists():
            path.unlink()
