"""Simple SQLite state tracking for pipeline jobs."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class Status(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    s3_key: str
    status: Status
    error: str | None
    created_at: datetime
    updated_at: datetime


class StateDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    s3_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)")
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def register_jobs(self, s3_keys: list[str]) -> int:
        """Register new jobs. Returns count added."""
        added = 0
        with self._connect() as conn:
            for key in s3_keys:
                try:
                    conn.execute("INSERT INTO jobs (s3_key) VALUES (?)", (key,))
                    added += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        return added

    def get_pending(self, limit: int | None = None) -> list[str]:
        """Get pending job keys."""
        with self._connect() as conn:
            if limit:
                rows = conn.execute(
                    "SELECT s3_key FROM jobs WHERE status = 'pending' ORDER BY created_at LIMIT ?",
                    (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT s3_key FROM jobs WHERE status = 'pending' ORDER BY created_at"
                ).fetchall()
        return [r["s3_key"] for r in rows]

    def mark_processing(self, s3_key: str):
        """Mark job as processing."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'processing', updated_at = CURRENT_TIMESTAMP WHERE s3_key = ?",
                (s3_key,)
            )
            conn.commit()

    def mark_done(self, s3_key: str):
        """Mark job as done."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'done', error = NULL, updated_at = CURRENT_TIMESTAMP WHERE s3_key = ?",
                (s3_key,)
            )
            conn.commit()

    def mark_failed(self, s3_key: str, error: str):
        """Mark job as failed."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, updated_at = CURRENT_TIMESTAMP WHERE s3_key = ?",
                (error, s3_key)
            )
            conn.commit()

    def reset_failed(self) -> int:
        """Reset failed jobs to pending."""
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE jobs SET status = 'pending', error = NULL, updated_at = CURRENT_TIMESTAMP WHERE status = 'failed'"
            )
            conn.commit()
            return result.rowcount

    def reset_processing(self) -> int:
        """Reset stuck processing jobs to pending."""
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE jobs SET status = 'pending', updated_at = CURRENT_TIMESTAMP WHERE status = 'processing'"
            )
            conn.commit()
            return result.rowcount

    def get_stats(self) -> dict:
        """Get job counts by status."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as count FROM jobs GROUP BY status"
            ).fetchall()
        stats = {r["status"]: r["count"] for r in rows}
        stats["total"] = sum(stats.values())
        return stats

    def get_failed(self, limit: int = 50) -> list[Job]:
        """Get failed jobs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = 'failed' ORDER BY updated_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def _row_to_job(self, row) -> Job:
        return Job(
            s3_key=row["s3_key"],
            status=Status(row["status"]),
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )
