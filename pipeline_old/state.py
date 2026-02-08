"""
SQLite-based pipeline state tracking.

Tracks each PDF through stages: pending → extracted → embedded → done
Records failures with error messages for debugging.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator


class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    s3_key: str
    status: JobStatus
    stage_failed: str | None  # which stage failed: extract, embed, index
    error_message: str | None
    attempts: int
    created_at: datetime
    updated_at: datetime
    # Extracted metadata (populated after extraction)
    city: str | None = None
    state: str | None = None
    year: int | None = None
    page_count: int | None = None
    skip_pages: list[int] | None = None  # Pages to skip during extraction (1-indexed)


class StateDB:
    """SQLite state tracker for pipeline jobs."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    s3_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    stage_failed TEXT,
                    error_message TEXT,
                    attempts INTEGER DEFAULT 0,
                    city TEXT,
                    state TEXT,
                    year INTEGER,
                    page_count INTEGER,
                    skip_pages TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Index for querying by status
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
            """)
            # Migration: add skip_pages column if it doesn't exist
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN skip_pages TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def register_jobs(self, s3_keys: list[str]) -> int:
        """Register new jobs (skip if already exists). Returns count added."""
        added = 0
        with self._connect() as conn:
            for key in s3_keys:
                try:
                    conn.execute(
                        "INSERT INTO jobs (s3_key) VALUES (?)",
                        (key,)
                    )
                    added += 1
                except sqlite3.IntegrityError:
                    pass  # Already exists
            conn.commit()
        return added

    def get_pending(self, stage: JobStatus, limit: int | None = None) -> list[Job]:
        """Get jobs ready for a given stage."""
        # Map stage to the status we're looking for
        status_for_stage = {
            JobStatus.EXTRACTING: JobStatus.PENDING,
            JobStatus.EMBEDDING: JobStatus.EXTRACTED,
        }
        target_status = status_for_stage.get(stage, stage)

        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status = ?
                    ORDER BY created_at
                    """,
                    (target_status.value,)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status = ?
                    ORDER BY created_at
                    LIMIT ?
                    """,
                    (target_status.value, limit)
                ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def get_failed(self, limit: int = 100) -> list[Job]:
        """Get failed jobs for inspection."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'failed'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def get_retryable(self, max_attempts: int = 3) -> list[Job]:
        """Get failed jobs that can be retried."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'failed' AND attempts < ?
                ORDER BY updated_at
                """,
                (max_attempts,)
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def update_status(
        self,
        s3_key: str,
        status: JobStatus,
        error_message: str | None = None,
        stage_failed: str | None = None,
        metadata: dict | None = None,
    ):
        """Update job status and optionally record error/metadata."""
        with self._connect() as conn:
            if error_message:
                conn.execute(
                    """
                    UPDATE jobs SET
                        status = ?,
                        stage_failed = ?,
                        error_message = ?,
                        attempts = attempts + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE s3_key = ?
                    """,
                    (status.value, stage_failed, error_message, s3_key)
                )
            elif metadata:
                skip_pages = metadata.get("skip_pages")
                skip_pages_json = json.dumps(skip_pages) if skip_pages else None
                conn.execute(
                    """
                    UPDATE jobs SET
                        status = ?,
                        city = ?,
                        state = ?,
                        year = ?,
                        page_count = ?,
                        skip_pages = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE s3_key = ?
                    """,
                    (
                        status.value,
                        metadata.get("city"),
                        metadata.get("state"),
                        metadata.get("year"),
                        metadata.get("page_count"),
                        skip_pages_json,
                        s3_key,
                    )
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs SET
                        status = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE s3_key = ?
                    """,
                    (status.value, s3_key)
                )
            conn.commit()

    def reset_for_retry(self, s3_key: str):
        """Reset a failed job to pending for retry."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    status = 'pending',
                    error_message = NULL,
                    stage_failed = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE s3_key = ?
                """,
                (s3_key,)
            )
            conn.commit()

    def reset_all_failed(self, max_attempts: int = 3):
        """Reset all failed jobs under max attempts to pending."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    status = 'pending',
                    error_message = NULL,
                    stage_failed = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = 'failed' AND attempts < ?
                """,
                (max_attempts,)
            )
            conn.commit()

    def set_skip_pages(self, s3_key: str, skip_pages: list[int] | None):
        """Set pages to skip during extraction."""
        with self._connect() as conn:
            skip_pages_json = json.dumps(skip_pages) if skip_pages else None
            conn.execute(
                """
                UPDATE jobs SET
                    skip_pages = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE s3_key = ?
                """,
                (skip_pages_json, s3_key)
            )
            conn.commit()

    def reset_stuck(self) -> int:
        """Reset jobs stuck in intermediate states back to pending."""
        stuck_states = ['extracting', 'extracted', 'embedding', 'embedded']
        with self._connect() as conn:
            result = conn.execute(
                f"""
                UPDATE jobs SET
                    status = 'pending',
                    stage_failed = NULL,
                    error_message = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status IN ({','.join('?' * len(stuck_states))})
                """,
                stuck_states
            )
            conn.commit()
            return result.rowcount

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM jobs
                GROUP BY status
                """
            ).fetchall()
        stats = {row["status"]: row["count"] for row in rows}
        stats["total"] = sum(stats.values())
        return stats

    def get_failure_summary(self) -> list[dict]:
        """Get summary of failures grouped by error type."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    stage_failed,
                    SUBSTR(error_message, 1, 100) as error_preview,
                    COUNT(*) as count
                FROM jobs
                WHERE status = 'failed'
                GROUP BY stage_failed, error_preview
                ORDER BY count DESC
                LIMIT 20
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        """Convert a database row to a Job object."""
        skip_pages = None
        if row["skip_pages"]:
            try:
                skip_pages = json.loads(row["skip_pages"])
            except json.JSONDecodeError:
                pass
        return Job(
            s3_key=row["s3_key"],
            status=JobStatus(row["status"]),
            stage_failed=row["stage_failed"],
            error_message=row["error_message"],
            attempts=row["attempts"],
            city=row["city"],
            state=row["state"],
            year=row["year"],
            page_count=row["page_count"],
            skip_pages=skip_pages,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )
