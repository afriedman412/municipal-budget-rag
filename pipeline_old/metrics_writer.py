# metrics_writer.py
"""
Writes pipeline metrics to a JSON file for external monitoring.

Same interface as RichMonitor - accepts update() calls with snapshots.
A separate web server can read the JSON file to display progress.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


def _get(snap: Any, key: str, default: Any = 0) -> Any:
    """Read key from dict-like or attribute-like snapshot."""
    if snap is None:
        return default
    if isinstance(snap, dict):
        return snap.get(key, default)
    return getattr(snap, key, default)


@dataclass
class PipelineMetrics:
    """Current pipeline state - serialized to JSON."""
    started_at: float = 0.0
    last_update_at: float = 0.0

    # Core counters
    extracted_docs: int = 0
    extracted_chunks: int = 0
    embedded_docs: int = 0
    embedded_chunks: int = 0

    # Batch tracking
    batches_started: int = 0
    batches_done: int = 0
    queue_depth: int = 0

    # Status
    errors: int = 0
    producer_done: bool = False

    # Current activity
    phase: str = ""  # "downloading", "extracting", "embedding", "idle"
    current_file: str = ""
    current_files: list[str] = field(default_factory=list)  # For parallel mode

    # Batch progress
    download_count: int = 0
    download_total: int = 0
    extract_done: int = 0
    extract_total: int = 0


class MetricsWriter:
    """
    Writes pipeline metrics to a JSON file.

    Drop-in replacement for RichMonitor - same update() interface.
    """

    def __init__(self, metrics_path: Path | str = "pipeline_metrics.json"):
        self.metrics_path = Path(metrics_path)
        self._metrics = PipelineMetrics(started_at=time.time())
        self._lock = threading.Lock()

    @property
    def callback(self) -> Callable[[Any], None]:
        """A function you pass into your orchestrator as progress_cb(snapshot)."""
        return self.update

    def start(self) -> None:
        """Initialize metrics file. Called when pipeline starts."""
        self._metrics.started_at = time.time()
        self._metrics.last_update_at = time.time()
        self._write()

    def stop(self) -> None:
        """Final write when pipeline stops."""
        with self._lock:
            self._metrics.phase = "stopped"
            self._metrics.last_update_at = time.time()
        self._write()

    def update(self, snapshot: Any) -> None:
        """Update metrics from a snapshot dict/object and write to file."""
        with self._lock:
            self._metrics.last_update_at = time.time()

            # Core counters
            self._metrics.extracted_docs = int(
                _get(snapshot, "extracted_docs", self._metrics.extracted_docs) or 0)
            self._metrics.extracted_chunks = int(
                _get(snapshot, "extracted_chunks", self._metrics.extracted_chunks) or 0)
            self._metrics.embedded_docs = int(
                _get(snapshot, "embedded_docs", self._metrics.embedded_docs) or 0)
            self._metrics.embedded_chunks = int(
                _get(snapshot, "embedded_chunks", self._metrics.embedded_chunks) or 0)

            # Batch tracking
            self._metrics.batches_started = int(
                _get(snapshot, "batches_started", self._metrics.batches_started) or 0)
            self._metrics.batches_done = int(
                _get(snapshot, "batches_done", self._metrics.batches_done) or 0)
            self._metrics.queue_depth = int(
                _get(snapshot, "queue_depth", self._metrics.queue_depth) or 0)

            # Status
            self._metrics.errors = int(
                _get(snapshot, "errors", self._metrics.errors) or 0)
            self._metrics.producer_done = bool(
                _get(snapshot, "producer_done", self._metrics.producer_done))

            # Current activity
            self._metrics.phase = str(
                _get(snapshot, "phase", self._metrics.phase) or "")
            self._metrics.current_file = str(
                _get(snapshot, "current_file", self._metrics.current_file) or "")

            # Handle multiple current files (for parallel extraction)
            current_files = _get(snapshot, "current_files", None)
            if current_files:
                self._metrics.current_files = list(current_files)
            elif self._metrics.current_file:
                self._metrics.current_files = [self._metrics.current_file]

            # Batch progress
            self._metrics.download_count = int(
                _get(snapshot, "download_count", self._metrics.download_count) or 0)
            self._metrics.download_total = int(
                _get(snapshot, "download_total", self._metrics.download_total) or 0)
            self._metrics.extract_done = int(
                _get(snapshot, "extract_done", self._metrics.extract_done) or 0)
            self._metrics.extract_total = int(
                _get(snapshot, "extract_total", self._metrics.extract_total) or 0)

        self._write()

    def _write(self) -> None:
        """Write current metrics to JSON file."""
        with self._lock:
            data = asdict(self._metrics)

        # Atomic write: write to temp file then rename
        tmp_path = self.metrics_path.with_suffix('.tmp')
        try:
            tmp_path.write_text(json.dumps(data, indent=2))
            tmp_path.rename(self.metrics_path)
        except Exception:
            # If atomic write fails, try direct write
            try:
                self.metrics_path.write_text(json.dumps(data, indent=2))
            except Exception:
                pass  # Don't crash pipeline if metrics write fails


def read_metrics(metrics_path: Path | str = "pipeline_metrics.json") -> PipelineMetrics | None:
    """Read metrics from JSON file. Returns None if file doesn't exist or is invalid."""
    path = Path(metrics_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return PipelineMetrics(**data)
    except Exception:
        return None
