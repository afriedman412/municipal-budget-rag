# rich_monitor.py
"""
Rich-powered live monitor for an async ingestion/orchestration pipeline.

How to use (callback-based):
----------------------------
1) In your orchestrator, accept an optional `progress_cb` callable and invoke it
   with a ProgressSnapshot-like object/dict.

2) Wrap that callback with `RichMonitor.callback()` and start the monitor before
   you kick off the run.

Example:
    from rich_monitor import RichMonitor

    monitor = RichMonitor(refresh_hz=4)
    monitor.start()

    await orchestrator.run(progress_cb=monitor.callback)

    monitor.stop()

Snapshot format:
----------------
The monitor accepts either:
- a dict with keys like:
    extracted_chunks, embedded_chunks, batches_started, batches_done,
    queue_depth, producer_done, errors
- OR an object with those as attributes (e.g. a dataclass).

Everything is optional; missing values are treated as 0/False.
"""

from __future__ import annotations

import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


def _get(snap: Any, key: str, default: Any = 0) -> Any:
    """Read key from dict-like or attribute-like snapshot."""
    if snap is None:
        return default
    if isinstance(snap, dict):
        return snap.get(key, default)
    return getattr(snap, key, default)


@dataclass
class _State:
    started_at: float = field(default_factory=time.time)
    last_update_at: float = field(default_factory=time.time)

    extracted_chunks: int = 0
    embedded_chunks: int = 0
    batches_started: int = 0
    batches_done: int = 0
    queue_depth: int = 0
    errors: int = 0
    producer_done: bool = False

    # Optional extras (nice to display if you have them)
    extracted_docs: int = 0
    embedded_docs: int = 0
    phase: str = ""  # e.g. "downloading", "extracting", "embedding"
    current_file: str = ""  # Current file being processed
    download_count: int = 0  # Files downloaded in current batch
    download_total: int = 0  # Total files to download in current batch
    extract_done: int = 0  # Files extracted in current batch
    extract_total: int = 0  # Total files to extract in current batch


class RichMonitor:
    """
    A small, dependency-contained Rich dashboard that you can keep out of the orchestrator.

    - Thread-safe callback you can call from anywhere (async loop, producer thread, etc.).
    - Runs a background thread that renders the dashboard at a fixed refresh rate.
    - Auto-disables itself if stdout isn't a TTY (unless force_tty=True).
    """

    def __init__(
        self,
        refresh_hz: float = 4.0,
        console: Optional[Console] = None,
        force_tty: bool = False,
        show_when_not_tty: bool = False,
    ) -> None:
        """
        Args:
            refresh_hz: How often to redraw per second.
            console: Optional Rich Console.
            force_tty: Treat output as TTY even if sys.stdout isn't.
            show_when_not_tty: If True, print periodic snapshots (no Live UI).
        """
        self.refresh_hz = max(0.5, float(refresh_hz))
        self.console = console or Console()
        self.force_tty = force_tty
        self.show_when_not_tty = show_when_not_tty

        self._state = _State()
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._enabled_live = bool(force_tty or sys.stdout.isatty())

    @property
    def callback(self) -> Callable[[Any], None]:
        """A function you pass into your orchestrator as progress_cb(snapshot)."""
        return self.update

    def start(self) -> None:
        """Start the background renderer thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, name="RichMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the renderer thread and finalize the display."""
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def update(self, snapshot: Any) -> None:
        """Update internal counters from a snapshot dict/object."""
        with self._lock:
            self._state.last_update_at = time.time()

            # Required-ish fields
            self._state.extracted_chunks = int(
                _get(snapshot, "extracted_chunks", self._state.extracted_chunks) or 0)
            self._state.embedded_chunks = int(
                _get(snapshot, "embedded_chunks", self._state.embedded_chunks) or 0)
            self._state.batches_started = int(
                _get(snapshot, "batches_started", self._state.batches_started) or 0)
            self._state.batches_done = int(
                _get(snapshot, "batches_done", self._state.batches_done) or 0)
            self._state.queue_depth = int(
                _get(snapshot, "queue_depth", self._state.queue_depth) or 0)
            self._state.errors = int(
                _get(snapshot, "errors", self._state.errors) or 0)
            self._state.producer_done = bool(
                _get(snapshot, "producer_done", self._state.producer_done))

            # Optional extras
            self._state.extracted_docs = int(
                _get(snapshot, "extracted_docs", self._state.extracted_docs) or 0)
            self._state.embedded_docs = int(
                _get(snapshot, "embedded_docs", self._state.embedded_docs) or 0)
            self._state.phase = str(
                _get(snapshot, "phase", self._state.phase) or "")
            self._state.current_file = str(
                _get(snapshot, "current_file", self._state.current_file) or "")
            self._state.download_count = int(
                _get(snapshot, "download_count", self._state.download_count) or 0)
            self._state.download_total = int(
                _get(snapshot, "download_total", self._state.download_total) or 0)
            self._state.extract_done = int(
                _get(snapshot, "extract_done", self._state.extract_done) or 0)
            self._state.extract_total = int(
                _get(snapshot, "extract_total", self._state.extract_total) or 0)

    def _render(self, st: _State) -> Table:
        elapsed = time.time() - st.started_at
        since_update = time.time() - st.last_update_at

        embed_rate = (st.embedded_chunks / elapsed) if elapsed > 0 else 0.0
        extract_rate = (st.extracted_chunks / elapsed) if elapsed > 0 else 0.0

        table = Table(title="Ingestion Pipeline Monitor", expand=True)
        table.add_column("Metric")
        table.add_column("Value", justify="right")

        # Status line
        status = Text("RUNNING", style="bold green")
        if st.errors > 0:
            status = Text("RUNNING (WITH ERRORS)", style="bold yellow")
        if st.producer_done and st.batches_done >= st.batches_started and st.queue_depth == 0:
            status = Text("DONE", style="bold cyan")
        if since_update > 10:
            status = Text(
                f"STALE ({since_update:.0f}s since update)", style="bold red")

        table.add_row("Status", status)
        table.add_row("Elapsed", f"{elapsed:,.1f}s")
        if st.phase:
            phase_text = st.phase
            if st.phase == "downloading" and st.download_total > 0:
                phase_text = f"downloading ({st.download_count}/{st.download_total})"
            elif st.phase == "extracting" and st.extract_total > 0:
                phase_text = f"extracting ({st.extract_done}/{st.extract_total})"
            table.add_row("Phase", phase_text)
        if st.current_file:
            # Display files (supports multiline for multiple files)
            table.add_row("Current file(s)", st.current_file)

        # Core counters
        if st.extracted_docs:
            table.add_row("Extracted docs", f"{st.extracted_docs:,}")
        table.add_row("Extracted chunks", f"{st.extracted_chunks:,}")
        table.add_row("Extract rate", f"{extract_rate:,.1f} chunks/s")

        if st.embedded_docs:
            table.add_row("Embedded docs", f"{st.embedded_docs:,}")
        table.add_row("Embedded chunks", f"{st.embedded_chunks:,}")
        table.add_row("Embed rate", f"{embed_rate:,.1f} chunks/s")

        table.add_row("Batches", f"{st.batches_done:,}/{st.batches_started:,}")
        table.add_row("Queue depth", f"{st.queue_depth:,}")
        table.add_row("Producer done", "✅" if st.producer_done else "⏳")
        table.add_row("Errors", f"{st.errors:,}")

        return table

    def _run(self) -> None:
        # Fallback mode: no TTY -> periodic prints (optional)
        if not self._enabled_live:
            if not self.show_when_not_tty:
                return

            # Print a line every ~2 seconds
            next_print = 0.0
            while not self._stop_evt.is_set():
                time.sleep(0.2)
                now = time.time()
                if now < next_print:
                    continue
                next_print = now + 2.0
                with self._lock:
                    st = self._state
                    elapsed = now - st.started_at
                    self.console.print(
                        f"[progress] {elapsed:6.1f}s "
                        f"extracted_chunks={st.extracted_chunks} embedded_chunks={st.embedded_chunks} "
                        f"batches={st.batches_done}/{st.batches_started} q={st.queue_depth} "
                        f"errors={st.errors} producer_done={st.producer_done}"
                    )
            return

        # Live UI mode
        refresh_per_second = int(self.refresh_hz)
        refresh_per_second = max(2, refresh_per_second)

        with Live(self._render(self._state), console=self.console, refresh_per_second=refresh_per_second) as live:
            while not self._stop_evt.is_set():
                time.sleep(1.0 / self.refresh_hz)
                with self._lock:
                    live.update(self._render(self._state))


# Optional: a tiny demo you can run directly:
if __name__ == "__main__":
    import random

    mon = RichMonitor(refresh_hz=6)
    mon.start()

    extracted = 0
    embedded = 0
    bs = 0
    bd = 0
    q = 0

    try:
        for _ in range(200):
            time.sleep(0.05)
            extracted += random.randint(5, 20)
            q = min(q + random.randint(0, 2), 50)
            if random.random() < 0.3:
                bs += 1
            if random.random() < 0.25 and bd < bs:
                bd += 1
                embedded += random.randint(10, 60)
                q = max(q - random.randint(0, 3), 0)

            mon.update(
                {
                    "extracted_chunks": extracted,
                    "embedded_chunks": embedded,
                    "batches_started": bs,
                    "batches_done": bd,
                    "queue_depth": q,
                    "producer_done": False,
                    "errors": 0,
                    "phase": "demo",
                }
            )

        mon.update(
            {
                "extracted_chunks": extracted,
                "embedded_chunks": embedded,
                "batches_started": bs,
                "batches_done": bs,
                "queue_depth": 0,
                "producer_done": True,
                "errors": 0,
                "phase": "done",
            }
        )
        time.sleep(0.5)
    finally:
        mon.stop()
