"""
PDF preflight checks to identify problematic files before extraction.

Catches issues that would crash or hang the main pipeline.
"""

import subprocess
import sys
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Process, Queue
from queue import Empty

logger = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    """Result of preflight check."""
    ok: bool
    issues: list[str]
    page_count: int | None = None
    file_size: int | None = None
    failed_pages: list[int] | None = None  # Page numbers that failed (1-indexed)


def preflight_pdf(path: Path, timeout: int = 60, thorough: bool = False) -> PreflightResult:
    """
    Run preflight checks on a PDF file.

    Checks:
    1. File size and basic validity
    2. PDF header magic bytes
    3. pdfinfo validation (if available)
    4. MuPDF open test with annotation scanning

    Args:
        path: Path to PDF file
        timeout: Timeout in seconds
        thorough: If True, test text extraction on ALL pages (slower but catches more issues)

    Returns PreflightResult with ok=False if any critical issues found.
    """
    issues = []
    page_count = None
    file_size = None

    # 1. Basic file checks
    try:
        file_size = os.path.getsize(path)
        if file_size < 100:
            return PreflightResult(False, ["File too small, likely corrupted"], file_size=file_size)
        if file_size > 500_000_000:  # 500MB
            issues.append(f"Very large file: {file_size / 1_000_000:.0f}MB")
    except OSError as e:
        return PreflightResult(False, [f"Cannot read file: {e}"])

    # 2. Magic bytes check
    try:
        with open(path, 'rb') as f:
            header = f.read(8)
            if not header.startswith(b'%PDF-'):
                return PreflightResult(False, ["Invalid PDF header - not a PDF file"], file_size=file_size)
    except Exception as e:
        return PreflightResult(False, [f"Cannot read file header: {e}"], file_size=file_size)

    # 3. pdfinfo check (if available)
    try:
        result = subprocess.run(
            ['pdfinfo', str(path)],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        if result.returncode != 0:
            # pdfinfo failed - could be encrypted or corrupted
            stderr = result.stderr[:200] if result.stderr else "unknown error"
            if 'Encrypted' in stderr or 'encrypted' in result.stdout.lower():
                issues.append("PDF is encrypted/password-protected")
            else:
                issues.append(f"pdfinfo failed: {stderr}")
        else:
            # Parse output
            for line in result.stdout.split('\n'):
                if line.startswith('Pages:'):
                    try:
                        page_count = int(line.split(':')[1].strip())
                        if page_count > 5000:
                            issues.append(f"Very large document: {page_count} pages")
                    except ValueError:
                        pass
                elif line.startswith('Encrypted:') and 'yes' in line.lower():
                    issues.append("PDF is encrypted")

    except subprocess.TimeoutExpired:
        issues.append(f"pdfinfo timeout after {timeout}s - possibly corrupted")
    except FileNotFoundError:
        # pdfinfo not installed, skip this check
        logger.debug("pdfinfo not available, skipping")
    except Exception as e:
        logger.debug(f"pdfinfo error: {e}")

    # 4. MuPDF open test - catches many issues pdfinfo misses
    failed_pages = []
    try:
        import fitz
        doc = fitz.open(str(path))

        if page_count is None:
            page_count = len(doc)

        # Check pages - either first 5 (quick) or all (thorough)
        pages_to_check = len(doc) if thorough else min(5, len(doc))
        problematic_annotations = []

        for i in range(pages_to_check):
            try:
                page = doc[i]
                annots = list(page.annots() or [])
                for annot in annots:
                    annot_type = annot.type[1] if annot.type else "unknown"
                    if annot_type in ('RichMedia', 'Screen', '3D', 'Movie', 'Sound'):
                        problematic_annotations.append(f"{annot_type} on page {i+1}")

                # In thorough mode, test text extraction on every page
                if thorough:
                    _ = page.get_text()
            except Exception as e:
                issues.append(f"Error reading page {i+1}: {e}")
                if thorough:
                    failed_pages.append(i + 1)  # 1-indexed

        if problematic_annotations:
            issues.append(f"Contains problematic content: {', '.join(problematic_annotations[:3])}")

        # In quick mode, just test first page text extraction
        if not thorough:
            try:
                if len(doc) > 0:
                    _ = doc[0].get_text()
            except Exception as e:
                issues.append(f"Text extraction test failed: {e}")

        doc.close()

    except Exception as e:
        error_msg = str(e)
        # Classify common MuPDF errors
        if 'encrypted' in error_msg.lower():
            issues.append("PDF is encrypted (MuPDF)")
        elif 'password' in error_msg.lower():
            issues.append("PDF requires password")
        else:
            issues.append(f"MuPDF open failed: {error_msg[:100]}")

    # Determine if critical (should skip entire file) vs warning (can try)
    critical_keywords = [
        'encrypted', 'password', 'Invalid PDF', 'corrupted',
        'too small', 'Cannot read', 'RichMedia', '3D', 'Movie'
    ]

    has_critical = any(
        any(kw.lower() in issue.lower() for kw in critical_keywords)
        for issue in issues
    )

    # In thorough mode: if only some pages failed, file is still OK (we'll skip those pages)
    # Only fail if ALL pages failed or there's a critical global error
    if thorough and failed_pages and page_count:
        if len(failed_pages) >= page_count:
            has_critical = True  # All pages failed
            issues.append("All pages failed extraction test")
        elif not has_critical:
            # Some pages failed but file is usable - mark as OK with failed_pages list
            issues.append(f"Pages {failed_pages} will be skipped during extraction")

    return PreflightResult(
        ok=not has_critical,
        issues=issues,
        failed_pages=failed_pages if failed_pages else None,
        page_count=page_count,
        file_size=file_size
    )


def check_pdfinfo_available() -> bool:
    """Check if pdfinfo is installed."""
    try:
        subprocess.run(['pdfinfo', '-v'], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _preflight_worker(path_str: str, timeout: int, thorough: bool, result_queue: Queue):
    """Worker function that runs in subprocess. Crashes here don't kill parent."""
    try:
        result = preflight_pdf(Path(path_str), timeout=timeout, thorough=thorough)
        result_queue.put({
            "ok": result.ok,
            "issues": result.issues,
            "page_count": result.page_count,
            "file_size": result.file_size,
            "failed_pages": result.failed_pages,
        })
    except Exception as e:
        result_queue.put({
            "ok": False,
            "issues": [f"Worker exception: {str(e)}"],
            "page_count": None,
            "file_size": None,
            "failed_pages": None,
        })


def preflight_pdf_safe(path: Path, timeout: int = 60, thorough: bool = False) -> PreflightResult:
    """
    Run preflight in a subprocess so segfaults don't kill the main process.

    Args:
        path: Path to PDF file
        timeout: Timeout in seconds
        thorough: If True, test text extraction on ALL pages (slower but catches more issues)

    If the subprocess crashes (segfault), returns a failed result.
    """
    result_queue = Queue()

    proc = Process(target=_preflight_worker, args=(str(path), timeout, thorough, result_queue))
    proc.start()
    proc.join(timeout=timeout + 10)  # Give a bit more time than the internal timeout

    if proc.is_alive():
        # Timed out
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        return PreflightResult(
            ok=False,
            issues=["Preflight timeout - PDF may be corrupted or very complex"],
            page_count=None,
            file_size=None
        )

    if proc.exitcode != 0:
        # Crashed (segfault, etc.)
        return PreflightResult(
            ok=False,
            issues=[f"Preflight crashed (exit code {proc.exitcode}) - PDF causes segfault"],
            page_count=None,
            file_size=None
        )

    # Get result from queue
    try:
        result_dict = result_queue.get_nowait()
        return PreflightResult(
            ok=result_dict["ok"],
            issues=result_dict["issues"],
            page_count=result_dict["page_count"],
            file_size=result_dict["file_size"],
            failed_pages=result_dict.get("failed_pages"),
        )
    except Empty:
        return PreflightResult(
            ok=False,
            issues=["No result from preflight worker - unknown error"],
            page_count=None,
            file_size=None,
            failed_pages=None
        )
