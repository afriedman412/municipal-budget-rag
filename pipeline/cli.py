#!/usr/bin/env python3
"""
CLI for the municipal budget PDF pipeline.

Usage:
    python -m pipeline.cli run          # Run full pipeline
    python -m pipeline.cli status       # Show pipeline status
    python -m pipeline.cli failures     # Show failed jobs
    python -m pipeline.cli retry        # Retry failed jobs
"""

import asyncio
import logging
import sys

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from .config import PipelineConfig
from .state import StateDB
from .orchestrator import Pipeline
from .chroma import ChromaClient

app = typer.Typer(help="Municipal Budget PDF Pipeline")
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
# Quiet down noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_config() -> PipelineConfig:
    """Load config from environment."""
    config = PipelineConfig.from_env()

    # Validate required settings
    if not config.s3_bucket:
        console.print("[red]Error: S3_BUCKET environment variable not set[/red]")
        raise typer.Exit(1)

    return config


@app.command()
def run(
    batch_size: int = typer.Option(100, help="Batch size for processing"),
    workers: int = typer.Option(None, help="Override PDF worker count"),
    simple: bool = typer.Option(False, help="Use simple sequential mode (for debugging)"),
):
    """Run the full pipeline: discover → extract → embed → index."""
    config = get_config()

    if workers:
        config.pdf_workers = workers

    console.print(f"[bold]Municipal Budget Pipeline[/bold]")
    console.print(f"  S3: s3://{config.s3_bucket}/{config.s3_prefix}")
    console.print(f"  Chroma: {config.chroma_host or 'local'}")
    console.print(f"  Workers: {config.pdf_workers}")
    console.print(f"  Mode: {'sequential' if simple else 'parallel (producer-consumer)'}")
    console.print()

    pipeline = Pipeline(config)
    if simple:
        asyncio.run(pipeline.run_simple(batch_size=batch_size))
    else:
        asyncio.run(pipeline.run(batch_size=batch_size))


@app.command()
def status():
    """Show current pipeline status."""
    config = get_config()
    state = StateDB(config.state_db_path)

    stats = state.get_stats()

    table = Table(title="Pipeline Status")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")

    status_order = ["pending", "extracting", "extracted", "embedding", "done", "failed"]
    for s in status_order:
        count = stats.get(s, 0)
        style = "red" if s == "failed" and count > 0 else None
        table.add_row(s, str(count), style=style)

    table.add_row("─" * 10, "─" * 5)
    table.add_row("total", str(stats.get("total", 0)), style="bold")

    console.print(table)

    # ChromaDB stats
    try:
        chroma = ChromaClient(config)
        chroma_stats = chroma.get_stats()
        console.print()
        console.print(f"[bold]ChromaDB:[/bold] {chroma_stats['total_chunks']:,} chunks, "
                      f"{chroma_stats['unique_documents']} documents")
        if chroma_stats['states']:
            console.print(f"  States: {', '.join(chroma_stats['states'])}")
    except Exception as e:
        console.print(f"[yellow]ChromaDB unavailable: {e}[/yellow]")


@app.command()
def failures(
    limit: int = typer.Option(20, help="Max failures to show"),
    verbose: bool = typer.Option(False, "-v", help="Show full error messages"),
):
    """Show failed jobs and error summary."""
    config = get_config()
    state = StateDB(config.state_db_path)

    # Error summary
    summary = state.get_failure_summary()
    if summary:
        table = Table(title="Failure Summary")
        table.add_column("Stage")
        table.add_column("Error")
        table.add_column("Count", justify="right")

        for row in summary:
            table.add_row(
                row["stage_failed"] or "?",
                row["error_preview"] or "?",
                str(row["count"])
            )
        console.print(table)
        console.print()

    # Individual failures
    failed = state.get_failed(limit=limit)
    if not failed:
        console.print("[green]No failed jobs[/green]")
        return

    table = Table(title=f"Failed Jobs (showing {len(failed)})")
    table.add_column("S3 Key")
    table.add_column("Stage")
    table.add_column("Attempts")
    table.add_column("Error")

    for job in failed:
        error = job.error_message or ""
        if not verbose and len(error) > 50:
            error = error[:50] + "..."

        table.add_row(
            job.s3_key[-50:] if len(job.s3_key) > 50 else job.s3_key,
            job.stage_failed or "?",
            str(job.attempts),
            error
        )

    console.print(table)


@app.command()
def retry(
    max_attempts: int = typer.Option(3, help="Max attempts before giving up"),
):
    """Reset failed jobs and retry them."""
    config = get_config()
    state = StateDB(config.state_db_path)

    retryable = state.get_retryable(max_attempts)
    if not retryable:
        console.print("[yellow]No jobs to retry (all exceeded max attempts)[/yellow]")
        return

    console.print(f"Retrying {len(retryable)} failed jobs...")

    pipeline = Pipeline(config)
    asyncio.run(pipeline.retry_failed())


@app.command()
def reset_failed():
    """Reset ALL failed jobs to pending (use with caution)."""
    config = get_config()
    state = StateDB(config.state_db_path)

    failed = state.get_failed(limit=10000)
    if not failed:
        console.print("[green]No failed jobs to reset[/green]")
        return

    if not typer.confirm(f"Reset {len(failed)} failed jobs to pending?"):
        raise typer.Abort()

    for job in failed:
        state.reset_for_retry(job.s3_key)

    console.print(f"[green]Reset {len(failed)} jobs to pending[/green]")


@app.command()
def discover():
    """Discover PDFs in S3 and register as jobs (without processing)."""
    config = get_config()
    pipeline = Pipeline(config)

    added = pipeline.discover_pdfs()
    console.print(f"[green]Registered {added} new jobs[/green]")


def main():
    app()


if __name__ == "__main__":
    main()
