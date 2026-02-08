"""CLI for the pipeline."""

import asyncio
import logging

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

app = typer.Typer(help="Municipal budget PDF pipeline: S3 → Aryn → OpenAI → ChromaDB")
console = Console()


def get_config():
    from .config import Config
    return Config.from_env()


@app.command()
def run(
    batch_size: int = typer.Option(None, "--batch", "-b", help="Batch size"),
    limit: int = typer.Option(None, "--limit", "-n", help="Max documents to process"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Run the pipeline on pending jobs."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from .pipeline import Pipeline

    config = get_config()
    pipeline = Pipeline(config)

    stats = asyncio.run(pipeline.run(batch_size=batch_size, limit=limit))

    console.print()
    _print_stats(stats)


@app.command()
def status():
    """Show pipeline status."""
    from .state import StateDB

    config = get_config()
    state = StateDB(config.state_db)
    stats = state.get_stats()

    _print_stats(stats)


@app.command()
def failures(
    limit: int = typer.Option(20, "--limit", "-n", help="Max failures to show"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full errors"),
):
    """Show failed jobs."""
    from .state import StateDB

    config = get_config()
    state = StateDB(config.state_db)
    failed = state.get_failed(limit=limit)

    if not failed:
        console.print("[green]No failures[/green]")
        return

    table = Table(title=f"Failed Jobs ({len(failed)})")
    table.add_column("File")
    table.add_column("Error")

    for job in failed:
        filename = job.s3_key.split("/")[-1]
        error = job.error or "Unknown"
        if not verbose:
            error = error[:60] + "..." if len(error) > 60 else error
        table.add_row(filename, error)

    console.print(table)


@app.command()
def retry():
    """Reset failed jobs to pending."""
    from .state import StateDB

    config = get_config()
    state = StateDB(config.state_db)
    count = state.reset_failed()
    console.print(f"[green]Reset {count} failed jobs to pending[/green]")


@app.command()
def reset():
    """Reset stuck processing jobs to pending."""
    from .state import StateDB

    config = get_config()
    state = StateDB(config.state_db)
    count = state.reset_processing()
    console.print(f"[green]Reset {count} processing jobs to pending[/green]")


@app.command()
def discover():
    """Discover PDFs in S3 without processing."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from .pipeline import Pipeline

    config = get_config()
    pipeline = Pipeline(config)
    asyncio.run(pipeline.discover())

    status()


@app.command()
def stats():
    """Show ChromaDB statistics."""
    from .chroma import ChromaClient

    config = get_config()
    chroma = ChromaClient(config)
    info = chroma.get_stats()

    console.print(f"Collection: {info['collection']}")
    console.print(f"Total chunks: {info['total_chunks']:,}")


def _print_stats(stats: dict):
    table = Table(title="Pipeline Status")
    table.add_column("Status")
    table.add_column("Count", justify="right")

    for status in ["pending", "processing", "done", "failed"]:
        count = stats.get(status, 0)
        color = {
            "pending": "yellow",
            "processing": "blue",
            "done": "green",
            "failed": "red",
        }.get(status, "white")
        table.add_row(status.capitalize(), f"[{color}]{count}[/{color}]")

    table.add_row("Total", str(stats.get("total", 0)), style="bold")
    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
