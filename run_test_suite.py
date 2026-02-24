"""Run all 6 test suites and print a summary grid."""

import argparse
import subprocess
import re
import sys

SUITES = [
    ("pymupdf",    "clean", "training/test_chunks_val.json"),
    ("pymupdf",    "d4",    "training/test_chunks_val_d4.json"),
    ("pdfplumber", "clean", "training/test_chunks_val_pdfplumber.json"),
    ("pdfplumber", "d4",    "training/test_chunks_val_pdfplumber_d4.json"),
    ("aryn",       "clean", "training/test_chunks_val_aryn.json"),
    ("aryn",       "d4",    "training/test_chunks_val_aryn_d4.json"),
]

def run_suite(parser, condition, cache, model, version, llm_url, workers, extra_args):
    ver = f"{version}-val-{parser}-{condition}"
    cmd = [
        sys.executable, "test_llm_extraction.py",
        "--llm-url", llm_url,
        "--model", model,
        "--cache", cache,
        "--version", ver,
        "--workers", str(workers),
    ] + extra_args
    print(f"\n{'='*60}")
    print(f"  {parser} / {condition}  ({cache})")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.stdout


def parse_summary(output):
    """Extract match/close/miss/in_chunks from the summary line."""
    m = re.search(r"Match:\s*(\d+)/(\d+)\s*\((\d+)%\)", output)
    if not m:
        return None
    info = {"match": int(m.group(1)), "total": int(m.group(2)), "pct": int(m.group(3))}
    c = re.search(r"Close:\s*(\d+)", output)
    if c:
        info["close"] = int(c.group(1))
    mi = re.search(r"Miss:\s*(\d+)", output)
    if mi:
        info["miss"] = int(mi.group(1))
    ic = re.search(r"In chunks:\s*(\d+)/(\d+)", output)
    if ic:
        info["in_chunks"] = int(ic.group(1))
    return info


def main():
    p = argparse.ArgumentParser(description="Run all 6 test suites and print summary")
    p.add_argument("--model", required=True, help="Model name (e.g. budget-mistral-lora-v6-merged)")
    p.add_argument("--version", required=True, help="Version prefix (e.g. v6)")
    p.add_argument("--llm-url", default="http://localhost:8000/v1")
    p.add_argument("--workers", "-w", type=int, default=1)
    p.add_argument("--wandb", action="store_true", help="Pass --wandb to test script")
    args = p.parse_args()

    extra = ["--wandb"] if args.wandb else []
    results = []

    for parser, condition, cache in SUITES:
        output = run_suite(parser, condition, cache, args.model, args.version,
                           args.llm_url, args.workers, extra)
        info = parse_summary(output)
        results.append((parser, condition, info))

    # Summary grid
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {args.model} ({args.version})")
    print(f"{'='*60}")
    print(f"{'Parser':<12} {'Condition':<10} {'Match':>8} {'Close':>8} {'Miss':>8} {'In Chunks':>10}")
    print("-" * 60)
    for parser, condition, info in results:
        if info:
            close = info.get("close", "-")
            miss = info.get("miss", "-")
            ic = info.get("in_chunks", "-")
            print(f"{parser:<12} {condition:<10} {info['match']:>3}/{info['total']} ({info['pct']}%) {close:>8} {miss:>8} {ic:>10}")
        else:
            print(f"{parser:<12} {condition:<10} {'FAILED':>8}")


if __name__ == "__main__":
    main()
