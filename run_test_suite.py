"""Run all 6 test suites and print a summary grid."""

import argparse
import json
import subprocess
import re
import sys
import urllib.request

SUITES = [
    ("pymupdf",    "clean", "training/test_chunks_val.json"),
    ("pymupdf",    "d4",    "training/test_chunks_val_d4.json"),
    ("pdfplumber", "clean", "training/test_chunks_val_pdfplumber.json"),
    ("pdfplumber", "d4",    "training/test_chunks_val_pdfplumber_d4.json"),
    ("aryn",       "clean", "training/test_chunks_val_aryn.json"),
    ("aryn",       "d4",    "training/test_chunks_val_aryn_d4.json"),
]

BAR_WIDTH = 30

def progress_bar(done, total, label):
    frac = done / total if total else 0
    filled = int(BAR_WIDTH * frac)
    bar = "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)
    sys.stderr.write(f"\r  {label}  {bar} {done}/{total}")
    sys.stderr.flush()


def run_suite(parser, condition, cache, model, version, llm_url, workers, extra_args):
    ver = f"{version}-val-{parser}-{condition}"
    cmd = [
        sys.executable, "-u", "test_llm_extraction.py",
        "--llm-url", llm_url,
        "--model", model,
        "--cache", cache,
        "--version", ver,
        "--workers", str(workers),
    ] + extra_args

    # Get total record count from cache file
    try:
        total = len(json.load(open(cache)))
    except Exception:
        total = 60

    label = f"{parser}/{condition}"
    done = 0
    progress_bar(done, total, label)

    output_lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        output_lines.append(line)
        if "OK" in line or "MISS" in line or "CLOSE" in line or "NO CHUNKS" in line:
            done += 1
            progress_bar(done, total, label)
    proc.wait()
    if proc.stderr:
        err = proc.stderr.read()
        if err:
            print(err, file=sys.stderr)

    # Clear progress bar and print summary line
    sys.stderr.write("\r" + " " * 80 + "\r")
    sys.stderr.flush()
    output = "".join(output_lines)
    info = parse_summary(output)
    if info:
        close = info.get("close", 0)
        miss = info.get("miss", 0)
        print(f"  {label:<20} {info['match']:>3}/{info['total']} ({info['pct']}%)   close:{close}  miss:{miss}")
    else:
        print(f"  {label:<20} FAILED")
    return output


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
    p.add_argument("--samples", type=int, default=1, help="Run each record N times (majority vote)")
    p.add_argument("--wandb", action="store_true", help="Pass --wandb to test script")
    args = p.parse_args()

    # Check vLLM is reachable
    try:
        url = args.llm_url.rstrip("/") + "/models"
        urllib.request.urlopen(url, timeout=5)
    except Exception:
        print(f"ERROR: vLLM not reachable at {args.llm_url}", file=sys.stderr)
        sys.exit(1)

    extra = []
    if args.wandb:
        extra += ["--wandb"]
    if args.samples > 1:
        extra += ["--samples", str(args.samples)]
    results = []

    print(f"\n  {args.model} ({args.version})")
    print(f"  {'='*50}")

    for parser, condition, cache in SUITES:
        output = run_suite(parser, condition, cache, args.model, args.version,
                           args.llm_url, args.workers, extra)
        info = parse_summary(output)
        results.append((parser, condition, info))

    # Summary grid
    print(f"\n  {'='*50}")
    print(f"  SUMMARY")
    print(f"  {'-'*50}")
    for parser, condition, info in results:
        label = f"{parser}/{condition}"
        if info:
            close = info.get("close", 0)
            miss = info.get("miss", 0)
            print(f"  {label:<20} {info['match']:>3}/{info['total']} ({info['pct']}%)   close:{close}  miss:{miss}")
        else:
            print(f"  {label:<20} FAILED")
    print()


if __name__ == "__main__":
    main()
