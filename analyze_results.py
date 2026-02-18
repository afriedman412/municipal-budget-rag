"""Analyze LLM extraction test results.

Usage:
  python analyze_results.py runs/some_run.json                    # Analyze single run
  python analyze_results.py runs/base.json runs/finetuned.json    # Compare two runs
"""

import json
import sys


def classify_error(r):
    """Classify a MISS result into error category."""
    if r.get("no_chunks"):
        return "no_chunks"
    if not r["in_chunks"]:
        return "not_in_chunks"
    if r["answer"] is None:
        return "no_answer"

    expected = int(r["expected"])
    # Try to extract the main number from the answer
    import re
    raw = re.findall(r'[\$]?[\d,]+(?:\.\d+)?', r["answer"])
    answer_nums = set()
    for s in raw:
        cleaned = s.replace('$', '').replace(',', '')
        if '.' in cleaned:
            cleaned = cleaned.split('.')[0]
        if cleaned.isdigit() and len(cleaned) >= 4:
            answer_nums.add(int(cleaned))

    if not answer_nums:
        return "no_number"

    # Find the closest number to expected
    closest = min(answer_nums, key=lambda x: abs(x - expected))
    ratio = closest / expected if expected else 0

    if 0.85 <= ratio <= 1.15:
        return "close"  # within 15%
    elif closest > expected * 1.5:
        return "wrong_fund"  # grabbed a larger scope (total budget vs GF)
    else:
        return "wrong_scope"  # grabbed a smaller scope (line item vs total)


def analyze_run(run):
    """Print analysis of a single run."""
    results = run["results"]
    print(f"Model: {run['model']}")
    print(f"Chunks: {run['n_chunks']}  |  Cache: {run['cache_file']}")
    print(f"Date: {run['timestamp']}")
    print(f"Overall: {run['match']}/{run['total']} ({100*run['match']/run['total']:.0f}%)")
    print(f"In chunks: {run['in_chunks']}/{run['total']}")
    print()

    # Break down by expense type
    by_expense = {}
    for r in results:
        exp = r["expense"]
        if exp not in by_expense:
            by_expense[exp] = {"total": 0, "match": 0, "in_chunks": 0, "errors": {}}
        by_expense[exp]["total"] += 1
        if r["match"]:
            by_expense[exp]["match"] += 1
        if r["in_chunks"]:
            by_expense[exp]["in_chunks"] += 1
        if not r["match"]:
            cat = classify_error(r)
            by_expense[exp]["errors"][cat] = by_expense[exp]["errors"].get(cat, 0) + 1

    for exp in sorted(by_expense):
        b = by_expense[exp]
        pct = 100 * b["match"] / b["total"] if b["total"] else 0
        in_pct = 100 * b["in_chunks"] / b["total"] if b["total"] else 0
        ext_rate = 100 * b["match"] / b["in_chunks"] if b["in_chunks"] else 0
        print(f"  {exp}:")
        print(f"    Match: {b['match']}/{b['total']} ({pct:.0f}%)  |  In chunks: {b['in_chunks']}/{b['total']} ({in_pct:.0f}%)  |  Extraction: {b['match']}/{b['in_chunks']} ({ext_rate:.0f}%)")
        if b["errors"]:
            errs = sorted(b["errors"].items(), key=lambda x: -x[1])
            print(f"    Errors: {', '.join(f'{k}={v}' for k, v in errs)}")
    print()

    # Overall error breakdown
    errors = {}
    for r in results:
        if not r["match"]:
            cat = classify_error(r)
            errors[cat] = errors.get(cat, 0) + 1
    errs = sorted(errors.items(), key=lambda x: -x[1])
    print(f"  Error breakdown: {', '.join(f'{k}={v}' for k, v in errs)}")


def compare_runs(run_a, run_b):
    """Compare two runs side by side."""
    print(f"{'':40} {'Run A':>15} {'Run B':>15} {'Delta':>10}")
    print(f"{'Model':<40} {run_a['model']:>15} {run_b['model']:>15}")
    print(f"{'Chunks':<40} {run_a['n_chunks']:>15} {run_b['n_chunks']:>15}")
    print("-" * 85)

    ma, mb = run_a["match"], run_b["match"]
    ta, tb = run_a["total"], run_b["total"]
    print(f"{'Match':<40} {f'{ma}/{ta} ({100*ma/ta:.0f}%)':>15} {f'{mb}/{tb} ({100*mb/tb:.0f}%)':>15} {mb-ma:>+10}")
    ia, ib = run_a["in_chunks"], run_b["in_chunks"]
    print(f"{'In chunks':<40} {f'{ia}/{ta}':>15} {f'{ib}/{tb}':>15} {ib-ia:>+10}")
    print()

    # Per-record comparison
    key_fn = lambda r: (r["state"], r["city"], r["year"], r["expense"])
    a_map = {key_fn(r): r for r in run_a["results"]}
    b_map = {key_fn(r): r for r in run_b["results"]}

    improved = []
    regressed = []
    for key in a_map:
        if key not in b_map:
            continue
        ra, rb = a_map[key], b_map[key]
        if not ra["match"] and rb["match"]:
            improved.append(key)
        elif ra["match"] and not rb["match"]:
            regressed.append(key)

    if improved:
        print(f"Improved ({len(improved)}):")
        for s, c, y, e in sorted(improved):
            print(f"  {s.upper()} {c} {y} {e}")
    if regressed:
        print(f"\nRegressed ({len(regressed)}):")
        for s, c, y, e in sorted(regressed):
            print(f"  {s.upper()} {c} {y} {e}")
    if not improved and not regressed:
        print("No changes between runs.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <run.json> [run2.json]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        run_a = json.load(f)

    if len(sys.argv) >= 3:
        with open(sys.argv[2]) as f:
            run_b = json.load(f)
        compare_runs(run_a, run_b)
    else:
        analyze_run(run_a)


if __name__ == "__main__":
    main()
