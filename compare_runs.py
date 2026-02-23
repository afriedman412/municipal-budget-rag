"""Compare misses across multiple run files.

Usage:
    python compare_runs.py runs/*v5-val*
    python compare_runs.py runs/*20260223*
"""

import json
import os
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_runs.py <run_files...>")
        sys.exit(1)

    all_names = []
    misses = {}

    for path in sorted(sys.argv[1:]):
        with open(path) as f:
            data = json.load(f)
        name = data.get("version", os.path.basename(path))
        all_names.append(name)
        for rec in data["results"]:
            key = (rec["state"], rec["city"], rec.get("year", ""), rec["expense"])
            misses.setdefault(key, {})[name] = {
                "match": rec.get("match", False),
                "answer": rec.get("answer", ""),
                "expected": rec.get("expected", ""),
            }

    all_names = sorted(set(all_names))
    short = [n.replace("v5-val-", "") for n in all_names]

    # Header
    hdr = "  ".join("{:<14}".format(s) for s in short)
    print("{:<40} {}".format("Record", hdr))
    print("-" * (40 + 16 * len(short)))

    # Only show rows with at least one miss
    miss_counts = {n: 0 for n in all_names}
    all_miss = 0
    for key in sorted(misses.keys()):
        results = misses[key]
        if all(results.get(n, {}).get("match", False) for n in all_names):
            continue

        st, city, yr, exp = key
        parts = []
        is_all_miss = True
        for n in all_names:
            info = results.get(n, {})
            if info.get("match", False):
                parts.append("{:<14}".format("OK"))
                is_all_miss = False
            else:
                miss_counts[n] += 1
                ans = str(info.get("answer", ""))[:14]
                parts.append("{:<14}".format(ans or "MISS"))
        if is_all_miss:
            all_miss += 1

        label = "{} {} {} {}".format(st.upper(), city, yr, exp)
        print("{:<40} {}".format(label[:40], "  ".join(parts)))

    # Summary
    print("-" * (40 + 16 * len(short)))
    print("Misses:  ", "  ".join("{:<14}".format(str(miss_counts[n])) for n in all_names))
    print("All-miss (every run):", all_miss)


if __name__ == "__main__":
    main()
