#!/usr/bin/env python3
"""
Analyze general fund subdivisions across all city budgets.
Outputs both a summary analysis and a CSV with city/state/year/subdivision/amount.
"""

import csv
import json
import re
import chromadb
from collections import defaultdict
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma_data"
CHROMA_COLLECTION_NAME = "municipal-budgets"
OUTPUT_CSV = "./general_fund_subdivisions.csv"


def extract_city_from_filename(filename: str) -> str:
    """Extract city name from filename like 'az_kingman_18.pdf'."""
    stem = filename.replace(".pdf", "").lower()
    # Remove common suffixes
    stem = re.sub(r'_(proposed|approved|appr|mayor|pt\d+|copy|vol\d*|summary|detail\d*|adopted|tentative|final|prop|book_\d+|overview|gf|pos)$', '', stem, flags=re.IGNORECASE)
    stem = re.sub(r'_\(\d+\)$', '', stem)  # Remove (1) suffix
    stem = re.sub(r'_\d{4}$', '', stem)  # Remove 4-digit year
    stem = re.sub(r'_\d{2}$', '', stem)  # Remove 2-digit year
    stem = re.sub(r'_\d{2}_\d{2}$', '', stem)  # Remove year ranges

    parts = stem.split("_")
    if len(parts) >= 2:
        state = parts[0].upper()
        city_parts = parts[1:]
        city_parts = [p for p in city_parts if not (p.isdigit() and len(p) <= 4)]
        city = " ".join(city_parts).title()
        return city, state
    return "Unknown", "XX"


def extract_year_from_filename(filename: str) -> int:
    """Extract year from filename."""
    # Look for 2-digit or 4-digit year patterns
    match = re.search(r'_(\d{4})[\._]', filename)
    if match:
        return int(match.group(1))
    match = re.search(r'_(\d{2})[\._]', filename)
    if match:
        year = int(match.group(1))
        return 2000 + year if year < 50 else 1900 + year
    match = re.search(r'_(\d{2})_', filename)
    if match:
        year = int(match.group(1))
        return 2000 + year if year < 50 else 1900 + year
    return 0


def get_all_documents():
    """Get AZ and TX budget documents from ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(CHROMA_COLLECTION_NAME)

    # Only get AZ and TX documents
    az_results = collection.get(
        where={"state": "AZ"},
        include=["documents", "metadatas"]
    )
    tx_results = collection.get(
        where={"state": "TX"},
        include=["documents", "metadatas"]
    )

    # Combine results
    class CombinedResults:
        def __init__(self):
            self.documents = az_results["documents"] + tx_results["documents"]
            self.metadatas = az_results["metadatas"] + tx_results["metadatas"]

    results = CombinedResults()

    # Group by city+year (using source filename as key to dedupe)
    by_source = defaultdict(list)
    for doc, meta in zip(results.documents, results.metadatas):
        source = meta.get("source", "")
        if source:
            by_source[source].append({
                "text": doc,
                "page": meta.get("page", 0),
                "source": source
            })

    return by_source


def find_general_fund_sections(documents):
    """Find chunks that mention general fund departments/divisions with amounts."""
    relevant = []
    for doc in documents:
        text_lower = doc["text"].lower()
        # Look for general fund content with dollar amounts or budget tables
        has_gf = "general fund" in text_lower
        has_amounts = bool(re.search(r'\$[\d,]+|\d{1,3}(?:,\d{3})+', doc["text"]))
        has_dept_context = any(kw in text_lower for kw in [
            "department", "division", "appropriation", "expenditure",
            "budget", "personnel", "operating", "total"
        ])

        if has_gf and (has_amounts or has_dept_context):
            relevant.append(doc)

    # Sort by page number to get structured content
    relevant.sort(key=lambda x: x["page"])
    return relevant


def extract_departments_with_amounts(city: str, state: str, year: int, chunks: list, client: OpenAI) -> list[dict]:
    """Use LLM to extract department names and amounts from budget chunks."""
    combined_text = "\n\n---PAGE BREAK---\n\n".join([c["text"][:2000] for c in chunks[:15]])

    prompt = f"""Analyze these budget document excerpts from {city}, {state} (fiscal year {year}).

Extract General Fund department/division appropriations. For each department found, provide:
- Department name (normalized, e.g., "Police Department" -> "Police")
- Budget amount (the total appropriation/expenditure for that department)

IMPORTANT:
- Only include General Fund departments (not enterprise funds, special revenue, etc.)
- Use the most specific department-level amounts, not category totals
- If a department appears multiple times, use the total/final amount
- Amounts should be in dollars (no thousands notation - convert "1,234" to 1234000 if the table header says "in thousands")

Return a JSON array like this (no other text):
[
  {{"department": "Police", "amount": 15000000}},
  {{"department": "Fire", "amount": 12000000}}
]

If you cannot find clear department amounts, return: []

Budget excerpts:
{combined_text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in content:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                content = match.group(1)

        departments = json.loads(content)
        return departments
    except Exception as e:
        print(f"    Error parsing response: {e}")
        return []


def main():
    print("Loading all budget documents from ChromaDB...")
    by_source = get_all_documents()

    print(f"Found {len(by_source)} unique budget documents\n")

    openai_client = OpenAI()

    # Track all extractions for CSV
    all_records = []

    # Track for summary
    dept_counts = defaultdict(lambda: {"count": 0, "cities": set(), "total_amount": 0})

    # Group sources by city for cleaner output
    sources_by_city = defaultdict(list)
    for source in by_source.keys():
        city, state = extract_city_from_filename(source)
        sources_by_city[(city, state)].append(source)

    for (city, state), sources in sorted(sources_by_city.items()):
        if city == "Unknown":
            continue

        print(f"\n{'='*60}")
        print(f"{city}, {state} ({len(sources)} budget files)")
        print("="*60)

        for source in sorted(sources):
            year = extract_year_from_filename(source)
            docs = by_source[source]

            print(f"\n  {source} (year: {year}, {len(docs)} chunks)")

            # Find general fund related sections
            gf_sections = find_general_fund_sections(docs)
            print(f"    Found {len(gf_sections)} general fund chunks with amounts")

            if gf_sections:
                departments = extract_departments_with_amounts(
                    city, state, year, gf_sections, openai_client
                )

                if departments:
                    print(f"    Extracted {len(departments)} departments:")
                    for dept in departments[:5]:  # Show first 5
                        amt = dept.get('amount', 0)
                        print(f"      - {dept['department']}: ${amt:,.0f}")
                    if len(departments) > 5:
                        print(f"      ... and {len(departments) - 5} more")

                    # Add to records
                    for dept in departments:
                        record = {
                            "city": city,
                            "state": state,
                            "year": year,
                            "subdivision": dept.get("department", "Unknown"),
                            "amount": dept.get("amount", 0)
                        }
                        all_records.append(record)

                        # Track for summary
                        dept_name = dept.get("department", "").lower()
                        dept_counts[dept_name]["count"] += 1
                        dept_counts[dept_name]["cities"].add(f"{city}, {state}")
                        dept_counts[dept_name]["total_amount"] += dept.get("amount", 0)
                else:
                    print("    No department amounts extracted")
            else:
                # Try broader search
                budget_chunks = [d for d in docs if re.search(r'\$[\d,]+', d["text"])][:10]
                if budget_chunks:
                    print(f"    Trying broader search ({len(budget_chunks)} chunks with $)")
                    departments = extract_departments_with_amounts(
                        city, state, year, budget_chunks, openai_client
                    )
                    if departments:
                        print(f"    Extracted {len(departments)} departments")
                        for dept in departments:
                            record = {
                                "city": city,
                                "state": state,
                                "year": year,
                                "subdivision": dept.get("department", "Unknown"),
                                "amount": dept.get("amount", 0)
                            }
                            all_records.append(record)

                            dept_name = dept.get("department", "").lower()
                            dept_counts[dept_name]["count"] += 1
                            dept_counts[dept_name]["cities"].add(f"{city}, {state}")
                            dept_counts[dept_name]["total_amount"] += dept.get("amount", 0)

    # Write CSV
    print(f"\n\n{'='*60}")
    print(f"Writing {len(all_records)} records to {OUTPUT_CSV}")
    print("="*60)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["city", "state", "year", "subdivision", "amount"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"CSV saved to {OUTPUT_CSV}")

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY: Most Common General Fund Departments")
    print("="*60)

    sorted_depts = sorted(
        dept_counts.items(),
        key=lambda x: (-x[1]["count"], x[0])
    )

    print(f"\n{'Department':<35} {'Occurrences':<12} {'Total $':<18} {'# Cities'}")
    print("-" * 85)

    for dept, stats in sorted_depts[:40]:
        if dept:
            print(f"{dept.title():<35} {stats['count']:<12} ${stats['total_amount']:>14,.0f}   {len(stats['cities'])}")


if __name__ == "__main__":
    main()
