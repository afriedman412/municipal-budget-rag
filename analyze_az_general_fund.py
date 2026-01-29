#!/usr/bin/env python3
"""Analyze general fund subdivisions in Arizona city budgets."""

import re
import chromadb
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma_data"
CHROMA_COLLECTION_NAME = "municipal-budgets"


def extract_city_from_filename(filename: str) -> str:
    """Extract city name from filename like 'az_kingman_18.pdf'."""
    stem = filename.replace(".pdf", "").lower()
    # Remove common suffixes
    stem = re.sub(r'_(proposed|approved|appr|mayor|pt\d+|copy|vol\d+|summary|detail|adopted|tentative|final|prop|book_\d+)$', '', stem, flags=re.IGNORECASE)
    stem = re.sub(r'_\d+$', '', stem)  # Remove trailing year
    stem = re.sub(r'_\d+_\d+$', '', stem)  # Remove year ranges like _21_22

    parts = stem.split("_")
    # First part is state (az, tx, etc), rest is city
    if len(parts) >= 2:
        city_parts = parts[1:]  # Skip state
        # Remove any remaining year-like parts
        city_parts = [p for p in city_parts if not (p.isdigit() and len(p) <= 4)]
        return " ".join(city_parts).title()
    return "Unknown"


def get_az_documents():
    """Get all Arizona budget documents from ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(CHROMA_COLLECTION_NAME)

    # Query for Arizona documents
    results = collection.get(
        where={"state": "AZ"},
        include=["documents", "metadatas"]
    )

    # Group by city (extracted from source filename)
    by_city = defaultdict(list)
    for doc, meta in zip(results["documents"], results["metadatas"]):
        source = meta.get("source", "")
        city = extract_city_from_filename(source) if source else "Unknown"
        by_city[city].append({
            "text": doc,
            "page": meta.get("page", 0),
            "year": meta.get("year", 0),
            "source": source
        })

    return by_city


def find_general_fund_sections(documents):
    """Find chunks that mention general fund departments/divisions."""
    relevant = []
    for doc in documents:
        text_lower = doc["text"].lower()
        if "general fund" in text_lower:
            # Check if it looks like a department listing or budget summary
            if any(kw in text_lower for kw in ["department", "division", "appropriation", "expenditure", "budget"]):
                relevant.append(doc)

    return relevant


def extract_departments_with_llm(city: str, chunks: list, client: OpenAI) -> str:
    """Use LLM to extract department names from budget chunks."""
    # Combine relevant chunks (limit to avoid token issues)
    combined_text = "\n\n---\n\n".join([c["text"][:2500] for c in chunks[:12]])

    prompt = f"""Analyze these budget document excerpts from {city}, Arizona.

Extract a list of all General Fund departments, divisions, or program areas mentioned.
Focus on organizational units that receive General Fund appropriations.

Return ONLY a comma-separated list of department/division names, nothing else.
Normalize names (e.g., "Police Department" -> "Police", "Fire & Rescue" -> "Fire").
If you can't find clear department names, return "UNCLEAR".

Budget excerpts:
{combined_text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0
    )

    return response.choices[0].message.content.strip()


def main():
    print("Loading Arizona budget documents from ChromaDB...")
    by_city = get_az_documents()

    print(f"Found {len(by_city)} Arizona cities\n")

    openai_client = OpenAI()
    all_departments = defaultdict(set)

    for city in sorted(by_city.keys()):
        if city == "Unknown":
            continue

        docs = by_city[city]
        print(f"\n{'='*60}")
        print(f"{city} ({len(docs)} chunks)")
        print("="*60)

        # Find general fund related sections
        gf_sections = find_general_fund_sections(docs)
        print(f"Found {len(gf_sections)} general fund related chunks")

        if gf_sections:
            # Extract departments using LLM
            departments = extract_departments_with_llm(city, gf_sections, openai_client)
            print(f"Departments: {departments}")

            # Track for summary
            if departments != "UNCLEAR":
                for dept in departments.split(","):
                    dept = dept.strip().lower()
                    # Normalize common variations
                    dept = dept.replace(" department", "").replace(" dept", "")
                    dept = dept.replace(" division", "").replace(" services", "")
                    if dept and len(dept) > 1:
                        all_departments[dept].add(city)
        else:
            print("No general fund sections found - checking broader search...")
            # Try a broader search
            budget_chunks = [d for d in docs if "budget" in d["text"].lower()][:8]
            if budget_chunks:
                departments = extract_departments_with_llm(city, budget_chunks, openai_client)
                print(f"Departments (from budget sections): {departments}")
                if departments != "UNCLEAR":
                    for dept in departments.split(","):
                        dept = dept.strip().lower()
                        dept = dept.replace(" department", "").replace(" dept", "")
                        if dept and len(dept) > 1:
                            all_departments[dept].add(city)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: General Fund Departments Across AZ Cities")
    print("="*60)

    # Sort by how many cities have each department
    sorted_depts = sorted(all_departments.items(), key=lambda x: (-len(x[1]), x[0]))

    print(f"\n{'Department':<40} {'# Cities':<10} Cities")
    print("-" * 80)

    for dept, cities in sorted_depts:
        print(f"{dept.title():<40} {len(cities):<10} {', '.join(sorted(cities))}")


if __name__ == "__main__":
    main()
