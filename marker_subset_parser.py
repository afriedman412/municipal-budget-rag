from dataclasses import dataclass
from pathlib import Path
import re
import os
import json
import pandas as pd
from build_page_cache import _html_to_text

csv_path = "/home/user/municipal-budget-rag/easy_budget_page_index.csv"


@dataclass(frozen=True)
class BudgetPdfName:
    state: str
    city: str
    year: str
    qualifier: str | None = None

    @classmethod
    def parse(cls, filename: str) -> "BudgetPdfName":
        stem = Path(filename).stem.lower()
        parts = stem.split("_")

        if len(parts) < 3:
            raise ValueError(f"Unexpected filename format: {filename}")

        state = parts[0].upper()

        year_idx = None
        for i in range(len(parts) - 1, -1, -1):
            if re.fullmatch(r"\d{2}", parts[i]):
                year_idx = i
                break

        if year_idx is None:
            raise ValueError(
                f"Could not find 2-digit year in filename: {filename}")

        city = " ".join(parts[1:year_idx]).lower()
        qualifier = "_".join(parts[year_idx + 1:]) or None

        return cls(state=state, city=city, year=parts[year_idx], qualifier=qualifier)

    @property
    def gf_type(self) -> str:
        return f"gf{self.year}"

    @property
    def p_type(self) -> str:
        return f"p{self.year}"

    @property
    def canonical_stem(self) -> str:
        return f"{self.state.lower()}_{self.city.replace(' ', '_')}_{self.year}"


class BudgetPageLookup:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path).copy()
        df["state"] = df["state"].astype(str).str.upper().str.strip()
        df["city"] = df["city"].astype(str).str.lower().str.strip()
        df["budget type"] = df["budget type"].astype(
            str).str.lower().str.strip()
        self.df = df.set_index(["state", "city", "budget type"]).sort_index()

    def lookup_budget_pages(self, pdf_name: str) -> dict:
        doc = BudgetPdfName.parse(pdf_name)

        result = {"gf": None, "p": None}
        for label, budget_type in [("gf", doc.gf_type), ("p", doc.p_type)]:
            try:
                value = self.df.loc[(doc.state, doc.city,
                                     budget_type), "pdf page"]
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                result[label] = None if pd.isna(value) else int(value)
            except KeyError:
                result[label] = None

        return result


def _get_marker_converter(page_range: str | None = None):
    """specific for page ranges!"""
    from marker.converters.pdf import PdfConverter
    from marker.config.parser import ConfigParser
    from marker.models import create_model_dict

    config = {"output_format": "json"}
    if page_range:
        config["page_range"] = page_range

    config_parser = ConfigParser(config)
    return PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )


def make_marker_page_range(target_pages_1idx: list[int], window: int = 5) -> str:
    """
    Input pages are 1-indexed from your CSV/PDF.
    Output is Marker page_range string in 0-indexed form.
    """
    selected = set()

    for p in target_pages_1idx:
        if p is None:
            continue
        center = p - 1  # convert to Marker-style 0-index
        for page in range(max(0, center - window), center + window + 1):
            selected.add(page)

    if not selected:
        return ""

    pages = sorted(selected)

    # compress consecutive pages into ranges
    ranges = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = p
    ranges.append(f"{start}-{prev}" if start != prev else str(start))

    return ",".join(ranges)


def get_selected_pages_and_range(target_pages_1idx: list[int], window: int = 5) -> tuple[list[int], str]:
    selected = set()

    for p in target_pages_1idx:
        if p is None:
            continue
        center = p - 1
        for page in range(max(0, center - window), center + window + 1):
            selected.add(page)

    if not selected:
        return [], ""

    pages = sorted(selected)

    ranges = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = p
    ranges.append(f"{start}-{prev}" if start != prev else str(start))

    return pages, ",".join(ranges)


def process_pdf(pdf_path: Path, lookup: BudgetPageLookup, output_dir: Path):

    doc = BudgetPdfName.parse(pdf_path.name)
    targets = lookup.lookup_budget_pages(pdf_path.name)

    target_pages = [p for p in [targets["gf"], targets["p"]] if p is not None]
    if not target_pages:
        print(f"SKIP {pdf_path.name}: no target pages")
        return

    out_path = output_dir / f"{doc.canonical_stem}.json"
    if out_path.exists():
        print(f"SKIP {pdf_path.name}: already parsed")
        return

    selected_pages_0idx, page_range = get_selected_pages_and_range(
        target_pages, window=5)
    converter = _get_marker_converter(page_range=page_range)
    rendered = converter(str(pdf_path))

    blocks = []
    selected_pages_1idx = [p + 1 for p in selected_pages_0idx]

    for page_idx, page in enumerate(rendered.children):
        original_page_1idx = selected_pages_1idx[page_idx]

        for block_idx, block in enumerate(page.children):
            html = getattr(block, "html", "") or ""
            if not html.strip():
                continue

            text = _html_to_text(html).strip()
            if not text:
                continue

            blocks.append({
                "subset_page": page_idx,
                "original_page": original_page_1idx,
                "block_idx": block_idx,
                "block_type": str(getattr(block, "block_type", "unknown")),
                "is_gf_target_page": original_page_1idx == targets["gf"],
                "is_p_target_page": original_page_1idx == targets["p"],
                "text": text,
                "html": html,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_pdf": pdf_path.name,
                "target_pages": targets,
                "selected_pages_1idx": selected_pages_1idx,
                "page_range": page_range,
                "blocks": blocks,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {out_path.name}")


def process_pdfs(csv_path: str, pdf_dir: str = "pdfs_2026", output_dir: str = "marker_blocks"):
    lookup = BudgetPageLookup(csv_path)

    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    total_docs = len(pdf_paths)
    for n, pdf_path in enumerate(pdf_paths, start=1):
        print(f"Processing {pdf_path.name} ({n}/{total_docs})")
        try:
            process_pdf(pdf_path, lookup, output_dir)
        except Exception as e:
            print(f"ERROR {pdf_path.name}: {e}")


if __name__ == "__main__":
    process_pdfs(csv_path)
