"""Project-wide non-secret configuration."""

from pathlib import Path

# Directories
PDF_DIR = Path("pdfs_2026")
TRAINING_DIR = Path("training")
RUNS_DIR = Path("runs")

# 6-city parser test set (2 always_correct, 2 close, 2 always_fail)
PARSER_TEST_PDFS = [
    "az_flagstaff_19.pdf",
    "tx_pflugerville_20.pdf",
    "ca_san_rafael_18.pdf",
    "oh_cincinnati_24_25_bi.pdf",
    "or_corvallis_18.pdf",
    "wi_milwaukee_19.pdf",
]

# 6-city LLM test set (each tested for General Fund + Police = 12 records)
LLM_TEST_CITIES = [
    {"city": "vallejo", "state": "ca", "year": 2021},
    {"city": "el_paso", "state": "tx", "year": 2020},
    {"city": "corvallis", "state": "or", "year": 2018},
    {"city": "peekskill", "state": "ny", "year": 2023},
    {"city": "carrollton", "state": "ga", "year": 2022},
    {"city": "paducah", "state": "ky", "year": 2022},
]

# Embedding
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 1536

# ChromaDB collection naming: budgets-{parser_name}
PARSERS = ["aryn", "pymupdf", "llamaparse", "unstructured"]
