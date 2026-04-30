"""
config.py — Central configuration for zotero_graph.

Loads Zotero credentials from a .env file and exposes all tunable constants
for the data pipeline and graph construction. Changing a value here takes
effect on the next `pixi run start` without touching any other module.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Zotero ---
ZOTERO_LIBRARY_ID: str = os.environ["ZOTERO_LIBRARY_ID"]
ZOTERO_API_KEY: str = os.environ["ZOTERO_API_KEY"]
ZOTERO_LIBRARY_TYPE: str = "user"  # "user" or "group"

# --- OpenAlex ---
OPENALEX_EMAIL: str = "sabarinkumar@gmail.com"  # polite-pool identifier
OPENALEX_BATCH_SIZE: int = 50  # DOIs per request (OpenAlex max is ~100)
OPENALEX_CONCURRENCY: int = 5  # max simultaneous async requests
OPENALEX_BATCH_DELAY: float = 0.1  # seconds of sleep inside each semaphore slot

# --- Cache ---
# Resolve two levels up from src/zotero_graph/ to reach the project root
DB_PATH: Path = Path(__file__).parents[2] / "cache.db"

# --- Graph tuning ---
MIN_JACCARD: float = 0.15  # minimum tag-Jaccard to emit a semantic edge
CITATION_BONUS: float = 0.4  # raw weight added per citation link
ALPHA: float = 1.0  # multiplier for the tag-Jaccard component
BETA: float = 0.4  # multiplier for the citation bonus component
MIN_EDGE_WEIGHT: float = 0.15  # composite edges below this threshold are dropped
