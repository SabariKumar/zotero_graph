from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- Zotero ---
ZOTERO_LIBRARY_ID = os.environ["ZOTERO_LIBRARY_ID"]
ZOTERO_API_KEY = os.environ["ZOTERO_API_KEY"]
ZOTERO_LIBRARY_TYPE = "user"  # "user" or "group"

# --- OpenAlex ---
OPENALEX_EMAIL = "sabarinkumar@gmail.com"
OPENALEX_BATCH_SIZE = 50       # DOIs per request
OPENALEX_CONCURRENCY = 5       # max simultaneous requests
OPENALEX_BATCH_DELAY = 0.1     # seconds between batches

# --- Cache ---
DB_PATH = Path(__file__).parent / "cache.db"

# --- Graph tuning ---
MIN_JACCARD = 0.15      # minimum tag-Jaccard to emit a semantic edge
CITATION_BONUS = 0.4    # weight added to an edge when a citation link exists
ALPHA = 1.0             # multiplier for tag-Jaccard component
BETA = 0.4              # multiplier for citation bonus component
MIN_EDGE_WEIGHT = 0.15  # edges below this composite weight are dropped
