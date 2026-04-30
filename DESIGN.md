# Zotero Graph — Design Document

A Python web app that renders your Zotero library as an interactive topic graph,
combining tag-based semantic similarity (from the autotagger plugin) with
citation-based structural links (from OpenAlex).

---

## Goals

- Surface clusters of related papers that share research topics
- Expose citation structure inside the library as a second relatedness signal
- Provide a fast, filterable UI for library exploration
- Stay fully local after the initial data fetch (no live API calls during browsing)

## Deferred to later phases (not v1)

- **Semantic / embedding similarity** — third edge type from abstract embeddings
- **Ghost nodes** — papers outside the library cited by ≥3 library papers
- **Cited-by edges** — reverse citation links (expensive per-paper fetch)
- **Full-text search** — FTS5 index over titles + abstracts
- **Improved identifier resolution** — title-search fallback for items with no DOI match
- **Subgraph expansion** — on-demand 1-hop neighborhood fetch
- **OpenAlex topics overlay** — alternative node coloring from OpenAlex taxonomy
- **Export & auto-refresh**

All of the above are planned; see [PLAN.md](PLAN.md) stretch goals for implementation notes.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Data layer (fetcher.py)                                        │
│  ┌──────────────┐   pyzotero    ┌──────────────────────────┐   │
│  │ Zotero Web   │ ───────────── │  items: title, abstract, │   │
│  │ API          │               │  DOI, tags, zotero_key   │   │
│  └──────────────┘               └──────────────────────────┘   │
│                                            │                    │
│  ┌──────────────┐   httpx       ┌──────────────────────────┐   │
│  │ OpenAlex API │ ───────────── │  openalex_id,            │   │
│  │ (polite pool)│               │  referenced_works,       │   │
│  └──────────────┘               │  topics, cited_by_count  │   │
│                                 └──────────────────────────┘   │
│                                            │                    │
│                              ┌─────────────▼──────────────┐    │
│                              │   cache.db  (SQLite)        │    │
│                              │   papers | edges | meta     │    │
│                              └────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                             │
┌────────────────────────────────────────────▼────────────────────┐
│  Graph builder (graph.py)                                        │
│  • Build NetworkX graph from cache                               │
│  • Tag-Jaccard edge weights (semantic)                           │
│  • In-library citation edge weights (structural)                 │
│  • Louvain community detection                                   │
│  • Composite edge weight = α·tag_jaccard + β·citation_overlap   │
└──────────────────────────────────────────────────────────────────┘
                                             │
┌────────────────────────────────────────────▼────────────────────┐
│  Streamlit UI (app.py)                                           │
│  • Sidebar: filters (domain, year, min edge weight, tag search) │
│  • Main panel: PyVis interactive graph (physics layout)         │
│  • Node click → open paper in Zotero via zotero:// deep link   │
│  • Stats panel: top tags, cluster summary, paper count          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data model

### SQLite schema

```sql
CREATE TABLE papers (
    zotero_key      TEXT PRIMARY KEY,
    openalex_id     TEXT,           -- e.g. "W2741809809"
    doi             TEXT,
    title           TEXT NOT NULL,
    abstract        TEXT,
    year            INTEGER,
    domain_tag      TEXT,           -- autotagger broad domain
    content_tags    TEXT,           -- JSON array of strings
    openalex_topics TEXT,           -- JSON array of {id, name, score}
    cited_by_count  INTEGER,
    fetched_at      TEXT            -- ISO timestamp
);

CREATE TABLE citation_edges (
    source_key  TEXT REFERENCES papers(zotero_key),
    target_key  TEXT REFERENCES papers(zotero_key),
    PRIMARY KEY (source_key, target_key)
);

CREATE TABLE meta (
    key     TEXT PRIMARY KEY,
    value   TEXT
);
-- meta rows: last_zotero_sync, last_openalex_sync, zotero_library_id
```

**Notes:**
- `citation_edges` contains only edges where **both** source and target are in the
  library (in-library citation pairs only).
- `content_tags` and `openalex_topics` are stored as JSON strings for simplicity;
  deserialized at graph-build time.
- `domain_tag` is the first tag written by the autotagger (the broad domain).
- Items without a `domain_tag` (not yet autotagged) are included as isolated nodes.

---

## Data pipeline

### Stage 1 — Zotero fetch (`fetcher.py::sync_zotero`)

1. Call `pyzotero.zotero.Zotero(library_id, "user", api_key).items(itemType="journalArticle,preprint,bookSection,conferencePaper")`.
2. For each item, extract: `key`, `title`, `abstractNote`, `DOI`, `date` (year), `tags`.
3. Parse tags: the autotagger writes one `domain` tag first, then content tags. Detect the domain tag as the tag that appears in a known domain list, or simply use the first tag (position 0 in the sorted automatic tag list).
4. Upsert into `papers`. Skip items with no title.
5. Record `last_zotero_sync` in `meta`.

### Stage 2 — OpenAlex enrichment (`fetcher.py::sync_openalex`)

1. Collect all DOIs from `papers` where `openalex_id IS NULL`.
2. Batch into groups of 50. For each batch, call:
   ```
   GET https://api.openalex.org/works
       ?filter=doi:{doi1}|{doi2}|...
       &select=id,doi,referenced_works,topics,cited_by_count
       &mailto=sabarinkumar@gmail.com
   ```
3. For arXiv items without a DOI, construct `doi = 10.48550/arXiv.{arxiv_id}` from the stored URL and try that.
4. For each returned work:
   - Update `papers` with `openalex_id`, `openalex_topics`, `cited_by_count`.
   - Store the `referenced_works` list (OpenAlex IDs) temporarily.
5. After all batches, build `citation_edges`:
   - Construct a set `library_oa_ids = {openalex_id for all papers}`.
   - For each paper, intersect its `referenced_works` with `library_oa_ids`.
   - Insert matching pairs into `citation_edges`.
6. Record `last_openalex_sync` in `meta`.

**Rate limiting:** Use `httpx.AsyncClient` with a semaphore of 5 concurrent requests and a 0.1 s delay between batches. With the polite pool this stays well within 10 req/s.

---

## Graph construction (`graph.py`)

### Nodes

Each paper in `papers` becomes a node with attributes:
```python
{
    "label": title[:60],
    "zotero_key": key,
    "domain": domain_tag or "untagged",
    "tags": content_tags,           # list[str]
    "year": year,
    "cited_by_count": cited_by_count,
    "community": int,               # assigned by Louvain
}
```

### Edges

**Tag-Jaccard edge** between papers A and B:
```
jaccard(A.tags, B.tags) = |A.tags ∩ B.tags| / |A.tags ∪ B.tags|
```
Only emit an edge if `jaccard > threshold` (default 0.15, configurable).

**Citation edge** between A and B:
- Present if `(A.key, B.key)` or `(B.key, A.key)` is in `citation_edges`.
- Weight = `citation_bonus` (default 0.4, added on top of any tag weight).

**Composite weight:**
```python
weight = α * jaccard_weight + β * citation_bonus
# default α=1.0, β=0.4  (citation lifts the edge but doesn't dominate)
```

Edges below a minimum composite weight are dropped to avoid a hairball graph.

### Community detection

Run `python-louvain` (`community.best_partition`) on the weighted graph.
Community IDs are assigned to nodes and used for color mapping in the UI.

---

## UI (`app.py`)

### Layout

```
┌──────────────┬─────────────────────────────────────────────────┐
│   Sidebar    │                  Graph panel                    │
│              │                                                  │
│  Filters:    │   [PyVis iframe — physics-based layout]         │
│  • Domain    │   Nodes colored by Louvain community            │
│  • Year      │   Node size ∝ cited_by_count (log scale)        │
│  • Min edges │   Edge opacity ∝ composite weight               │
│  • Tag text  │                                                  │
│              ├─────────────────────────────────────────────────┤
│  Stats:      │                  Info panel                      │
│  • N papers  │   (appears on node click)                       │
│  • N edges   │   Title, authors, year, domain, tags            │
│  • Top tags  │   [Open in Zotero ↗]  [View on OpenAlex ↗]    │
│              │                                                  │
│  [Refresh]   │                                                  │
└──────────────┴─────────────────────────────────────────────────┘
```

### PyVis configuration

- Physics: `forceAtlas2Based` with `gravitationalConstant=-50`, `springLength=100`
- Node color: mapped from community ID via a categorical palette (≤20 communities)
- Node size: `10 + 5 * log(1 + cited_by_count)`
- Edge color: `rgba(150,150,150, weight)` — heavier edges are more opaque
- Tooltip: title + domain tag + top 3 content tags

### Interactivity

- **Node click**: populate the info panel (via Streamlit session state + JS postMessage).
- **Sidebar filters**: re-run graph build with the filtered node/edge set and re-render.
- **Refresh button**: re-runs the fetch pipeline in a background thread, shows a spinner.

---

## File structure

```
zotero_graph/
├── DESIGN.md               # this document
├── README.md               # setup instructions
├── requirements.txt
├── .env.example            # ZOTERO_LIBRARY_ID, ZOTERO_API_KEY
├── app.py                  # Streamlit entry point
├── fetcher.py              # Zotero + OpenAlex sync
├── graph.py                # NetworkX graph builder + Louvain
├── cache.py                # SQLite read/write helpers
└── config.py               # constants, env var loading
```

---

## Configuration (`config.py` / `.env`)

| Variable | Source | Description |
|---|---|---|
| `ZOTERO_LIBRARY_ID` | `.env` | Numeric Zotero user/group ID |
| `ZOTERO_API_KEY` | `.env` | Zotero Web API key (read-only is fine) |
| `OPENALEX_EMAIL` | hardcoded | `sabarinkumar@gmail.com` (polite pool) |
| `DB_PATH` | `config.py` | `./cache.db` |
| `MIN_JACCARD` | `config.py` | `0.15` — minimum tag overlap for an edge |
| `CITATION_BONUS` | `config.py` | `0.4` — weight added for citation links |
| `ALPHA` | `config.py` | `1.0` — tag weight multiplier |
| `BETA` | `config.py` | `0.4` — citation weight multiplier |

---

## Dependencies

```
pyzotero          # Zotero Web API client
httpx             # async HTTP for OpenAlex
networkx          # graph construction + algorithms
python-louvain    # community detection (package name: community)
pyvis             # vis.js network rendering
streamlit         # web UI framework
python-dotenv     # .env loading
```

---

## Implementation plan

See [PLAN.md](PLAN.md).
