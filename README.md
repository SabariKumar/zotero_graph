# Zotero Topic Graph

An interactive web app that renders your Zotero library as a physics-based network graph, where papers cluster by shared research topics and are connected by both tag-based semantic similarity and real citation links pulled from OpenAlex. The app is designed to complement the [zotero-autotagger](../zotero_autotagger) plugin: autotagger writes structured tags to every paper as it lands in Zotero, and this app consumes those tags to build the graph.

---

## Setup

**Prerequisites:** [Zotero 7](https://www.zotero.org/) with the autotagger plugin installed, and a Zotero Web API key.

```sh
# 1. Install the environment
pixi install

# 2. Configure credentials
cp .env.example .env
# Edit .env and fill in ZOTERO_LIBRARY_ID and ZOTERO_API_KEY
# Get both from: https://www.zotero.org/settings/keys

# 3. Launch the app
pixi run start
# Then click "Sync Zotero" in the sidebar to populate the cache
```

---

## Purpose

Browsing a large Zotero library by folder or search is linear. This app makes the topical structure of a library visible at once — densely-connected clusters reveal research themes, and citation edges between papers confirm that the clustering reflects actual intellectual lineage rather than coincidental tag overlap.

The data comes from two sources: the autotagger plugin (which writes a broad domain tag and 5–12 specific content tags to each paper on import), and OpenAlex (an open citation database queried by DOI to find which papers in the library cite each other). These two signals are combined into a single composite edge weight, then partitioned into communities by the Louvain algorithm.

---

## Module contents

### `config.py`

Loads Zotero credentials from `.env` and exposes all tunable constants for the pipeline and graph. `ZOTERO_LIBRARY_ID` and `ZOTERO_API_KEY` are required at import time and will raise `KeyError` if absent. All graph-tuning constants (`MIN_JACCARD`, `CITATION_BONUS`, `ALPHA`, `BETA`, `MIN_EDGE_WEIGHT`) live here so they can be adjusted without touching any other file.

### `cache.py`

A thin SQLite layer over three tables — `papers`, `citation_edges`, and `meta`. The key design decision is that `upsert_paper` uses `COALESCE` in the conflict clause: a partial update (e.g. writing only the DOI) never overwrites an existing non-NULL value with NULL. The OpenAlex enrichment step uses a separate `update_paper_openalex` function that updates only the four OpenAlex columns directly, avoiding any risk of overwriting the Zotero title with a blank OpenAlex title. `commit()` is a standalone call so callers can batch writes and commit once.

### `fetcher.py`

Two-stage sync pipeline. `sync_zotero` uses pyzotero's `everything()` helper to pull all citable item types, splits auto-tags (type=1) into the domain tag (position 0) and content tags (positions 1+), and upserts everything into the cache. `sync_openalex` is async internally but exposed as a normal synchronous function via `asyncio.run`. It batches DOIs into groups of 50, fans out all batches concurrently with `asyncio.gather` bounded by a `Semaphore(5)`, then intersects each paper's `referenced_works` list with the set of OpenAlex IDs in the library to produce directed citation edges. For arXiv preprints without an explicit DOI, the Zotero URL field is parsed to construct a canonical DOI (`10.48550/arXiv.{id}`); coverage is partial (see note below).

### `graph.py`

Builds a weighted `nx.Graph` and renders it to HTML. `build_graph` applies the four sidebar filters, computes pairwise tag-Jaccard for every paper pair with overlapping tags (O(n²), acceptable for n ≤ 2000), adds a citation bonus to any edge where a citation link exists, drops edges below `MIN_EDGE_WEIGHT`, and runs Louvain community detection on the result. `render_pyvis` maps Louvain community IDs to the matplotlib `tab20` palette, log-scales node sizes on `cited_by_count`, and colours citation edges amber to distinguish them visually from tag-only edges. The rendered HTML is written to a temp file, read into memory, and the file is deleted — the graph is passed to Streamlit as an in-memory string.

### `app.py`

Streamlit entry point. All helper functions are defined before the main script body because Streamlit runs the file top-to-bottom on every rerender and forward references will fail. The SQLite connection is wrapped in `@st.cache_resource` so it persists across reruns without reopening the file. Node clicks are bridged from the PyVis iframe to Streamlit session state via `localStorage`: the injected `selectNode` handler writes the Zotero key to `localStorage["zotero_selected_node"]`, and `st_javascript` polls that key on each render cycle, triggering `st.rerun()` when the value changes. The sidebar selectbox serves as a reliable fallback and stays in sync with session state in both directions.

---

## Data contracts

### SQLite schema (`cache.db`)

| Table | Key columns |
|---|---|
| `papers` | `zotero_key` (PK), `doi`, `url`, `title`, `year`, `domain_tag`, `content_tags` (JSON array), `openalex_id`, `openalex_topics` (JSON array of `{id, name, score}`), `cited_by_count` |
| `citation_edges` | `source_key`, `target_key` — in-library pairs only |
| `meta` | `last_zotero_sync`, `last_openalex_sync` (ISO timestamps) |

### Tag schema (from autotagger plugin)

All tags are lowercase hyphenated strings written as automatic tags (Zotero type=1). The first automatic tag is always the broad domain (e.g. `structural-biology`); subsequent tags are content tags (e.g. `protein-folding`, `diffusion-model`). Manual tags (type=0) are ignored.

### Graph edge attributes

| Attribute | Type | Meaning |
|---|---|---|
| `weight` | float | Composite: `ALPHA × jaccard + BETA × CITATION_BONUS` |
| `tag_weight` | float | Raw Jaccard component (0.0 for citation-only edges) |
| `citation` | bool | True if a citation link exists between the two papers |

---

## Critical parameters and constraints

**`MIN_JACCARD = 0.15`** — the threshold below which a tag-similarity edge is not emitted. Too low produces a hairball; too high disconnects the graph. Adjust in `config.py`.

**`MIN_EDGE_WEIGHT = 0.15`** — composite edges below this are dropped after all bonuses are applied. A citation-only edge has weight `BETA × CITATION_BONUS = 0.4 × 0.4 = 0.16`, which is just above the default threshold. If you raise `MIN_EDGE_WEIGHT` above 0.16, citation-only edges (papers that cite each other but share no tags) will be silently dropped.

**`ZOTERO_LIBRARY_TYPE = "user"`** — change this to `"group"` in `config.py` if your library is a Zotero group library. The library ID format differs between the two.

**arXiv DOI coverage** — OpenAlex does not reliably index arXiv-only preprints under the `10.48550/arXiv.*` DOI scheme. Papers with formal journal DOIs (Nature, ACL, IEEE, etc.) match at 100%; arXiv-only preprints may not appear in OpenAlex at all or may be indexed under a different identifier. Unmatched papers appear in the graph as isolated nodes or with tag-only edges. A title-search fallback is planned for a future version.

**Pairwise Jaccard complexity** — `build_graph` computes all n×(n−1)/2 tag pairs. For a library of 500 papers this is ~125,000 comparisons and takes under a second. Performance degrades quadratically; consider profiling before using on a library larger than ~2,000 papers.

**`@st.cache_resource` on the DB connection** — the SQLite connection is shared across all Streamlit reruns in the same process. If you need to reload after a schema change, restart the Streamlit server rather than just rerunning.

---

## Dependencies on other modules

| Dependency | Direction | What is consumed / produced |
|---|---|---|
| `zotero_autotagger` plugin | upstream | writes `domain_tag` + `content_tags` as type=1 tags in Zotero |
| Zotero Web API | upstream | item metadata (titles, abstracts, DOIs, URLs, tags) |
| OpenAlex API | upstream | `openalex_id`, `referenced_works`, `topics`, `cited_by_count` |
| `cache.db` | internal | all modules read/write through `cache.py`; no module accesses SQLite directly |

---

## Graph tuning reference

```
weight = ALPHA × jaccard(A.tags, B.tags) + BETA × CITATION_BONUS
```

Default values: `ALPHA=1.0`, `BETA=0.4`, `CITATION_BONUS=0.4`, giving a maximum citation bonus of 0.16 on top of the tag-Jaccard component. All four constants are in `config.py`.
