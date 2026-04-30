# Zotero Graph — Implementation Plan

Ordered implementation steps. Each step is independently testable before moving on.

---

## Step 1 — Project skeleton & config

**Files:** `config.py`, `.env.example`, `requirements.txt`

- [ ] Create `requirements.txt` with all dependencies
- [ ] Create `.env.example` with `ZOTERO_LIBRARY_ID` and `ZOTERO_API_KEY` placeholders
- [ ] Create `config.py`:
  - Load `ZOTERO_LIBRARY_ID` and `ZOTERO_API_KEY` from `.env` via `python-dotenv`
  - Hardcode `OPENALEX_EMAIL = "sabarinkumar@gmail.com"`
  - Define graph tuning constants: `MIN_JACCARD`, `CITATION_BONUS`, `ALPHA`, `BETA`
  - Define `DB_PATH = Path("cache.db")`

**Verification:** `python -c "from config import ZOTERO_LIBRARY_ID; print(ZOTERO_LIBRARY_ID)"` prints the ID.

---

## Step 2 — SQLite cache layer

**File:** `cache.py`

- [ ] `init_db(db_path)` — create tables if not exist (papers, citation_edges, meta)
- [ ] `upsert_paper(conn, paper_dict)` — insert or replace a paper row
- [ ] `get_all_papers(conn)` → `list[dict]`
- [ ] `get_papers_missing_openalex(conn)` → papers where `openalex_id IS NULL`
- [ ] `upsert_citation_edge(conn, source_key, target_key)`
- [ ] `get_all_citation_edges(conn)` → `list[tuple[str, str]]`
- [ ] `set_meta(conn, key, value)` / `get_meta(conn, key)`

**Verification:** Unit test — insert a paper, read it back, confirm roundtrip.

---

## Step 3 — Zotero fetcher

**File:** `fetcher.py::sync_zotero(conn)`

- [ ] Initialize `pyzotero.Zotero` with library ID + API key
- [ ] Fetch all items with `itemType` filter (journalArticle, preprint, bookSection, conferencePaper)
- [ ] For each item:
  - Extract `key`, `title`, `abstractNote`, `DOI`, `year` (parse from `date` field)
  - Separate `domain_tag` (first automatic tag) from `content_tags` (remaining automatic tags)
  - Skip items with no title
  - Call `upsert_paper`
- [ ] Call `set_meta(conn, "last_zotero_sync", datetime.now().isoformat())`

**Tag parsing detail:**
Zotero returns tags as `[{"tag": "name", "type": 1}, ...]`. Filter for `type == 1`
(automatic). The autotagger writes domain first, so position 0 is the domain tag.
If no automatic tags exist, both fields are empty.

**Verification:** Run sync, query `papers` table, confirm count matches Zotero library size.

---

## Step 4 — OpenAlex fetcher

**File:** `fetcher.py::sync_openalex(conn)`

- [ ] Collect DOIs from `get_papers_missing_openalex`
- [ ] For arXiv items (DOI is null but URL contains `arxiv.org`), construct DOI:
  `10.48550/arXiv.{arxiv_id}` from the URL
- [ ] Batch DOIs into groups of 50
- [ ] For each batch, `GET https://api.openalex.org/works` with:
  - `filter=doi:{doi1}|{doi2}|...`
  - `select=id,doi,referenced_works,topics,cited_by_count`
  - `mailto=sabarinkumar@gmail.com`
- [ ] Parse response, match returned works back to `zotero_key` by DOI
- [ ] Update `papers` with `openalex_id`, `openalex_topics` (JSON), `cited_by_count`
- [ ] Collect all `(paper_key, [referenced_openalex_ids])` pairs
- [ ] Build `library_openalex_ids` set from all non-null `openalex_id` values
- [ ] For each paper, intersect its references with `library_openalex_ids`
- [ ] Insert matching pairs into `citation_edges`
- [ ] Call `set_meta(conn, "last_openalex_sync", datetime.now().isoformat())`

**Rate limit:** Use `asyncio` + `httpx.AsyncClient` with `asyncio.gather` across all
batches (not a sequential loop) with a shared `Semaphore(5)` to cap concurrency.
Wrap in a sync-friendly `asyncio.run()` entrypoint.

**Implementation note — targeted update:** Use `update_paper_openalex()` (a bare `UPDATE`
on the four OpenAlex columns) rather than `upsert_paper()` for the enrichment step.
`upsert_paper` always overwrites `title`; passing an empty OpenAlex title would silently
erase the title fetched from Zotero.

**Verification:** Run sync on a small set; confirm `openalex_id` populated, `citation_edges`
has rows for papers that cite each other.

**Finding — arXiv DOI coverage (discovered 2026-04-30):**
OpenAlex does not reliably index preprints under their arXiv DOI (`10.48550/arXiv.*`).
Confirmed via live API testing:
- Papers with real journal DOIs (Nature, ACL, IEEE CVPR, etc.) match 100% via the
  `filter=doi:` parameter.
- Papers that exist only as arXiv preprints (e.g. "Attention Is All You Need", which
  has no formal NeurIPS proceedings DOI) are stored under a different identifier
  (e.g. `10.65215/...`) and are not found via the arXiv DOI filter.
- The constructed arXiv DOI fallback (`10.48550/arXiv.{id}`) is still worth attempting
  — some arXiv papers do resolve — but coverage will be partial.
- **Mitigation:** the title-search fallback in the stretch goals ("Improved identifier
  resolution") is the correct fix for preprint-heavy libraries. For v1, unmatched
  papers appear as nodes without citation edges, which is acceptable.

---

## Step 5 — Graph builder

**File:** `graph.py`

- [ ] `build_graph(conn, filters=None)` → `(nx.Graph, community_map)`
  - Load all papers from cache (apply year/domain filters if provided)
  - Add a node per paper with all attributes
  - Compute tag-Jaccard for every paper pair with overlapping tags (skip pairs with no overlap)
  - Add tag-Jaccard edges above `MIN_JACCARD`
  - Load citation edges; for each in-library pair, add/update edge with `CITATION_BONUS`
  - Compute composite weight: `α * jaccard + β * citation_bonus`
  - Drop edges below composite threshold
  - Run `community.best_partition(G, weight="weight")` → `community_map`
  - Attach `community` attribute to each node

- [ ] `graph_stats(G)` → dict with `n_papers`, `n_edges`, `top_tags`, `n_communities`

**Performance note:** Pairwise Jaccard is O(n²). For n ≤ 2000 papers this is fine.
Use set intersection for speed: represent each paper's tags as a `frozenset`.

**Verification:** Print graph stats; inspect that densely-tagged papers have many edges.

---

## Step 6 — PyVis renderer

**File:** `graph.py::render_pyvis(G, community_map, output_path)`

- [ ] Map community IDs to a categorical color palette (use `matplotlib.colormaps["tab20"]`)
- [ ] Initialize `pyvis.network.Network(height="750px", width="100%", bgcolor="#1a1a1a", font_color="white")`
- [ ] Configure physics: `forceAtlas2Based` with tuned `gravitationalConstant`, `springLength`
- [ ] For each node: add with `color`, `size` (log-scaled by `cited_by_count`), `title` (HTML tooltip)
- [ ] For each edge: add with `color` as `rgba(200,200,200,{weight})`, `width=weight*3`
- [ ] `net.save_graph(output_path)` → writes an HTML file
- [ ] Return `output_path`

**Verification:** Open the HTML file in a browser; confirm interactive physics layout renders.

---

## Step 7 — Streamlit app skeleton

**File:** `app.py`

- [ ] Load config; initialize DB on first run (`init_db`)
- [ ] Sidebar:
  - Domain filter: multiselect populated from distinct `domain_tag` values in DB
  - Year range: slider from min to max year in DB
  - Min edge weight: slider 0.0 – 1.0 (default 0.15)
  - Tag search: text input (filters to papers whose tags contain the string)
  - Refresh button: runs `sync_zotero` + `sync_openalex` with a `st.spinner`
- [ ] Main panel:
  - Call `build_graph(conn, filters=sidebar_state)`
  - Call `render_pyvis(G, ...)` → write to a temp HTML file
  - `st.components.v1.html(open(html_path).read(), height=760, scrolling=False)`
  - Stats row below graph: paper count, edge count, community count
- [ ] Info panel placeholder (Step 8)

**Verification:** `streamlit run app.py` renders the graph; filters re-render correctly.

---

## Step 8 — Node click → info panel

**File:** `app.py`

Node click in PyVis emits a JavaScript event. Bridge to Streamlit via a small JS shim
that writes the selected node ID into a Streamlit component.

- [ ] In PyVis HTML template, inject JS:
  ```js
  network.on("selectNode", function(params) {
      const nodeId = params.nodes[0];
      window.parent.postMessage({type: "node_selected", nodeId: nodeId}, "*");
  });
  ```
- [ ] Wrap PyVis output in `st.components.v1.html` with a companion `st.components.v1.declare_component`
  or use Streamlit's experimental `st.session_state` approach with a custom component.
- [ ] On selection, display below (or beside) the graph:
  - Title (bold), year, `cited_by_count`
  - Domain tag (colored chip), content tags (gray chips)
  - `[Open in Zotero ↗]` button → `zotero://select/library/items/{zotero_key}`
  - `[View on OpenAlex ↗]` link → `https://openalex.org/works/{openalex_id}`

**Note:** Streamlit's iframe boundary makes JS→Python communication tricky. A simpler
v1 alternative is to add a selectbox in the sidebar populated with paper titles,
with the graph serving as a visual guide. The JS postMessage approach is ideal but
may require a small custom Streamlit component.

**Verification:** Clicking (or selecting) a paper populates the info panel correctly.

---

## Step 9 — Polish & README

**File:** `README.md`

- [ ] Write setup instructions:
  1. `pip install -r requirements.txt`
  2. Copy `.env.example` → `.env`, fill in Zotero credentials
  3. Get Zotero API key: https://www.zotero.org/settings/keys
  4. Find library ID: shown in https://www.zotero.org/settings/keys after creating a key
  5. `streamlit run app.py`
- [ ] Document graph tuning parameters in README
- [ ] Add `.gitignore` (exclude `.env`, `cache.db`, `__pycache__`, `.pyvis_temp/`)

---

## Stretch goals (post-v1)

### Graph richness

- **Cited-by edges**: fetch `cited_by` for each paper from OpenAlex (separate paginated
  endpoint per work) and add reverse citation edges. Deferred because cited-by counts
  can be large (thousands) and the cost/benefit vs. forward references is low for an
  in-library graph; worth adding once the core graph is stable.

- **Ghost nodes**: show papers outside the library that are cited by ≥3 library papers.
  Requires fetching OpenAlex metadata for non-library works and rendering them as
  visually distinct (dimmed, dashed border) nodes. Useful as a "paper discovery" feature.

- **Subgraph expansion**: click a ghost node or any node to fetch and render its full
  1-hop citation neighborhood on demand, expanding the graph without a full re-render.

- **Embedding similarity edge**: use a third edge type derived from abstract embeddings
  (OpenAlex provides abstract inverted indexes; reconstruct and embed with a local model
  or the Anthropic embeddings API). Weight = cosine similarity above a threshold. This
  captures relatedness for papers with few or no shared tags.

- **OpenAlex topics overlay**: toggle to color nodes by OpenAlex topic (4-level hierarchy:
  domain → field → subfield → topic) instead of Louvain community. Good cross-check
  against the autotagger's domain tags.

### Identifier coverage

- **Improved DOI resolution for unmatched items**: for items where arXiv DOI construction
  fails and OpenAlex still returns no match, fall back to an OpenAlex title search
  (`/works?search={title}`). Accept the top result only if the title similarity exceeds
  a threshold (e.g. difflib ratio > 0.9) to avoid false matches. This lifts coverage
  for conference papers, book chapters, and preprints with non-standard identifiers.

### Search & filtering

- **Full-text search**: index paper titles and abstracts in SQLite FTS5. Add a search
  bar to the sidebar that highlights matching nodes in the graph and filters the info
  panel. Deferred because it requires rebuilding the DB schema with an FTS virtual table.

### UX & export

- **Export**: download the current filtered subgraph as a CSV edge list, GEXF (Gephi),
  or GraphML file for use in external tools.

- **Auto-refresh**: poll Zotero for new items every N minutes in the background using
  a Streamlit background thread or APScheduler; show a badge when new papers are available.

---

## Risk log

| Risk | Mitigation |
|---|---|
| DOI coverage < 100% | Use arXiv DOI construction as fallback; unmatched papers still appear as nodes |
| PyVis iframe JS boundary limits interactivity | Fall back to sidebar selectbox for paper detail if postMessage proves brittle |
| Pairwise Jaccard slow for large libraries | Cap at n=2000; or use sparse tag-document matrix + sklearn cosine for speedup |
| Zotero API pagination | pyzotero handles this automatically; confirm with `everything()` vs `items()` |
| OpenAlex rate limit | Polite pool + semaphore + sleep; cache aggressively so re-runs are free |
