"""
fetcher.py — Zotero and OpenAlex sync pipeline.

Stage 1 (sync_zotero):  pull all items from the Zotero Web API and upsert
                         into the local SQLite cache.
Stage 2 (sync_openalex): enrich cached papers with OpenAlex IDs, topics,
                         cited_by_count, and build in-library citation edges.
"""

import asyncio
import json
import re
import sqlite3
from datetime import datetime, timezone

import httpx
from pyzotero import zotero

from cache import (
    commit,
    get_papers_missing_openalex,
    get_all_papers,
    set_meta,
    update_paper_openalex,
    upsert_citation_edge,
    upsert_paper,
)
from config import (
    OPENALEX_BATCH_DELAY,
    OPENALEX_BATCH_SIZE,
    OPENALEX_CONCURRENCY,
    OPENALEX_EMAIL,
    ZOTERO_API_KEY,
    ZOTERO_LIBRARY_ID,
    ZOTERO_LIBRARY_TYPE,
)

# Item types to pull from Zotero (excludes attachments, notes, etc.)
ZOTERO_ITEM_TYPES = [
    "journalArticle",
    "preprint",
    "bookSection",
    "conferencePaper",
    "report",
    "thesis",
]

# Regex to extract a 4-digit year from Zotero's free-form date field
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Regex to extract arXiv ID from a URL (strips version suffix, e.g. v2)
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Stage 1 — Zotero
# ---------------------------------------------------------------------------

def sync_zotero(conn: sqlite3.Connection, *, verbose: bool = True) -> int:
    """
    Pull all citable items from the Zotero Web API and upsert them into the cache.

    Fetches all pages of results using pyzotero's everything() helper, extracts
    metadata and auto-tags written by the autotagger plugin, then upserts each
    item. Items with no title are skipped. Records last_zotero_sync in meta.

    Params:
        conn:    sqlite3.Connection : open database connection.
        verbose: bool               : if True, print progress to stdout.
    Returns:
        int number of items upserted (untitled items not counted).
    """
    zot = zotero.Zotero(ZOTERO_LIBRARY_ID, ZOTERO_LIBRARY_TYPE, ZOTERO_API_KEY)
    item_type_filter = ",".join(ZOTERO_ITEM_TYPES)

    if verbose:
        print("Fetching items from Zotero…")

    items = zot.everything(zot.items(itemType=item_type_filter))

    if verbose:
        print(f"  {len(items)} items retrieved")

    now = datetime.now(timezone.utc).isoformat()
    count = 0

    for item in items:
        data = item.get("data", {})
        title = (data.get("title") or "").strip()
        if not title:
            continue

        key = item["key"]
        domain_tag, content_tags = _parse_tags(data.get("tags", []))

        upsert_paper(conn, {
            "zotero_key":   key,
            "title":        title,
            "abstract":     data.get("abstractNote") or None,
            "doi":          _clean_doi(data.get("DOI")),
            "url":          data.get("url") or None,
            "year":         _parse_year(data.get("date")),
            "domain_tag":   domain_tag,
            "content_tags": content_tags,
            "fetched_at":   now,
        })
        count += 1

    commit(conn)
    set_meta(conn, "last_zotero_sync", now)
    commit(conn)

    if verbose:
        print(f"  {count} items upserted into cache")

    return count


def _parse_tags(raw_tags: list[dict]) -> tuple[str | None, list[str]]:
    """
    Split a Zotero tag list into (domain_tag, content_tags).

    The autotagger plugin writes automatic tags (type=1) in a fixed order: the
    broad domain tag first, then 5–12 specific content tags. Manual tags
    (type=0) are ignored entirely so user-applied tags do not pollute the graph.

    Params:
        raw_tags: list[dict] : Zotero tag objects, each with 'tag' and 'type' keys.
    Returns:
        tuple of (domain_tag: str | None, content_tags: list[str]).
    """
    auto = [t["tag"] for t in raw_tags if t.get("type") == 1]
    if not auto:
        return None, []
    return auto[0], auto[1:]


def _parse_year(date_str: str | None) -> int | None:
    """
    Extract a 4-digit year from Zotero's free-form date field.

    Zotero stores dates as user-entered strings ('2017', 'June 2017',
    '2017-06-12', etc.). The first 19xx/20xx sequence found is returned.

    Params:
        date_str: str | None : raw Zotero date string.
    Returns:
        int year if found, otherwise None.
    """
    if not date_str:
        return None
    m = _YEAR_RE.search(date_str)
    return int(m.group()) if m else None


def _clean_doi(doi: str | None) -> str | None:
    """
    Normalise a DOI to bare form, stripping any https://doi.org/ prefix.

    Params:
        doi: str | None : raw DOI string from Zotero.
    Returns:
        str bare DOI (e.g. '10.1038/nature14539'), or None if input is empty.
    """
    if not doi:
        return None
    doi = doi.strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi or None


# ---------------------------------------------------------------------------
# Stage 2 — OpenAlex
# ---------------------------------------------------------------------------

def sync_openalex(conn: sqlite3.Connection, *, verbose: bool = True) -> int:
    """
    Enrich cached papers with OpenAlex metadata and build in-library citation edges.

    Runs the async pipeline synchronously via asyncio.run(). Collects all papers
    lacking an openalex_id, batches their DOIs, fans out concurrent requests to
    the OpenAlex API, then resolves citation edges between library members.

    Params:
        conn:    sqlite3.Connection : open database connection.
        verbose: bool               : if True, print progress to stdout.
    Returns:
        int number of papers successfully matched to an OpenAlex work.
    """
    return asyncio.run(_sync_openalex_async(conn, verbose=verbose))


async def _sync_openalex_async(conn: sqlite3.Connection, *, verbose: bool) -> int:
    """
    Async implementation of the OpenAlex enrichment pipeline.

    Batches DOIs into groups of OPENALEX_BATCH_SIZE, fans them out with
    asyncio.gather bounded by a semaphore, then builds citation edges by
    intersecting each paper's referenced_works with the library's OpenAlex IDs.

    Params:
        conn:    sqlite3.Connection : open database connection.
        verbose: bool               : if True, print progress to stdout.
    Returns:
        int number of papers matched.
    """
    papers = get_papers_missing_openalex(conn)
    if not papers:
        if verbose:
            print("No papers missing OpenAlex data — skipping")
        return 0

    if verbose:
        print(f"Fetching OpenAlex data for {len(papers)} papers…")

    doi_map: dict[str, str] = {}  # normalised doi → zotero_key
    no_doi: list[dict] = []
    for p in papers:
        doi = p.get("doi") or _arxiv_doi_from_url(p)
        if doi:
            doi_map[doi.lower()] = p["zotero_key"]
        else:
            no_doi.append(p)

    if verbose and no_doi:
        print(f"  {len(no_doi)} papers have no DOI and will remain unmatched for now")

    dois = list(doi_map.keys())
    batches = [dois[i:i + OPENALEX_BATCH_SIZE] for i in range(0, len(dois), OPENALEX_BATCH_SIZE)]

    sem = asyncio.Semaphore(OPENALEX_CONCURRENCY)
    matched = 0

    # {zotero_key: [openalex_id, ...]} — accumulated across all batches
    references: dict[str, list[str]] = {}

    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [
            _fetch_openalex_batch(client, batch, sem, verbose=verbose)
            for batch in batches
        ]
        batch_results = await asyncio.gather(*tasks)

    now = datetime.now(timezone.utc).isoformat()
    for works in batch_results:
        for work in works:
            doi_key = (work.get("doi") or "").lower()
            doi_key = re.sub(r"^https?://doi\.org/", "", doi_key)
            zotero_key = doi_map.get(doi_key)
            if not zotero_key:
                continue

            oa_id = work.get("id", "").replace("https://openalex.org/", "")
            topics = [
                {"id": t["id"], "name": t["display_name"], "score": t.get("score")}
                for t in work.get("topics", [])
            ]

            update_paper_openalex(
                conn,
                zotero_key,
                openalex_id=oa_id,
                openalex_topics=topics,
                cited_by_count=work.get("cited_by_count"),
                fetched_at=now,
            )
            references[zotero_key] = [
                r.replace("https://openalex.org/", "")
                for r in work.get("referenced_works", [])
            ]
            matched += 1

    commit(conn)

    # Build in-library citation edges
    all_papers = get_all_papers(conn)
    oa_id_to_key = {
        p["openalex_id"]: p["zotero_key"]
        for p in all_papers
        if p.get("openalex_id")
    }

    edge_count = 0
    for source_key, ref_ids in references.items():
        for ref_id in ref_ids:
            target_key = oa_id_to_key.get(ref_id)
            if target_key and target_key != source_key:
                upsert_citation_edge(conn, source_key, target_key)
                edge_count += 1

    now = datetime.now(timezone.utc).isoformat()
    set_meta(conn, "last_openalex_sync", now)
    commit(conn)

    if verbose:
        print(f"  {matched}/{len(dois)} papers matched to OpenAlex works")
        print(f"  {edge_count} in-library citation edges written")

    return matched


async def _fetch_openalex_batch(
    client: httpx.AsyncClient,
    dois: list[str],
    sem: asyncio.Semaphore,
    *,
    verbose: bool,
) -> list[dict]:
    """
    Fetch a batch of works from the OpenAlex API filtered by DOI.

    Acquires the shared semaphore before making the request to cap concurrency
    at OPENALEX_CONCURRENCY. Sleeps OPENALEX_BATCH_DELAY seconds inside the
    semaphore slot to stay within the polite-pool rate limit. Returns an empty
    list on any HTTP or network error without raising.

    Params:
        client:  httpx.AsyncClient  : shared async HTTP client.
        dois:    list[str]          : normalised DOIs (bare, lowercase).
        sem:     asyncio.Semaphore  : concurrency limiter shared across all batches.
        verbose: bool               : if True, print errors to stdout.
    Returns:
        list[dict] of OpenAlex work objects, or [] on error.
    """
    async with sem:
        await asyncio.sleep(OPENALEX_BATCH_DELAY)
        filter_str = "doi:" + "|".join(dois)
        params = {
            "filter": filter_str,
            "select": "id,doi,title,referenced_works,topics,cited_by_count",
            "per-page": len(dois),
            "mailto": OPENALEX_EMAIL,
        }
        try:
            resp = await client.get("https://api.openalex.org/works", params=params)
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as exc:
            if verbose:
                print(f"  OpenAlex batch error: {exc}")
            return []


def _arxiv_doi_from_url(paper: dict) -> str | None:
    """
    Construct a canonical DOI for an arXiv preprint from its stored URL.

    The autotagger stores the arXiv abstract URL in the Zotero url field.
    OpenAlex indexes arXiv papers under the DOI 10.48550/arXiv.{id}. The
    version suffix (e.g. v2) is stripped so the bare ID is used.

    Note: coverage is partial — some arXiv-only papers are not indexed by
    OpenAlex under this DOI scheme. See PLAN.md for details.

    Params:
        paper: dict : paper dict with at least a 'url' key.
    Returns:
        str DOI of the form '10.48550/arXiv.XXXX.XXXXX', or None if not arXiv.
    """
    url = paper.get("url") or ""
    m = _ARXIV_ID_RE.search(url)
    if m:
        return f"10.48550/arXiv.{m.group(1)}"
    return None
