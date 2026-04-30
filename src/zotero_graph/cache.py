import json
import sqlite3
from pathlib import Path


def init_db(db_path: Path) -> sqlite3.Connection:
    """
    Create the SQLite database and return an open connection.

    Creates the three tables (papers, citation_edges, meta) if they do not
    already exist. Sets row_factory to sqlite3.Row so callers receive
    dict-like rows without a separate conversion step.

    Params:
        db_path: Path : absolute path to the SQLite file (created if absent).
    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            zotero_key      TEXT PRIMARY KEY,
            openalex_id     TEXT,
            doi             TEXT,
            url             TEXT,
            title           TEXT NOT NULL,
            abstract        TEXT,
            year            INTEGER,
            domain_tag      TEXT,
            content_tags    TEXT,
            openalex_topics TEXT,
            cited_by_count  INTEGER,
            fetched_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS citation_edges (
            source_key  TEXT REFERENCES papers(zotero_key),
            target_key  TEXT REFERENCES papers(zotero_key),
            PRIMARY KEY (source_key, target_key)
        );

        CREATE TABLE IF NOT EXISTS meta (
            key     TEXT PRIMARY KEY,
            value   TEXT
        );
    """)
    conn.commit()
    return conn


def upsert_paper(conn: sqlite3.Connection, paper: dict) -> None:
    """
    Insert or update a paper row using COALESCE-safe conflict resolution.

    On conflict, every column except title and fetched_at is updated only if
    the incoming value is non-NULL, so a partial update (e.g. adding only the
    DOI) never overwrites existing data with NULL. title and fetched_at always
    overwrite because they reflect the latest Zotero state.

    content_tags and openalex_topics must be provided as Python lists; they
    are serialised to JSON before insertion.

    Params:
        conn:  sqlite3.Connection : open database connection.
        paper: dict               : must contain 'zotero_key' and 'title';
                                    all other keys are optional.
    Returns:
        None
    """
    conn.execute(
        """
        INSERT INTO papers (
            zotero_key, openalex_id, doi, url, title, abstract, year,
            domain_tag, content_tags, openalex_topics, cited_by_count, fetched_at
        ) VALUES (
            :zotero_key, :openalex_id, :doi, :url, :title, :abstract, :year,
            :domain_tag, :content_tags, :openalex_topics, :cited_by_count, :fetched_at
        )
        ON CONFLICT(zotero_key) DO UPDATE SET
            openalex_id     = COALESCE(excluded.openalex_id, openalex_id),
            doi             = COALESCE(excluded.doi, doi),
            url             = COALESCE(excluded.url, url),
            title           = excluded.title,
            abstract        = COALESCE(excluded.abstract, abstract),
            year            = COALESCE(excluded.year, year),
            domain_tag      = COALESCE(excluded.domain_tag, domain_tag),
            content_tags    = COALESCE(excluded.content_tags, content_tags),
            openalex_topics = COALESCE(excluded.openalex_topics, openalex_topics),
            cited_by_count  = COALESCE(excluded.cited_by_count, cited_by_count),
            fetched_at      = excluded.fetched_at
    """,
        {
            "zotero_key": paper["zotero_key"],
            "openalex_id": paper.get("openalex_id"),
            "doi": paper.get("doi"),
            "url": paper.get("url"),
            "title": paper["title"],
            "abstract": paper.get("abstract"),
            "year": paper.get("year"),
            "domain_tag": paper.get("domain_tag"),
            "content_tags": (
                json.dumps(paper["content_tags"]) if "content_tags" in paper else None
            ),
            "openalex_topics": (
                json.dumps(paper["openalex_topics"])
                if "openalex_topics" in paper
                else None
            ),
            "cited_by_count": paper.get("cited_by_count"),
            "fetched_at": paper.get("fetched_at"),
        },
    )


def update_paper_openalex(
    conn: sqlite3.Connection,
    zotero_key: str,
    *,
    openalex_id: str,
    openalex_topics: list,
    cited_by_count: int | None,
    fetched_at: str,
) -> None:
    """
    Update only the four OpenAlex-derived columns for an existing paper row.

    Used instead of upsert_paper during OpenAlex enrichment to avoid
    overwriting the Zotero title with a potentially blank OpenAlex title.

    Params:
        conn:            sqlite3.Connection : open database connection.
        zotero_key:      str                : primary key of the row to update.
        openalex_id:     str                : OpenAlex work ID (e.g. 'W2963403868').
        openalex_topics: list               : list of topic dicts {id, name, score}.
        cited_by_count:  int | None         : total citation count from OpenAlex.
        fetched_at:      str                : ISO timestamp of the fetch.
    Returns:
        None
    """
    conn.execute(
        """
        UPDATE papers
        SET openalex_id     = ?,
            openalex_topics = ?,
            cited_by_count  = ?,
            fetched_at      = ?
        WHERE zotero_key = ?
    """,
        (
            openalex_id,
            json.dumps(openalex_topics),
            cited_by_count,
            fetched_at,
            zotero_key,
        ),
    )


def get_all_papers(conn: sqlite3.Connection) -> list[dict]:
    """
    Return all rows from the papers table as deserialised dicts.

    JSON columns (content_tags, openalex_topics) are decoded to Python lists.

    Params:
        conn: sqlite3.Connection : open database connection.
    Returns:
        list[dict] where each dict mirrors the papers table schema.
    """
    rows = conn.execute("SELECT * FROM papers").fetchall()
    return [_deserialize(dict(r)) for r in rows]


def get_papers_missing_openalex(conn: sqlite3.Connection) -> list[dict]:
    """
    Return papers that have not yet been matched to an OpenAlex work.

    Params:
        conn: sqlite3.Connection : open database connection.
    Returns:
        list[dict] of paper rows where openalex_id IS NULL.
    """
    rows = conn.execute("SELECT * FROM papers WHERE openalex_id IS NULL").fetchall()
    return [_deserialize(dict(r)) for r in rows]


def upsert_citation_edge(
    conn: sqlite3.Connection, source_key: str, target_key: str
) -> None:
    """
    Insert a directed citation edge, silently ignoring duplicates.

    Only in-library edges (both keys present in the papers table) should be
    inserted; enforcement is left to the caller.

    Params:
        conn:       sqlite3.Connection : open database connection.
        source_key: str                : zotero_key of the citing paper.
        target_key: str                : zotero_key of the cited paper.
    Returns:
        None
    """
    conn.execute(
        "INSERT OR IGNORE INTO citation_edges (source_key, target_key) VALUES (?, ?)",
        (source_key, target_key),
    )


def get_all_citation_edges(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """
    Return all in-library citation edges as (source_key, target_key) pairs.

    Params:
        conn: sqlite3.Connection : open database connection.
    Returns:
        list[tuple[str, str]] of (source_key, target_key) pairs.
    """
    rows = conn.execute("SELECT source_key, target_key FROM citation_edges").fetchall()
    return [(r[0], r[1]) for r in rows]


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """
    Upsert a key-value pair in the meta table.

    Params:
        conn:  sqlite3.Connection : open database connection.
        key:   str                : metadata key (e.g. 'last_zotero_sync').
        value: str                : metadata value.
    Returns:
        None
    """
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """
    Retrieve a metadata value by key, returning None if absent.

    Params:
        conn: sqlite3.Connection : open database connection.
        key:  str                : metadata key to look up.
    Returns:
        str value if the key exists, otherwise None.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def commit(conn: sqlite3.Connection) -> None:
    """
    Commit the current transaction.

    Callers batch multiple writes before calling commit() once, rather than
    auto-committing per row, for performance.

    Params:
        conn: sqlite3.Connection : open database connection.
    Returns:
        None
    """
    conn.commit()


def _deserialize(row: dict) -> dict:
    """
    Decode JSON-serialised list columns in a paper row dict in place.

    content_tags is decoded to a list (defaulting to []); openalex_topics is
    decoded to a list (defaulting to None when absent).

    Params:
        row: dict : raw paper row with JSON strings for list columns.
    Returns:
        dict with content_tags and openalex_topics as Python objects.
    """
    for field in ("content_tags", "openalex_topics"):
        if row.get(field) is not None:
            row[field] = json.loads(row[field])
        else:
            row[field] = [] if field == "content_tags" else None
    return row
