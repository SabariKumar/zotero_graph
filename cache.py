import json
import sqlite3
from pathlib import Path


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            zotero_key      TEXT PRIMARY KEY,
            openalex_id     TEXT,
            doi             TEXT,
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
    conn.execute("""
        INSERT INTO papers (
            zotero_key, openalex_id, doi, title, abstract, year,
            domain_tag, content_tags, openalex_topics, cited_by_count, fetched_at
        ) VALUES (
            :zotero_key, :openalex_id, :doi, :title, :abstract, :year,
            :domain_tag, :content_tags, :openalex_topics, :cited_by_count, :fetched_at
        )
        ON CONFLICT(zotero_key) DO UPDATE SET
            openalex_id     = COALESCE(excluded.openalex_id, openalex_id),
            doi             = COALESCE(excluded.doi, doi),
            title           = excluded.title,
            abstract        = COALESCE(excluded.abstract, abstract),
            year            = COALESCE(excluded.year, year),
            domain_tag      = COALESCE(excluded.domain_tag, domain_tag),
            content_tags    = COALESCE(excluded.content_tags, content_tags),
            openalex_topics = COALESCE(excluded.openalex_topics, openalex_topics),
            cited_by_count  = COALESCE(excluded.cited_by_count, cited_by_count),
            fetched_at      = excluded.fetched_at
    """, {
        "zotero_key":      paper["zotero_key"],
        "openalex_id":     paper.get("openalex_id"),
        "doi":             paper.get("doi"),
        "title":           paper["title"],
        "abstract":        paper.get("abstract"),
        "year":            paper.get("year"),
        "domain_tag":      paper.get("domain_tag"),
        "content_tags":    json.dumps(paper["content_tags"]) if "content_tags" in paper else None,
        "openalex_topics": json.dumps(paper["openalex_topics"]) if "openalex_topics" in paper else None,
        "cited_by_count":  paper.get("cited_by_count"),
        "fetched_at":      paper.get("fetched_at"),
    })


def get_all_papers(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM papers").fetchall()
    return [_deserialize(dict(r)) for r in rows]


def get_papers_missing_openalex(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM papers WHERE openalex_id IS NULL"
    ).fetchall()
    return [_deserialize(dict(r)) for r in rows]


def upsert_citation_edge(conn: sqlite3.Connection, source_key: str, target_key: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO citation_edges (source_key, target_key) VALUES (?, ?)",
        (source_key, target_key),
    )


def get_all_citation_edges(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    rows = conn.execute("SELECT source_key, target_key FROM citation_edges").fetchall()
    return [(r[0], r[1]) for r in rows]


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def commit(conn: sqlite3.Connection) -> None:
    conn.commit()


def _deserialize(row: dict) -> dict:
    for field in ("content_tags", "openalex_topics"):
        if row.get(field) is not None:
            row[field] = json.loads(row[field])
        else:
            row[field] = [] if field == "content_tags" else None
    return row
