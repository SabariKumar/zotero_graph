"""Tests for cache.py — SQLite read/write helpers."""

import tempfile
from pathlib import Path

import pytest

from zotero_graph.cache import (
    _deserialize,
    commit,
    get_all_citation_edges,
    get_all_papers,
    get_meta,
    get_papers_missing_openalex,
    init_db,
    set_meta,
    update_paper_openalex,
    upsert_citation_edge,
    upsert_paper,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    """Provide a fresh in-memory (temp-file) database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    c = init_db(db_path)
    yield c
    c.close()
    db_path.unlink(missing_ok=True)


def _paper(**overrides) -> dict:
    """Return a minimal valid paper dict, with optional overrides."""
    base = {
        "zotero_key": "ABC123",
        "title": "Test Paper",
        "year": 2022,
        "domain_tag": "machine-learning",
        "content_tags": ["transformer", "attention"],
        "fetched_at": "2026-04-30T00:00:00",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_creates_papers_table(self, conn):
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "papers" in tables

    def test_creates_citation_edges_table(self, conn):
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "citation_edges" in tables

    def test_creates_meta_table(self, conn):
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "meta" in tables

    def test_idempotent(self, conn):
        """Calling init_db on an existing DB must not raise."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        c1 = init_db(db_path)
        c2 = init_db(db_path)
        c1.close()
        c2.close()
        db_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# upsert_paper / get_all_papers
# ---------------------------------------------------------------------------


class TestUpsertPaper:
    def test_insert_and_retrieve(self, conn):
        upsert_paper(conn, _paper())
        commit(conn)
        papers = get_all_papers(conn)
        assert len(papers) == 1
        assert papers[0]["zotero_key"] == "ABC123"
        assert papers[0]["title"] == "Test Paper"

    def test_content_tags_roundtrip(self, conn):
        upsert_paper(conn, _paper(content_tags=["tag-a", "tag-b", "tag-c"]))
        commit(conn)
        p = get_all_papers(conn)[0]
        assert p["content_tags"] == ["tag-a", "tag-b", "tag-c"]

    def test_empty_content_tags_roundtrip(self, conn):
        upsert_paper(conn, _paper(content_tags=[]))
        commit(conn)
        p = get_all_papers(conn)[0]
        assert p["content_tags"] == []

    def test_coalesce_does_not_overwrite_with_none(self, conn):
        """A partial upsert with no DOI must not erase an existing DOI."""
        upsert_paper(conn, _paper(doi="10.1234/original"))
        commit(conn)
        # Second upsert omits doi
        upsert_paper(conn, _paper(doi=None))
        commit(conn)
        p = get_all_papers(conn)[0]
        assert p["doi"] == "10.1234/original"

    def test_title_always_overwrites(self, conn):
        upsert_paper(conn, _paper(title="Old Title"))
        commit(conn)
        upsert_paper(conn, _paper(title="New Title"))
        commit(conn)
        p = get_all_papers(conn)[0]
        assert p["title"] == "New Title"

    def test_multiple_papers(self, conn):
        upsert_paper(conn, _paper(zotero_key="A", title="Alpha"))
        upsert_paper(conn, _paper(zotero_key="B", title="Beta"))
        commit(conn)
        assert len(get_all_papers(conn)) == 2


# ---------------------------------------------------------------------------
# update_paper_openalex
# ---------------------------------------------------------------------------


class TestUpdatePaperOpenalex:
    def test_updates_openalex_fields(self, conn):
        upsert_paper(conn, _paper())
        commit(conn)
        update_paper_openalex(
            conn,
            "ABC123",
            openalex_id="W123",
            openalex_topics=[{"id": "T1", "name": "ML", "score": 0.9}],
            cited_by_count=500,
            fetched_at="2026-04-30T12:00:00",
        )
        commit(conn)
        p = get_all_papers(conn)[0]
        assert p["openalex_id"] == "W123"
        assert p["cited_by_count"] == 500
        assert p["openalex_topics"][0]["name"] == "ML"

    def test_does_not_overwrite_title(self, conn):
        """update_paper_openalex must never touch the title column."""
        upsert_paper(conn, _paper(title="Original Zotero Title"))
        commit(conn)
        update_paper_openalex(
            conn,
            "ABC123",
            openalex_id="W123",
            openalex_topics=[],
            cited_by_count=None,
            fetched_at="2026-04-30T12:00:00",
        )
        commit(conn)
        p = get_all_papers(conn)[0]
        assert p["title"] == "Original Zotero Title"


# ---------------------------------------------------------------------------
# get_papers_missing_openalex
# ---------------------------------------------------------------------------


class TestGetPapersMissingOpenalex:
    def test_returns_only_unmatched(self, conn):
        upsert_paper(conn, _paper(zotero_key="A"))
        upsert_paper(conn, _paper(zotero_key="B"))
        commit(conn)
        update_paper_openalex(
            conn,
            "A",
            openalex_id="W1",
            openalex_topics=[],
            cited_by_count=None,
            fetched_at="2026-04-30",
        )
        commit(conn)
        missing = get_papers_missing_openalex(conn)
        assert len(missing) == 1
        assert missing[0]["zotero_key"] == "B"

    def test_empty_when_all_matched(self, conn):
        upsert_paper(conn, _paper())
        commit(conn)
        update_paper_openalex(
            conn,
            "ABC123",
            openalex_id="W1",
            openalex_topics=[],
            cited_by_count=None,
            fetched_at="2026-04-30",
        )
        commit(conn)
        assert get_papers_missing_openalex(conn) == []


# ---------------------------------------------------------------------------
# citation edges
# ---------------------------------------------------------------------------


class TestCitationEdges:
    def test_insert_and_retrieve(self, conn):
        upsert_paper(conn, _paper(zotero_key="A"))
        upsert_paper(conn, _paper(zotero_key="B"))
        commit(conn)
        upsert_citation_edge(conn, "A", "B")
        commit(conn)
        edges = get_all_citation_edges(conn)
        assert ("A", "B") in edges

    def test_duplicate_silently_ignored(self, conn):
        upsert_paper(conn, _paper(zotero_key="A"))
        upsert_paper(conn, _paper(zotero_key="B"))
        commit(conn)
        upsert_citation_edge(conn, "A", "B")
        upsert_citation_edge(conn, "A", "B")  # duplicate
        commit(conn)
        assert len(get_all_citation_edges(conn)) == 1

    def test_directed(self, conn):
        """(A→B) and (B→A) are distinct edges."""
        upsert_paper(conn, _paper(zotero_key="A"))
        upsert_paper(conn, _paper(zotero_key="B"))
        commit(conn)
        upsert_citation_edge(conn, "A", "B")
        upsert_citation_edge(conn, "B", "A")
        commit(conn)
        edges = get_all_citation_edges(conn)
        assert len(edges) == 2


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------


class TestMeta:
    def test_set_and_get(self, conn):
        set_meta(conn, "last_sync", "2026-04-30")
        commit(conn)
        assert get_meta(conn, "last_sync") == "2026-04-30"

    def test_get_missing_returns_none(self, conn):
        assert get_meta(conn, "nonexistent") is None

    def test_upsert_overwrites(self, conn):
        set_meta(conn, "k", "v1")
        commit(conn)
        set_meta(conn, "k", "v2")
        commit(conn)
        assert get_meta(conn, "k") == "v2"


# ---------------------------------------------------------------------------
# _deserialize
# ---------------------------------------------------------------------------


class TestDeserialize:
    def test_decodes_content_tags(self):
        row = {"content_tags": '["a", "b"]', "openalex_topics": None}
        result = _deserialize(row)
        assert result["content_tags"] == ["a", "b"]

    def test_null_content_tags_becomes_empty_list(self):
        row = {"content_tags": None, "openalex_topics": None}
        result = _deserialize(row)
        assert result["content_tags"] == []

    def test_decodes_openalex_topics(self):
        row = {
            "content_tags": "[]",
            "openalex_topics": '[{"id": "T1", "name": "ML", "score": 0.9}]',
        }
        result = _deserialize(row)
        assert result["openalex_topics"][0]["name"] == "ML"

    def test_null_openalex_topics_stays_none(self):
        row = {"content_tags": "[]", "openalex_topics": None}
        result = _deserialize(row)
        assert result["openalex_topics"] is None
