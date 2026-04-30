"""Tests for fetcher.py — Zotero and OpenAlex sync helpers.

sync_zotero and sync_openalex require live API credentials; they are tested
via their pure helper functions. The async OpenAlex batch fetch is tested with
a mocked httpx client.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zotero_graph.cache import (
    commit,
    get_all_citation_edges,
    get_all_papers,
    init_db,
    upsert_paper,
)
from zotero_graph.fetcher import (
    _arxiv_doi_from_url,
    _clean_doi,
    _fetch_openalex_batch,
    _parse_tags,
    _parse_year,
    _sync_openalex_async,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    c = init_db(db_path)
    yield c
    c.close()
    db_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# _parse_tags
# ---------------------------------------------------------------------------


class TestParseTags:
    def test_empty_list(self):
        assert _parse_tags([]) == (None, [])

    def test_auto_tags_only(self):
        tags = [
            {"tag": "machine-learning", "type": 1},
            {"tag": "transformer", "type": 1},
            {"tag": "attention", "type": 1},
        ]
        domain, content = _parse_tags(tags)
        assert domain == "machine-learning"
        assert content == ["transformer", "attention"]

    def test_manual_tags_ignored(self):
        tags = [
            {"tag": "manual-tag", "type": 0},
            {"tag": "auto-domain", "type": 1},
            {"tag": "auto-content", "type": 1},
        ]
        domain, content = _parse_tags(tags)
        assert domain == "auto-domain"
        assert content == ["auto-content"]

    def test_only_manual_tags_returns_none(self):
        tags = [{"tag": "manual", "type": 0}]
        domain, content = _parse_tags(tags)
        assert domain is None
        assert content == []

    def test_single_auto_tag_is_domain_only(self):
        tags = [{"tag": "neuroscience", "type": 1}]
        domain, content = _parse_tags(tags)
        assert domain == "neuroscience"
        assert content == []

    def test_missing_type_field_excluded(self):
        """Tags without a type key must not be treated as automatic."""
        tags = [{"tag": "no-type-key"}]
        domain, content = _parse_tags(tags)
        assert domain is None
        assert content == []


# ---------------------------------------------------------------------------
# _parse_year
# ---------------------------------------------------------------------------


class TestParseYear:
    def test_none_returns_none(self):
        assert _parse_year(None) is None

    def test_bare_year(self):
        assert _parse_year("2017") == 2017

    def test_full_date(self):
        assert _parse_year("2017-06-12") == 2017

    def test_month_year(self):
        assert _parse_year("June 2017") == 2017

    def test_no_year_returns_none(self):
        assert _parse_year("no year here") is None

    def test_prefers_first_year(self):
        assert _parse_year("Submitted 2016, published 2017") == 2016

    def test_century_boundary_19xx(self):
        assert _parse_year("1998") == 1998

    def test_rejects_non_century_years(self):
        """Years outside 19xx/20xx should not match."""
        assert _parse_year("only 1850") is None


# ---------------------------------------------------------------------------
# _clean_doi
# ---------------------------------------------------------------------------


class TestCleanDoi:
    def test_none_returns_none(self):
        assert _clean_doi(None) is None

    def test_whitespace_only_returns_none(self):
        assert _clean_doi("   ") is None

    def test_bare_doi_unchanged(self):
        assert _clean_doi("10.1234/foo") == "10.1234/foo"

    def test_strips_https_prefix(self):
        assert _clean_doi("https://doi.org/10.1234/foo") == "10.1234/foo"

    def test_strips_http_prefix(self):
        assert _clean_doi("http://doi.org/10.1234/foo") == "10.1234/foo"

    def test_strips_dx_prefix(self):
        assert _clean_doi("https://dx.doi.org/10.1234/foo") == "10.1234/foo"

    def test_strips_whitespace(self):
        assert _clean_doi("  10.1234/foo  ") == "10.1234/foo"


# ---------------------------------------------------------------------------
# _arxiv_doi_from_url
# ---------------------------------------------------------------------------


class TestArxivDoiFromUrl:
    def test_none_url_returns_none(self):
        assert _arxiv_doi_from_url({"url": None}) is None

    def test_missing_url_key_returns_none(self):
        assert _arxiv_doi_from_url({}) is None

    def test_abs_url(self):
        result = _arxiv_doi_from_url({"url": "https://arxiv.org/abs/1706.03762"})
        assert result == "10.48550/arXiv.1706.03762"

    def test_pdf_url(self):
        result = _arxiv_doi_from_url({"url": "https://arxiv.org/pdf/2301.12345"})
        assert result == "10.48550/arXiv.2301.12345"

    def test_versioned_url_strips_version(self):
        result = _arxiv_doi_from_url({"url": "https://arxiv.org/pdf/2301.12345v2"})
        assert result == "10.48550/arXiv.2301.12345"

    def test_non_arxiv_url_returns_none(self):
        assert _arxiv_doi_from_url({"url": "https://example.com/paper"}) is None

    def test_case_insensitive(self):
        result = _arxiv_doi_from_url({"url": "https://ArXiv.org/abs/1706.03762"})
        assert result == "10.48550/arXiv.1706.03762"


# ---------------------------------------------------------------------------
# _fetch_openalex_batch (mocked HTTP)
# ---------------------------------------------------------------------------


class TestFetchOpenalexBatch:
    async def test_returns_results_on_success(self):
        fake_work = {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1/test",
            "title": "Test",
            "referenced_works": [],
            "topics": [],
            "cited_by_count": 5,
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": [fake_work]}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp

        sem = asyncio.Semaphore(1)
        results = await _fetch_openalex_batch(
            mock_client, ["10.1/test"], sem, verbose=False
        )
        assert len(results) == 1
        assert results[0]["title"] == "Test"

    async def test_returns_empty_list_on_http_error(self):
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("connection refused")

        sem = asyncio.Semaphore(1)
        results = await _fetch_openalex_batch(
            mock_client, ["10.1/test"], sem, verbose=False
        )
        assert results == []

    async def test_semaphore_limits_concurrency(self):
        """Verify the semaphore is acquired (i.e. the function awaits it)."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp

        sem = asyncio.Semaphore(0)  # locked — should block
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                _fetch_openalex_batch(mock_client, ["10.1/x"], sem, verbose=False),
                timeout=0.1,
            )


# ---------------------------------------------------------------------------
# _sync_openalex_async — integration with mocked HTTP
# ---------------------------------------------------------------------------


class TestSyncOpenalexAsync:
    async def test_matches_doi_and_writes_citation_edge(self, conn):
        """End-to-end: two papers, one cites the other, citation edge is written."""
        upsert_paper(
            conn,
            {
                "zotero_key": "SRC",
                "title": "Source Paper",
                "doi": "10.1/src",
                "fetched_at": "2026-04-30",
            },
        )
        upsert_paper(
            conn,
            {
                "zotero_key": "TGT",
                "title": "Target Paper",
                "doi": "10.1/tgt",
                "fetched_at": "2026-04-30",
            },
        )
        commit(conn)

        src_work = {
            "id": "https://openalex.org/WSRC",
            "doi": "https://doi.org/10.1/src",
            "title": "Source Paper",
            "referenced_works": ["https://openalex.org/WTGT"],
            "topics": [],
            "cited_by_count": 10,
        }
        tgt_work = {
            "id": "https://openalex.org/WTGT",
            "doi": "https://doi.org/10.1/tgt",
            "title": "Target Paper",
            "referenced_works": [],
            "topics": [],
            "cited_by_count": 50,
        }

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": [src_work, tgt_work]}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("zotero_graph.fetcher.httpx.AsyncClient", return_value=mock_client):
            matched = await _sync_openalex_async(conn, verbose=False)

        assert matched == 2
        papers = {p["zotero_key"]: p for p in get_all_papers(conn)}
        assert papers["SRC"]["openalex_id"] == "WSRC"
        assert papers["TGT"]["openalex_id"] == "WTGT"
        assert papers["SRC"]["cited_by_count"] == 10

        edges = get_all_citation_edges(conn)
        assert ("SRC", "TGT") in edges

    async def test_skips_when_no_missing_papers(self, conn):
        """If all papers already have openalex_id, returns 0 without calling API."""
        from zotero_graph.cache import update_paper_openalex

        upsert_paper(
            conn, {"zotero_key": "A", "title": "A", "fetched_at": "2026-04-30"}
        )
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

        with patch("zotero_graph.fetcher.httpx.AsyncClient") as mock_cls:
            matched = await _sync_openalex_async(conn, verbose=False)

        mock_cls.assert_not_called()
        assert matched == 0
