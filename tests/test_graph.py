"""Tests for graph.py — NetworkX graph construction, stats, and PyVis rendering."""

import tempfile
from pathlib import Path

import pytest

from cache import commit, init_db, upsert_citation_edge, upsert_paper
from graph import (
    _apply_filters,
    _jaccard,
    _node_size,
    build_graph,
    graph_stats,
    render_pyvis,
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


def _seed(conn, papers, edges=None):
    """Insert papers and optional citation edges, then commit."""
    for p in papers:
        upsert_paper(conn, {"fetched_at": "2026-04-30", **p})
    if edges:
        for src, tgt in edges:
            upsert_citation_edge(conn, src, tgt)
    commit(conn)


# ---------------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical_sets(self):
        a = frozenset(["a", "b", "c"])
        assert _jaccard(a, a) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard(frozenset(["a"]), frozenset(["b"])) == 0.0

    def test_partial_overlap(self):
        a = frozenset(["a", "b"])
        b = frozenset(["b", "c"])
        # |intersection|=1, |union|=3
        assert _jaccard(a, b) == pytest.approx(1 / 3)

    def test_both_empty(self):
        assert _jaccard(frozenset(), frozenset()) == 0.0

    def test_one_empty(self):
        assert _jaccard(frozenset(["a"]), frozenset()) == 0.0

    def test_subset(self):
        a = frozenset(["a"])
        b = frozenset(["a", "b", "c"])
        assert _jaccard(a, b) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# _node_size
# ---------------------------------------------------------------------------


class TestNodeSize:
    def test_zero_citations(self):
        assert _node_size(0) == pytest.approx(10.0)

    def test_positive_citations_larger(self):
        assert _node_size(100) > _node_size(0)

    def test_monotone(self):
        sizes = [_node_size(n) for n in [0, 10, 100, 1000, 10000]]
        assert sizes == sorted(sizes)

    def test_minimum_ten(self):
        assert _node_size(0) >= 10.0


# ---------------------------------------------------------------------------
# _apply_filters
# ---------------------------------------------------------------------------


class TestApplyFilters:
    _papers = [
        {
            "zotero_key": "A",
            "title": "A",
            "domain_tag": "ml",
            "year": 2020,
            "content_tags": ["transformer"],
        },
        {
            "zotero_key": "B",
            "title": "B",
            "domain_tag": "bio",
            "year": 2021,
            "content_tags": ["protein"],
        },
        {
            "zotero_key": "C",
            "title": "C",
            "domain_tag": "ml",
            "year": 2022,
            "content_tags": ["diffusion-model"],
        },
    ]

    def test_no_filters_returns_all(self):
        assert len(_apply_filters(self._papers, None, None, None, None)) == 3

    def test_domain_filter(self):
        result = _apply_filters(self._papers, ["ml"], None, None, None)
        assert {p["zotero_key"] for p in result} == {"A", "C"}

    def test_year_min(self):
        result = _apply_filters(self._papers, None, 2021, None, None)
        assert {p["zotero_key"] for p in result} == {"B", "C"}

    def test_year_max(self):
        result = _apply_filters(self._papers, None, None, 2021, None)
        assert {p["zotero_key"] for p in result} == {"A", "B"}

    def test_year_range(self):
        result = _apply_filters(self._papers, None, 2021, 2021, None)
        assert {p["zotero_key"] for p in result} == {"B"}

    def test_tag_query_substring(self):
        result = _apply_filters(self._papers, None, None, None, "diffusion")
        assert {p["zotero_key"] for p in result} == {"C"}

    def test_tag_query_matches_domain(self):
        result = _apply_filters(self._papers, None, None, None, "bio")
        assert {p["zotero_key"] for p in result} == {"B"}

    def test_combined_filters(self):
        result = _apply_filters(self._papers, ["ml"], 2021, None, None)
        assert {p["zotero_key"] for p in result} == {"C"}

    def test_empty_result(self):
        result = _apply_filters(self._papers, ["physics"], None, None, None)
        assert result == []

    def test_missing_year_treated_as_zero_for_min(self):
        papers = [
            {
                "zotero_key": "X",
                "title": "X",
                "domain_tag": "ml",
                "year": None,
                "content_tags": [],
            }
        ]
        result = _apply_filters(papers, None, 2020, None, None)
        assert result == []  # year=None → 0 < 2020

    def test_missing_year_excluded_by_year_max(self):
        """year=None uses sentinel 9999, so it fails year_max ≤ 2020."""
        papers = [
            {
                "zotero_key": "X",
                "title": "X",
                "domain_tag": "ml",
                "year": None,
                "content_tags": [],
            }
        ]
        result = _apply_filters(papers, None, None, 2020, None)
        # 9999 <= 2020 is False → excluded
        assert result == []


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_node_count(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "content_tags": ["x"],
                },
                {
                    "zotero_key": "B",
                    "title": "B",
                    "domain_tag": "ml",
                    "content_tags": ["x"],
                },
            ],
        )
        G, _ = build_graph(conn)
        assert G.number_of_nodes() == 2

    def test_tag_jaccard_edge_created(self, conn):
        """Two papers with 50% tag overlap must share an edge above MIN_JACCARD."""
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "content_tags": ["transformer", "attention"],
                },
                {
                    "zotero_key": "B",
                    "title": "B",
                    "domain_tag": "ml",
                    "content_tags": ["transformer", "language-model"],
                },
            ],
        )
        G, _ = build_graph(conn)
        assert G.has_edge("A", "B")
        assert G["A"]["B"]["tag_weight"] == pytest.approx(1 / 3)

    def test_no_edge_for_disjoint_tags(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "content_tags": ["x", "y"],
                },
                {
                    "zotero_key": "B",
                    "title": "B",
                    "domain_tag": "ml",
                    "content_tags": ["p", "q"],
                },
            ],
        )
        G, _ = build_graph(conn)
        assert not G.has_edge("A", "B")

    def test_citation_edge_adds_bonus(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "content_tags": ["transformer", "attention"],
                },
                {
                    "zotero_key": "B",
                    "title": "B",
                    "domain_tag": "ml",
                    "content_tags": ["transformer", "language-model"],
                },
            ],
            edges=[("B", "A")],
        )
        G, _ = build_graph(conn)
        assert G["A"]["B"]["citation"] is True
        # weight must be higher than tag-only
        jaccard = 1 / 3
        assert G["A"]["B"]["weight"] > jaccard

    def test_citation_only_edge_created(self, conn):
        """Papers with no shared tags but a citation link get a citation-only edge."""
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "content_tags": ["x"],
                },
                {
                    "zotero_key": "B",
                    "title": "B",
                    "domain_tag": "ml",
                    "content_tags": ["y"],
                },
            ],
            edges=[("A", "B")],
        )
        G, _ = build_graph(conn, min_edge_weight=0.0)
        assert G.has_edge("A", "B")
        assert G["A"]["B"]["citation"] is True
        assert G["A"]["B"]["tag_weight"] == 0.0

    def test_weak_edges_dropped(self, conn):
        """An edge just below min_edge_weight must be dropped."""
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "content_tags": ["a", "b", "c", "d", "e", "f", "g"],
                },
                {
                    "zotero_key": "B",
                    "title": "B",
                    "domain_tag": "ml",
                    "content_tags": ["a", "b", "b2", "b3", "b4", "b5", "b6"],
                },
            ],
        )
        G_tight, _ = build_graph(conn, min_edge_weight=0.99)
        assert not G_tight.has_edge("A", "B")

    def test_community_assigned(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "content_tags": ["transformer", "attention"],
                },
                {
                    "zotero_key": "B",
                    "title": "B",
                    "domain_tag": "ml",
                    "content_tags": ["transformer", "language-model"],
                },
            ],
        )
        G, community_map = build_graph(conn)
        assert "A" in community_map
        assert "B" in community_map
        assert isinstance(community_map["A"], int)

    def test_domain_filter(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "content_tags": [],
                },
                {
                    "zotero_key": "B",
                    "title": "B",
                    "domain_tag": "bio",
                    "content_tags": [],
                },
            ],
        )
        G, _ = build_graph(conn, domain_filter=["ml"])
        assert set(G.nodes) == {"A"}

    def test_year_filter(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "year": 2019,
                    "content_tags": [],
                },
                {
                    "zotero_key": "B",
                    "title": "B",
                    "domain_tag": "ml",
                    "year": 2022,
                    "content_tags": [],
                },
            ],
        )
        G, _ = build_graph(conn, year_min=2020)
        assert set(G.nodes) == {"B"}

    def test_empty_graph(self, conn):
        G, community_map = build_graph(conn)
        assert G.number_of_nodes() == 0
        assert community_map == {}


# ---------------------------------------------------------------------------
# graph_stats
# ---------------------------------------------------------------------------


class TestGraphStats:
    def test_correct_node_count(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "content_tags": ["x", "y"],
                },
                {
                    "zotero_key": "B",
                    "title": "B",
                    "domain_tag": "ml",
                    "content_tags": ["x", "z"],
                },
            ],
        )
        G, _ = build_graph(conn)
        stats = graph_stats(G)
        assert stats["n_papers"] == 2

    def test_citation_edge_counted(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "content_tags": ["transformer", "attention"],
                },
                {
                    "zotero_key": "B",
                    "title": "B",
                    "domain_tag": "ml",
                    "content_tags": ["transformer", "language-model"],
                },
            ],
            edges=[("B", "A")],
        )
        G, _ = build_graph(conn)
        stats = graph_stats(G)
        assert stats["citation_edges"] >= 1

    def test_top_tags_ordered_by_frequency(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "content_tags": ["common", "rare-a"],
                },
                {
                    "zotero_key": "B",
                    "title": "B",
                    "domain_tag": "ml",
                    "content_tags": ["common", "rare-b"],
                },
                {
                    "zotero_key": "C",
                    "title": "C",
                    "domain_tag": "ml",
                    "content_tags": ["common", "rare-c"],
                },
            ],
        )
        G, _ = build_graph(conn)
        stats = graph_stats(G)
        top_tag_names = [t for t, _ in stats["top_tags"]]
        assert top_tag_names[0] == "common"

    def test_empty_graph_stats(self, conn):
        import networkx as nx

        stats = graph_stats(nx.Graph())
        assert stats["n_papers"] == 0
        assert stats["n_edges"] == 0
        assert stats["top_tags"] == []


# ---------------------------------------------------------------------------
# render_pyvis
# ---------------------------------------------------------------------------


class TestRenderPyvis:
    def test_creates_html_file(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "Alpha Paper",
                    "domain_tag": "ml",
                    "content_tags": ["transformer"],
                    "cited_by_count": 100,
                },
            ],
        )
        G, _ = build_graph(conn)
        path = render_pyvis(G)
        assert path.exists()
        assert path.suffix == ".html"
        assert path.stat().st_size > 500
        path.unlink()

    def test_html_contains_vis_network(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "Alpha",
                    "domain_tag": "ml",
                    "content_tags": ["x", "y"],
                },
                {
                    "zotero_key": "B",
                    "title": "Beta",
                    "domain_tag": "ml",
                    "content_tags": ["x", "z"],
                },
            ],
        )
        G, _ = build_graph(conn)
        path = render_pyvis(G)
        html = path.read_text()
        assert "vis" in html.lower()
        path.unlink()

    def test_show_labels_false_uses_space(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "Alpha Paper",
                    "domain_tag": "ml",
                    "content_tags": ["transformer"],
                },
            ],
        )
        G, _ = build_graph(conn)
        path = render_pyvis(G, show_labels=False)
        html = path.read_text()
        # The label should be " " (space), not the full title
        assert '"label": "Alpha Paper"' not in html
        path.unlink()

    def test_show_labels_true_includes_title(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "Alpha Paper",
                    "domain_tag": "ml",
                    "content_tags": ["transformer"],
                },
            ],
        )
        G, _ = build_graph(conn)
        path = render_pyvis(G, show_labels=True)
        html = path.read_text()
        assert "Alpha" in html
        path.unlink()

    def test_output_path_respected(self, conn):
        _seed(
            conn,
            [
                {
                    "zotero_key": "A",
                    "title": "A",
                    "domain_tag": "ml",
                    "content_tags": [],
                },
            ],
        )
        G, _ = build_graph(conn)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            target = Path(f.name)
        path = render_pyvis(G, output_path=target)
        assert path == target
        assert target.exists()
        target.unlink()
