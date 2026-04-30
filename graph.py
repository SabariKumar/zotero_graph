"""
graph.py — NetworkX graph construction and PyVis rendering.

build_graph()   — assembles the weighted paper graph from the SQLite cache.
graph_stats()   — summary statistics for display in the UI.
render_pyvis()  — writes an interactive HTML graph via PyVis.
"""

import math
import sqlite3
import tempfile
from pathlib import Path

import community as community_louvain
import matplotlib
import matplotlib.colors as mcolors
import networkx as nx
from pyvis.network import Network

from cache import get_all_citation_edges, get_all_papers
from config import ALPHA, BETA, CITATION_BONUS, MIN_EDGE_WEIGHT, MIN_JACCARD

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    conn: sqlite3.Connection,
    *,
    domain_filter: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    tag_query: str | None = None,
    min_edge_weight: float = MIN_EDGE_WEIGHT,
) -> tuple[nx.Graph, dict[str, int]]:
    """Build a weighted NetworkX graph from the cached papers.

    Nodes  — one per paper passing the active filters.
    Edges  — tag-Jaccard (semantic) + citation bonus (structural).

    Returns (G, community_map) where community_map maps node id → int.
    """
    papers = get_all_papers(conn)
    papers = _apply_filters(papers, domain_filter, year_min, year_max, tag_query)

    G = nx.Graph()

    # --- Add nodes ---
    for p in papers:
        G.add_node(
            p["zotero_key"],
            label=p["title"][:60],
            title=p["title"],
            zotero_key=p["zotero_key"],
            openalex_id=p.get("openalex_id") or "",
            domain=p.get("domain_tag") or "untagged",
            tags=p.get("content_tags") or [],
            year=p.get("year"),
            cited_by_count=p.get("cited_by_count") or 0,
            community=0,  # filled in after Louvain
        )

    # --- Tag-Jaccard edges ---
    keys = list(G.nodes)
    tag_sets = {
        k: frozenset(G.nodes[k]["tags"]) for k in keys
    }

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            jaccard = _jaccard(tag_sets[a], tag_sets[b])
            if jaccard >= MIN_JACCARD:
                weight = ALPHA * jaccard
                G.add_edge(a, b, tag_weight=jaccard, citation=False, weight=weight)

    # --- Citation bonus ---
    node_set = set(G.nodes)
    for source_key, target_key in get_all_citation_edges(conn):
        if source_key not in node_set or target_key not in node_set:
            continue
        citation_w = BETA * CITATION_BONUS
        if G.has_edge(source_key, target_key):
            G[source_key][target_key]["weight"] += citation_w
            G[source_key][target_key]["citation"] = True
        else:
            G.add_edge(
                source_key, target_key,
                tag_weight=0.0, citation=True,
                weight=citation_w,
            )

    # --- Drop weak edges ---
    weak = [
        (u, v) for u, v, d in G.edges(data=True)
        if d["weight"] < min_edge_weight
    ]
    G.remove_edges_from(weak)

    # --- Louvain community detection ---
    community_map: dict[str, int] = {}
    if G.number_of_nodes() > 0:
        community_map = community_louvain.best_partition(G, weight="weight")
        nx.set_node_attributes(G, community_map, "community")

    return G, community_map


def graph_stats(G: nx.Graph) -> dict:
    """Return summary statistics for the sidebar."""
    all_tags: list[str] = []
    for _, data in G.nodes(data=True):
        all_tags.extend(data.get("tags") or [])

    tag_counts: dict[str, int] = {}
    for t in all_tags:
        tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    communities = {data["community"] for _, data in G.nodes(data=True)}

    return {
        "n_papers": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_communities": len(communities),
        "top_tags": top_tags,
        "citation_edges": sum(
            1 for _, _, d in G.edges(data=True) if d.get("citation")
        ),
    }


# ---------------------------------------------------------------------------
# PyVis rendering
# ---------------------------------------------------------------------------

_PALETTE = [
    mcolors.to_hex(c)
    for c in matplotlib.colormaps["tab20"].colors  # type: ignore[attr-defined]
]


def render_pyvis(
    G: nx.Graph,
    output_path: Path | None = None,
) -> Path:
    """Render the graph to an interactive HTML file via PyVis.

    If output_path is None a temporary file is created. Returns the path.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, prefix="zotero_graph_"
        )
        output_path = Path(tmp.name)
        tmp.close()

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#1a1a1a",
        font_color="white",
        directed=False,
    )
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=120,
        spring_strength=0.08,
        damping=0.4,
    )

    # --- Nodes ---
    community_ids = sorted({d["community"] for _, d in G.nodes(data=True)})
    color_map = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(community_ids)}

    for node_id, data in G.nodes(data=True):
        size = _node_size(data.get("cited_by_count") or 0)
        color = color_map.get(data.get("community", 0), _PALETTE[0])
        tooltip = _node_tooltip(data)

        net.add_node(
            node_id,
            label=data.get("label") or node_id,
            title=tooltip,
            color=color,
            size=size,
        )

    # --- Edges ---
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 0.0)
        citation = data.get("citation", False)
        # Citation edges get a slightly warmer tint
        if citation:
            color = f"rgba(255,180,80,{min(weight, 1.0):.2f})"
        else:
            color = f"rgba(200,200,200,{min(weight, 1.0):.2f})"
        net.add_edge(u, v, color=color, width=max(1, weight * 4), title=f"weight={weight:.2f}")

    net.save_graph(str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _node_size(cited_by_count: int) -> float:
    return 10 + 5 * math.log1p(cited_by_count)


def _node_tooltip(data: dict) -> str:
    tags = (data.get("tags") or [])[:5]
    tag_str = ", ".join(tags) if tags else "—"
    year = data.get("year") or "?"
    cites = data.get("cited_by_count") or 0
    domain = data.get("domain") or "untagged"
    title = data.get("title") or data.get("label") or ""
    return (
        f"<b>{title}</b><br>"
        f"<i>{domain}</i> · {year} · {cites:,} citations<br>"
        f"<b>Tags:</b> {tag_str}"
    )


def _apply_filters(
    papers: list[dict],
    domain_filter: list[str] | None,
    year_min: int | None,
    year_max: int | None,
    tag_query: str | None,
) -> list[dict]:
    out = papers
    if domain_filter:
        out = [p for p in out if p.get("domain_tag") in domain_filter]
    if year_min is not None:
        out = [p for p in out if (p.get("year") or 0) >= year_min]
    if year_max is not None:
        out = [p for p in out if (p.get("year") or 9999) <= year_max]
    if tag_query:
        q = tag_query.lower()
        out = [
            p for p in out
            if any(q in t for t in (p.get("content_tags") or []))
            or q in (p.get("domain_tag") or "").lower()
        ]
    return out
