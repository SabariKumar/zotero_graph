"""
app.py — Streamlit entry point for the Zotero topic graph.

Run with:  pixi run start
"""

import sqlite3

import streamlit as st
from streamlit_javascript import st_javascript

from cache import get_all_papers, init_db
from config import DB_PATH, MIN_EDGE_WEIGHT
from graph import build_graph, graph_stats, render_pyvis

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Zotero Topic Graph",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helper functions  (defined before use)
# ---------------------------------------------------------------------------

_LS_KEY = "zotero_selected_node"  # localStorage key shared with the PyVis iframe


def _inject_click_handler(html: str) -> str:
    """Inject selectNode/deselectNode listeners into the PyVis HTML.

    On click, writes the node ID to localStorage (same-origin, readable by the
    st_javascript poller below) and also fires a postMessage for any future
    listeners.
    """
    script = f"""
<script>
network.on("selectNode", function(params) {{
    if (params.nodes.length > 0) {{
        var nodeId = String(params.nodes[0]);
        localStorage.setItem("{_LS_KEY}", nodeId);
        window.parent.postMessage(
            {{type: "zotero_node_selected", nodeId: nodeId}}, "*"
        );
    }}
}});
network.on("deselectNode", function() {{
    localStorage.removeItem("{_LS_KEY}");
    window.parent.postMessage(
        {{type: "zotero_node_selected", nodeId: null}}, "*"
    );
}});
</script>
"""
    return html.replace("</body>", script + "\n</body>")


def _render_paper_card(p: dict) -> None:
    """Render a single paper's detail card."""
    st.markdown(f"**{p['title']}**")

    year = p.get("year") or "?"
    cites = p.get("cited_by_count")
    cites_str = f"{cites:,}" if cites else "—"
    st.caption(f"{year} · {cites_str} citations")

    domain = p.get("domain_tag")
    if domain:
        st.markdown(f"🏷️ `{domain}`")

    tags = p.get("content_tags") or []
    if tags:
        st.markdown(" ".join(f"`{t}`" for t in tags))

    st.divider()

    zotero_key = p["zotero_key"]
    oa_id = p.get("openalex_id")
    st.markdown(
        f"[Open in Zotero ↗](zotero://select/library/items/{zotero_key})",
        unsafe_allow_html=True,
    )
    if oa_id:
        st.markdown(
            f"[View on OpenAlex ↗](https://openalex.org/works/{oa_id})",
            unsafe_allow_html=True,
        )


def _render_info_panel(G, paper_map: dict[str, dict]) -> None:
    """Info panel driven by both graph node clicks and a dropdown fallback.

    Node clicks update st.session_state.selected_key (via localStorage poller),
    which pre-selects the matching entry in the dropdown and renders the card.
    The dropdown also works standalone when no node has been clicked.
    """
    visible_keys = [k for k in G.nodes if k in paper_map]
    if not visible_keys:
        st.caption("No papers in current view.")
        return

    visible_keys.sort(key=lambda k: paper_map[k]["title"])
    title_to_key = {paper_map[k]["title"][:70]: k for k in visible_keys}
    options = ["— select a paper —"] + list(title_to_key.keys())

    # If a node was clicked, pre-select it in the dropdown
    default_idx = 0
    clicked = st.session_state.get("selected_key")
    if clicked and clicked in paper_map:
        clicked_title = paper_map[clicked]["title"][:70]
        if clicked_title in options:
            default_idx = options.index(clicked_title)

    chosen_title = st.selectbox(
        "Select paper",
        options=options,
        index=default_idx,
        label_visibility="collapsed",
        key="paper_selectbox",
    )

    if chosen_title == "— select a paper —":
        st.caption("Click a node in the graph or select from the dropdown.")
        return

    key = title_to_key.get(chosen_title)
    if key and key in paper_map:
        # Sync session state with manual dropdown selection too
        if key != st.session_state.get("selected_key"):
            st.session_state.selected_key = key
        _render_paper_card(paper_map[key])


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "selected_key" not in st.session_state:
    st.session_state.selected_key = None

# ---------------------------------------------------------------------------
# DB connection (cached across reruns)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_conn() -> sqlite3.Connection:
    return init_db(DB_PATH)


conn = get_conn()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 Zotero Graph")
    st.caption("Tag-based + citation graph of your library")

    # Sync controls
    st.divider()
    st.subheader("Library sync")

    col_z, col_oa = st.columns(2)
    sync_zotero_btn = col_z.button("Sync Zotero", use_container_width=True)
    sync_all_btn = col_oa.button("Sync All", use_container_width=True, type="primary")

    if sync_zotero_btn or sync_all_btn:
        from fetcher import sync_openalex, sync_zotero

        with st.spinner("Syncing Zotero…"):
            n = sync_zotero(conn, verbose=False)
        st.success(f"Zotero: {n} items synced")
        if sync_all_btn:
            with st.spinner("Enriching via OpenAlex…"):
                m = sync_openalex(conn, verbose=False)
            st.success(f"OpenAlex: {m} papers enriched")
        st.rerun()

    # Load papers for filter widgets
    papers = get_all_papers(conn)

    if not papers:
        st.info("No papers in cache. Click **Sync Zotero** to load your library.")
        st.stop()

    # Build filter option ranges from the current cache
    all_domains = sorted({p["domain_tag"] for p in papers if p.get("domain_tag")})
    all_years = [p["year"] for p in papers if p.get("year")]
    year_lo = min(all_years) if all_years else 2000
    year_hi = max(all_years) if all_years else 2024

    # Filters
    st.divider()
    st.subheader("Filters")

    domain_filter = st.multiselect(
        "Domain",
        options=all_domains,
        default=[],
        placeholder="All domains",
    )

    year_range = st.slider(
        "Year range",
        min_value=year_lo,
        max_value=year_hi,
        value=(year_lo, year_hi),
    )

    tag_query = st.text_input("Tag search", placeholder="e.g. diffusion-model")

    min_weight = st.slider(
        "Min edge weight",
        min_value=0.0,
        max_value=1.0,
        value=float(MIN_EDGE_WEIGHT),
        step=0.05,
    )

    st.divider()
    st.subheader("Display")
    show_labels = st.toggle("Show node labels", value=False)

# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

G, community_map = build_graph(
    conn,
    domain_filter=domain_filter or None,
    year_min=year_range[0],
    year_max=year_range[1],
    tag_query=tag_query or None,
    min_edge_weight=min_weight,
)

stats = graph_stats(G)
paper_map = {p["zotero_key"]: p for p in papers}

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

graph_col, info_col = st.columns([3, 1])

with graph_col:
    # Stats strip
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Papers", stats["n_papers"])
    m2.metric("Edges", stats["n_edges"])
    m3.metric("Communities", stats["n_communities"])
    m4.metric("Citation edges", stats["citation_edges"])

    # Graph
    if G.number_of_nodes() == 0:
        st.warning("No papers match the current filters.")
    else:
        html_path = render_pyvis(G, show_labels=show_labels)
        html_content = _inject_click_handler(html_path.read_text())
        html_path.unlink(missing_ok=True)  # clean up temp file
        st.components.v1.html(html_content, height=760, scrolling=False)

        # --- localStorage poller ---
        # Reads the node ID written by the PyVis click handler. Both the PyVis
        # iframe and this st_javascript call share the same origin (localhost),
        # so localStorage is visible to both.
        clicked_key = st_javascript(
            f"localStorage.getItem('{_LS_KEY}') || ''",
        )
        if clicked_key and clicked_key != st.session_state.get("selected_key"):
            st.session_state.selected_key = clicked_key
            st.rerun()
        elif not clicked_key and st.session_state.get("selected_key"):
            # Node was deselected — clear without rerunning (avoids loop)
            st.session_state.selected_key = None

with info_col:
    st.subheader("Paper detail")
    _render_info_panel(G, paper_map)

# ---------------------------------------------------------------------------
# Top tags (below graph)
# ---------------------------------------------------------------------------

if stats["top_tags"]:
    st.divider()
    st.subheader("Top tags in current view")
    tag_cols = st.columns(5)
    for i, (tag, count) in enumerate(stats["top_tags"][:10]):
        tag_cols[i % 5].metric(tag, count)
