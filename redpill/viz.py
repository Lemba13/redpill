"""
viz.py — Visualize the article embedding space as an interactive 3D scatter plot.

Reads embeddings from seen_items in a redpill.db, reduces to 2D with UMAP,
and writes a self-contained HTML file with a Plotly Scatter3d figure.
X/Y axes are the UMAP projection; Z axis is first_seen_date.
Points are colored by vote status (upvoted / downvoted / not voted).
"""

import hashlib
import logging
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from redpill.state import deserialize_embedding

logger = logging.getLogger(__name__)

VIZ_DIR = Path("/home/ching/projects/redpill/viz")

_COLORS = {
    "voted up":    "#4ade80",
    "voted down":  "#f87171",
    "not voted":   "#94a3b8",
}


def _item_id(url: str) -> str:
    """Stable 12-char hex ID derived from URL — mirrors generate_item_id in deliver.py."""
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _load_votes(feedback_db_path: str | None) -> dict[str, str]:
    """Return {item_id: vote} from feedback.db. Returns empty dict on any failure."""
    if not feedback_db_path:
        return {}
    fb_path = Path(feedback_db_path)
    if not fb_path.exists():
        logger.info("Feedback DB not found at %s — all items shown as not voted", feedback_db_path)
        return {}
    try:
        conn = sqlite3.connect(f"file:{fb_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT item_id, vote FROM votes").fetchall()
        conn.close()
        return {row["item_id"]: row["vote"] for row in rows}
    except Exception as exc:
        logger.warning("Could not load votes from %s: %s — continuing without", feedback_db_path, exc)
        return {}


def run_viz(
    db_path: str,
    feedback_db_path: str | None = None,
    output_dir: Path = VIZ_DIR,
) -> Path:
    """
    Load embeddings from db_path, run UMAP, write HTML to output_dir.
    Returns the path of the written file.
    """
    if not output_dir.exists():
        raise FileNotFoundError(
            f"Viz output directory not found: {output_dir}\n"
            "Create it manually: mkdir -p /home/ching/projects/redpill/viz"
        )

    try:
        import umap
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            f"Missing viz dependency: {exc.name}. "
            "Install with: pip install 'redpill[viz]'"
        ) from exc

    # ------------------------------------------------------------------
    # Load rows from seen_items
    # ------------------------------------------------------------------
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db_rows = conn.execute(
        """
        SELECT url, title, summary, first_seen_date, topic, embedding
        FROM seen_items
        WHERE embedding IS NOT NULL
        """
    ).fetchall()
    topic_row = conn.execute("SELECT topic FROM seen_items LIMIT 1").fetchone()
    conn.close()

    topic: str = topic_row["topic"] if topic_row else "unknown"
    print(f"Loading embeddings from: {db_path}")

    # ------------------------------------------------------------------
    # Deserialize embeddings
    # ------------------------------------------------------------------
    rows = []
    embeddings = []
    for r in db_rows:
        try:
            vec = deserialize_embedding(r["embedding"])
        except Exception as exc:
            logger.warning("Failed to deserialize embedding for %s: %s — skipping", r["url"], exc)
            continue
        rows.append(r)
        embeddings.append(vec)

    if len(rows) < 2:
        raise ValueError(
            f"Not enough items with valid embeddings to visualize (found {len(rows)}, need at least 2)."
        )

    # ------------------------------------------------------------------
    # Load votes and classify
    # ------------------------------------------------------------------
    votes = _load_votes(feedback_db_path)

    categories = []
    for r in rows:
        iid = _item_id(r["url"])
        vote = votes.get(iid)
        if vote == "up":
            categories.append("voted up")
        elif vote == "down":
            categories.append("voted down")
        else:
            categories.append("not voted")

    categories_arr = np.array(categories)
    n_up   = int((categories_arr == "voted up").sum())
    n_down = int((categories_arr == "voted down").sum())
    n_none = int((categories_arr == "not voted").sum())
    print(f"Found {len(rows)} items with embeddings (voted up: {n_up}, voted down: {n_down}, not voted: {n_none})")

    # ------------------------------------------------------------------
    # UMAP reduction
    # ------------------------------------------------------------------
    print("Running UMAP reduction...")
    embeddings_matrix = np.vstack(embeddings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(rows) - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings_matrix)

    # ------------------------------------------------------------------
    # Date axis — use ISO date strings directly (Plotly handles datetime on 3D axes)
    # ------------------------------------------------------------------
    dates = np.array([r["first_seen_date"] or "" for r in rows])

    # ------------------------------------------------------------------
    # Hover text
    # ------------------------------------------------------------------
    hover_text = []
    for r, cat in zip(rows, categories):
        summary = (r["summary"] or "")[:120]
        hover_text.append(
            f"<b>{r['title'] or '(no title)'}</b><br>"
            f"Date: {r['first_seen_date']}<br>"
            f"Vote: {cat}<br>"
            f"<i>{summary}...</i>"
        )
    hover_arr = np.array(hover_text)

    # ------------------------------------------------------------------
    # Plotly figure (3D)
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig = go.Figure()

    for cat_name, color in _COLORS.items():
        mask = categories_arr == cat_name
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0],
            y=coords[mask, 1],
            z=dates[mask],
            mode="markers",
            name=cat_name,
            marker=dict(color=color, size=4, opacity=0.8),
            text=hover_arr[mask],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"RedPill — {topic} — {timestamp}",
            font=dict(color="#e2e8f0", size=15),
        ),
        paper_bgcolor="#0d1117",
        font=dict(color="#e2e8f0", family="monospace"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.04)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(size=12),
        ),
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                title="", showspikes=False,
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                title="", showspikes=False,
            ),
            zaxis=dict(
                title="date",
                gridcolor="rgba(255,255,255,0.08)",
                showgrid=True,
                zeroline=False,
                tickfont=dict(color="#94a3b8", size=10),
                title_font=dict(color="#94a3b8", size=11),
                showspikes=False,
            ),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#334155",
            font=dict(color="#e2e8f0", size=12),
        ),
    )

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"embeddings-{timestamp_str}.html"
    print(f"Writing HTML to {output_path}")
    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        full_html=True,
    )
    print("Done. Open in browser.")
    return output_path
