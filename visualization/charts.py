"""Reusable Plotly charts. Data preparation is pure pandas (testable without Plotly).

Every builder cleans NaN and +/-inf before plotting, caps categories at Top-N plus "Other", and returns
None when nothing plottable is left, so the UI can show a message instead of an empty chart.
Plotly is imported inside each builder so this module can be imported (and its helpers tested) without it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.descriptive import OTHER_LABEL, numeric_series, top_n_with_other

PALETTE = ["#3DA58A", "#5B8DEF", "#E0A458", "#C26D8B", "#8E7CC3", "#6BB7C9", "#A3B35C", "#D0745F", "#7F8C9B", "#B8A27A"]
TEXT, MUTED, GRID = "#E6E8EB", "#8B93A1", "#232830"
MAX_DONUT_SLICES = 8
MAX_BOX_GROUPS = 10
MAX_HEATMAP_COLUMNS = 20


# ---- data preparation (no Plotly) -------------------------------------------------------------
def clean_xy(df: pd.DataFrame, x: str, y: str, x_numeric: bool = False) -> pd.DataFrame:
    """Rows where both axes are usable. y (and x if numeric) must be finite; x may be a date or label."""
    frame = pd.DataFrame({"x": numeric_series(df[x]) if x_numeric else df[x], "y": numeric_series(df[y])})
    return frame.dropna(subset=["x", "y"]).reset_index(drop=True)


def bin_categories(labels: pd.Series, values: pd.Series | None = None, n: int = 10) -> pd.DataFrame:
    """Top-n categories plus one 'Other' row. Columns: label, value. Counts rows when values is None."""
    lab = labels.astype("string")
    if values is None:
        agg = lab.dropna().value_counts().astype("float64")
    else:
        agg = pd.DataFrame({"l": lab, "v": numeric_series(values)}).dropna().groupby("l")["v"].sum()
    shown = top_n_with_other(agg, n)
    return pd.DataFrame({"label": shown.index.astype(str), "value": shown.to_numpy(dtype="float64")})


def sample_points(frame: pd.DataFrame, max_points: int, seed: int = 0) -> tuple[pd.DataFrame, bool]:
    if len(frame) <= max_points:
        return frame, False
    return frame.sample(max_points, random_state=seed), True


# ---- styling ----------------------------------------------------------------------------------
def _style(fig, title: str, x_title: str = "", y_title: str = "", height: int = 360):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=TEXT)),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=12), colorway=PALETTE, height=height,
        margin=dict(l=10, r=10, t=48, b=10), legend=dict(orientation="h", y=-0.2, font=dict(color=MUTED)),
    )
    fig.update_xaxes(title_text=x_title, gridcolor=GRID, zeroline=False, automargin=True)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def line_chart(df: pd.DataFrame, x: str, y: str, title: str, y_title: str = "", x_title: str = "",
               partial: pd.Series | None = None):
    """Line over time. `partial` is a boolean Series aligned with df marking incomplete periods."""
    import plotly.graph_objects as go
    f = df[[x, y]].copy()
    f[y] = numeric_series(f[y])
    f = f.dropna(subset=[x])
    if f[y].notna().sum() == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f[x], y=f[y], mode="lines+markers", name=y_title or y,
                             line=dict(color=PALETTE[0], width=2), marker=dict(size=5), connectgaps=False))
    if partial is not None:
        p = partial.reindex(f.index).fillna(False).astype(bool)
        if p.any():
            fig.add_trace(go.Scatter(x=f.loc[p, x], y=f.loc[p, y], mode="markers", name="Partial period",
                                     marker=dict(symbol="x", size=11, color=PALETTE[2])))
    return _style(fig, title, x_title, y_title or y)


def bar_chart(table: pd.DataFrame, label: str, value: str, title: str, y_title: str = "", horizontal: bool = True):
    """Bars from an already-aggregated table (use bin_categories first for raw categories)."""
    import plotly.graph_objects as go
    t = table[[label, value]].copy()
    t[value] = numeric_series(t[value])
    t = t.dropna(subset=[value])
    if t.empty:
        return None
    if len(t) > 15:                              # defensive cap: callers should bin first
        b = bin_categories(t[label], t[value], 10)
        t = b.rename(columns={"label": label, "value": value})
    colors = [MUTED if str(l) == OTHER_LABEL else PALETTE[0] for l in t[label]]
    if horizontal:
        fig = go.Figure(go.Bar(x=t[value], y=t[label].astype(str), orientation="h", marker_color=colors))
        fig.update_yaxes(autorange="reversed")
        _style(fig, title, y_title or value, "", height=max(260, 30 * len(t) + 110))
    else:
        fig = go.Figure(go.Bar(x=t[label].astype(str), y=t[value], marker_color=colors))
        _style(fig, title, "", y_title or value)
    return fig


def share_chart(table: pd.DataFrame, label: str, value: str, title: str):
    """Donut for a handful of non-negative parts; falls back to bars when there are too many (no 500-slice pies)."""
    import plotly.graph_objects as go
    t = table[[label, value]].copy()
    t[value] = numeric_series(t[value])
    t = t.dropna(subset=[value])
    if t.empty:
        return None
    if len(t) > MAX_DONUT_SLICES or (t[value] < 0).any() or t[value].sum() <= 0:
        return bar_chart(t, label, value, title)
    fig = go.Figure(go.Pie(labels=t[label].astype(str), values=t[value], hole=0.55, sort=False,
                           marker=dict(colors=PALETTE[:len(t)])))
    return _style(fig, title)


def histogram(values: pd.Series, title: str, x_title: str = "", bins: int = 30):
    import plotly.graph_objects as go
    v = numeric_series(values).dropna()
    if v.empty:
        return None
    fig = go.Figure(go.Histogram(x=v, nbinsx=bins, marker_color=PALETTE[0]))
    return _style(fig, title, x_title or str(values.name or ""), "Rows")


def box_plot(df: pd.DataFrame, y: str, title: str, by: str | None = None, y_title: str = ""):
    import plotly.graph_objects as go
    v = numeric_series(df[y])
    fig = go.Figure()
    if by is None:
        v = v.dropna()
        if v.empty:
            return None
        fig.add_trace(go.Box(y=v, name=y, marker_color=PALETTE[0], boxpoints="outliers"))
    else:
        f = pd.DataFrame({"g": df[by].astype("string"), "v": v}).dropna()
        if f.empty:
            return None
        top = f["g"].value_counts().head(MAX_BOX_GROUPS).index
        for i, g in enumerate(top):
            fig.add_trace(go.Box(y=f.loc[f["g"] == g, "v"], name=str(g), marker_color=PALETTE[i % len(PALETTE)],
                                 boxpoints="outliers"))
    return _style(fig, title, "", y_title or y)


def scatter(df: pd.DataFrame, x: str, y: str, title: str, max_points: int = 5000, color: str | None = None):
    """Scatter of two numeric columns. Large data is sampled (fixed seed); the title says so."""
    import plotly.graph_objects as go
    f = clean_xy(df, x, y, x_numeric=True)
    if f.empty:
        return None
    f, sampled = sample_points(f, max_points)
    fig = go.Figure(go.Scattergl(x=f["x"], y=f["y"], mode="markers", marker=dict(size=5, color=PALETTE[0], opacity=0.6)))
    note = f" (random sample of {len(f):,} points)" if sampled else ""
    return _style(fig, title + note, x, y)


def heatmap(matrix: pd.DataFrame, title: str, max_columns: int = MAX_HEATMAP_COLUMNS):
    """Correlation-style heatmap fixed to the range -1..1. Wide matrices are cut to max_columns."""
    import plotly.graph_objects as go
    if matrix is None or matrix.empty:
        return None
    m = matrix.iloc[:max_columns, :max_columns].replace([np.inf, -np.inf], np.nan)
    fig = go.Figure(go.Heatmap(z=m.to_numpy(), x=list(m.columns), y=list(m.index), zmin=-1, zmax=1,
                               colorscale=[[0, "#C26D8B"], [0.5, "#101318"], [1, "#3DA58A"]],
                               text=np.round(m.to_numpy(dtype=float), 2), texttemplate="%{text}",
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    _style(fig, title, height=max(320, 34 * len(m) + 140))
    return fig
