"""Reusable Plotly charts. Data preparation is pure pandas (testable without Plotly).

Every builder cleans NaN and +/-inf before plotting, caps categories at Top-N plus "Other", and returns
None when nothing plottable is left, so the UI can show a message instead of an empty chart.
Plotly is imported inside each builder so this module can be imported (and its helpers tested) without it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.descriptive import OTHER_LABEL, numeric_series, top_n_with_other

PALETTE = ["#D9541E", "#2F6FDB", "#2F9E6B", "#8257D8", "#C9962B", "#5B6B7D", "#C2477A", "#2A9BB5", "#7C8B3A", "#9C6B4E"]
MUTED = "#8C887F"      # mid grey that reads on both light and dark backgrounds
# Light defaults; apply_theme() restyles a figure for the appearance it is shown in.
TEXT, GRID = "#1C1B19", "#E8E5DF"
_THEMES = {
    "light": {"template": "plotly_white", "text": "#1C1B19", "muted": "#625F58", "grid": "#ECE9E3", "mid": "#F4F2EE"},
    "dark": {"template": "plotly_dark", "text": "#EDEAE4", "muted": "#A8A39A", "grid": "#2D2B28", "mid": "#23211F"},
}
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
def apply_theme(fig, appearance: str):
    """Restyle a finished figure for light or dark: template, text, grid and the heatmap midpoint. Data colours stay."""
    t = _THEMES.get(appearance, _THEMES["light"])
    fig.update_layout(template=t["template"], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=t["text"]), title_font_color=t["text"], legend_font_color=t["muted"])
    fig.update_xaxes(gridcolor=t["grid"], linecolor=t["grid"], tickfont_color=t["muted"])
    fig.update_yaxes(gridcolor=t["grid"], linecolor=t["grid"], tickfont_color=t["muted"])
    for trace in fig.data:
        if getattr(trace, "type", "") == "heatmap" and trace.colorscale and len(trace.colorscale) == 3:
            lo, _, hi = trace.colorscale
            trace.colorscale = [lo, [0.5, t["mid"]], hi]
    return fig


def _style(fig, title: str, x_title: str = "", y_title: str = "", height: int = 360):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=TEXT)),
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
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


def multi_line_chart(df: pd.DataFrame, x: str, ys: tuple[str, ...] | list[str], title: str, y_title: str = ""):
    """Several numeric series over a shared x (for example stock vs reorder point)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for i, y in enumerate(ys):
        v = numeric_series(df[y])
        if v.notna().sum() == 0:
            continue
        fig.add_trace(go.Scatter(x=df[x], y=v, mode="lines", name=y.replace("_", " "),
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2), connectgaps=False))
    if not fig.data:
        return None
    return _style(fig, title, "", y_title)


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
                               colorscale=[[0, "#2F6FDB"], [0.5, "#F4F2EE"], [1, "#D9541E"]],
                               text=np.round(m.to_numpy(dtype=float), 2), texttemplate="%{text}",
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    _style(fig, title, height=max(320, 34 * len(m) + 140))
    return fig


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, title: str, y_title: str = "",
                   backtest: pd.DataFrame | None = None, interval_label: str = "Prediction interval"):
    """History, out-of-sample validation predictions and a forecast with its interval.

    history: period, value. forecast: period, forecast, lower, upper. backtest: period, predicted.
    """
    import plotly.graph_objects as go
    h = history.dropna(subset=["period"])
    f = forecast.dropna(subset=["period"])
    if h["value"].notna().sum() == 0 or f.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f["period"], y=f["upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=f["period"], y=f["lower"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(217,84,30,0.14)", name=interval_label, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=h["period"], y=h["value"], mode="lines", name="History", line=dict(color=PALETTE[1], width=2)))
    if backtest is not None and len(backtest):
        fig.add_trace(go.Scatter(x=backtest["period"], y=backtest["predicted"], mode="lines", name="Validation predictions",
                                 line=dict(color=PALETTE[2], width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=f["period"], y=f["forecast"], mode="lines+markers", name="Forecast",
                             line=dict(color=PALETTE[0], width=2.5), marker=dict(size=5)))
    return _style(fig, title, "", y_title, height=420)


def anomaly_series_chart(series: pd.DataFrame, title: str, y_title: str = ""):
    """Series with its baseline and the flagged periods marked. series: period, value, baseline, flagged."""
    import plotly.graph_objects as go
    s = series.dropna(subset=["period"])
    if s["value"].notna().sum() == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["period"], y=s["baseline"], mode="lines", name="Expected (baseline)",
                             line=dict(color=PALETTE[6], width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=s["period"], y=s["value"], mode="lines", name="Value", line=dict(color=PALETTE[1], width=2)))
    f = s[s["flagged"].astype(bool)]
    if len(f):
        fig.add_trace(go.Scatter(x=f["period"], y=f["value"], mode="markers", name="Flagged",
                                 marker=dict(color=PALETTE[3], size=10, symbol="circle-open", line=dict(width=2))))
    return _style(fig, title, "", y_title, height=400)


def segment_scatter(df: pd.DataFrame, x: str, y: str, cluster: str, title: str, x_title: str = "", y_title: str = ""):
    """One colour per cluster for two entity-level features. Returns None when nothing is plottable."""
    import plotly.graph_objects as go
    f = df[[x, y, cluster]].copy()
    f[x], f[y] = numeric_series(f[x]), numeric_series(f[y])
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    if f.empty:
        return None
    fig = go.Figure()
    for i, (c, g) in enumerate(f.groupby(cluster)):
        fig.add_trace(go.Scattergl(x=g[x], y=g[y], mode="markers", name=f"Cluster {c}",
                                   marker=dict(size=7, color=PALETTE[i % len(PALETTE)], opacity=0.8)))
    return _style(fig, title, x_title or x, y_title or y)
