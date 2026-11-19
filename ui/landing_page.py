"""Public homepage. Static content only: it reads no user data and needs no session.

Every capability described here exists in the platform (see README "Features"); nothing is invented, and the
product preview is labelled as using synthetic figures.
"""
from __future__ import annotations

import base64

import streamlit as st

from ui.public_ui import apply_public_theme, brand, footer, icon

FEATURES = [
    ("analysis", "Intelligent data analysis",
     "Profiles every column, detects types and works out what each one means for your business: revenue, dates, suppliers, quantities."),
    ("clean", "Automated data cleaning",
     "Proposes fixes for missing values, duplicates, types and outliers. You approve each step; the original file is never changed."),
    ("kpi", "Business KPI monitoring",
     "Computes only the KPIs your data supports, each shown with its formula, source columns and limitations."),
    ("forecast", "Predictive forecasting",
     "Validates seasonal, smoothing, Holt-Winters and ARIMA models on held-out periods and keeps the simplest one that performs."),
    ("anomaly", "Anomaly detection",
     "Flags unusual records and periods with robust statistics and Isolation Forest. Records are flagged, never deleted."),
    ("truck", "Supplier performance analysis",
     "On-time in full, fill rate and lead time by supplier, with segments that group suppliers by how they actually perform."),
    ("insight", "Automated business insights",
     "Rules turn computed results into insights, each carrying its evidence, threshold and a recommended next step."),
    ("report", "Interactive reporting",
     "Explore results interactively, then export a PDF or HTML report built only from figures computed from your data."),
]

SOLUTIONS = [
    ("network", "Supply chain analytics",
     "One view of orders, deliveries and lead times, with demand and volume forecasts validated on your own history.",
     ("On-time in full", "Lead time", "Demand forecast")),
    ("boxes", "Inventory optimization",
     "See stockout exposure and inventory cover, and screen stock movements for unusual periods before they become shortages.",
     ("Inventory cover", "Stockout rate", "Anomalies")),
    ("people", "Supplier management",
     "Compare suppliers on delivery reliability, fill rate and lead time, and group them into segments you can act on.",
     ("Fill rate", "Supplier segments", "Scorecards")),
    ("sliders", "Operational analytics",
     "Find which factors move your operational outcomes with driver analysis tested on held-out data.",
     ("Key drivers", "Data quality", "Trends")),
    ("pulse", "Business performance monitoring",
     "Track revenue, volume and other core measures over time for any business dataset, with breakdowns by segment.",
     ("KPIs", "Breakdowns", "Reports")),
]

STEPS = [
    ("upload", "Upload your business dataset",
     "CSV, TSV or Excel. Encodings, delimiters and worksheets are detected automatically, and the file is validated on upload."),
    ("cpu", "Let the platform analyse your data",
     "Profiling, quality checks, cleaning proposals, column roles, KPIs, forecasts and anomaly screening run on your data."),
    ("target", "Discover actionable insights",
     "Review evidence-backed insights and recommendations, then share a finished PDF or HTML report."),
]

PRINCIPLES = [
    ("eye", "Computed from your data",
     "Every figure comes from the dataset you upload. Analyses the data cannot support are listed with the reason, never filled in."),
    ("shield", "Explainable by design",
     "Formulas, validation methods and evidence accompany each KPI, forecast and insight, so results can be checked."),
    ("server", "Private by default",
     "Runs on your own infrastructure with no external AI service. Each account sees only its own datasets and reports."),
]


def _nav() -> str:
    return f"""
<nav class="lp-nav" aria-label="Main">
  {brand()}
  <div class="lp-links"><a href="#lp-home">Home</a><a href="#lp-features">Features</a><a href="#lp-solutions">Solutions</a><a href="#lp-about">About</a></div>
  <div class="lp-cta"><a class="pb-btn sm ghost" href="login" target="_self">Log in</a>
    <a class="pb-btn sm primary" href="register" target="_self">Get started</a></div>
</nav>"""


def _preview() -> str:
    hist = [(0, 88), (22.5, 80), (45, 84), (67.5, 72), (90, 76), (112.5, 64), (135, 68), (157.5, 58), (180, 62), (202.5, 50),
            (225, 54), (247.5, 44), (270, 40)]
    fc = [(270, 40), (302.5, 36), (335, 37), (367.5, 31), (400, 28)]
    upper = [(270, 40), (302.5, 27), (335, 25), (367.5, 16), (400, 11)]
    lower = [(400, 45), (367.5, 46), (335, 49), (302.5, 45), (270, 40)]
    pts = lambda seq: " ".join(f"{x},{y}" for x, y in seq)
    area = pts(hist) + " 270,118 0,118"
    grid = "".join(f'<line x1="0" x2="400" y1="{y}" y2="{y}" stroke="#1F242B" stroke-width="1"/>' for y in (18, 51, 84, 117))
    chart_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 120" preserveAspectRatio="none">
  {grid}
  <polygon points="{area}" fill="#3DA58A" fill-opacity=".07"/>
  <polygon points="{pts(upper + lower)}" fill="#4FBC9F" fill-opacity=".12"/>
  <line x1="270" x2="270" y1="6" y2="118" stroke="#2A3038" stroke-dasharray="3 3"/>
  <polyline points="{pts(hist)}" fill="none" stroke="#3DA58A" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="{pts(fc)}" fill="none" stroke="#4FBC9F" stroke-width="2" stroke-dasharray="5 4" stroke-linecap="round"/>
</svg>"""
    chart = (f'<img class="pv-chart" alt="Orders per week, actual and forecast" '                   # st.html strips inline SVG
             f'src="data:image/svg+xml;base64,{base64.b64encode(chart_svg.encode()).decode()}">')
    stats = [("On-time in full", "92.4%", "+1.8 pts", ""), ("Fill rate", "96.1%", "+0.6 pts", ""),
             ("Avg lead time", "6.8 days", "-0.4 days", ""), ("Stockout rate", "3.2%", "+0.5 pts", "neg")]
    stat_html = "".join(f'<div class="pv-stat"><div class="pv-k">{k}</div><div class="pv-v">{v}</div><div class="pv-d {c}">{d}</div></div>'
                        for k, v, d, c in stats)
    bars = [("Northwind", 9.4, "w"), ("Apex", 7.1, ""), ("Meridian", 6.2, ""), ("Harbor", 5.5, ""), ("Summit", 4.8, "")]
    bar_html = "".join(f'<div class="pv-b"><span>{n}</span><div class="pv-track"><i class="{c}" style="width:{v / 10 * 100:.0f}%"></i></div>'
                       f"<em>{v}</em></div>" for n, v, c in bars)
    nav = [("Start", ["Home", "Upload"]), ("Find insights", ["Dashboard", "Forecasting", "Anomalies", "Insights"]), ("Deliver", ["Reports"])]
    side = "".join(f'<div class="pv-sec">{sec}</div>' + "".join(
        f'<div class="pv-nav{" on" if item == "Dashboard" else ""}">{item}</div>' for item in items) for sec, items in nav)
    return f"""
<div class="pv-wrap">
  <div class="pv" aria-label="Product preview">
    <div class="pv-bar"><i></i><i></i><i></i><span>Dashboard</span></div>
    <div class="pv-body">
      <div class="pv-side">{side}</div>
      <div class="pv-main">
        <div class="pv-ctx"><span class="pv-dot"></span><b>supply_chain_orders.csv</b><span>12,480 rows · 14 columns · cleaned data</span></div>
        <div class="pv-stats">{stat_html}</div>
        <div class="pv-card"><div class="pv-ct">Orders per week <span>Holt-Winters · 8-week forecast</span></div>{chart}
          <div class="pv-leg"><span><i></i>Actual</span><span><i class="f"></i>Forecast, 80% interval</span></div></div>
        <div class="pv-row">
          <div class="pv-card"><div class="pv-ct">Lead time by supplier <span>days</span></div><div class="pv-bars">{bar_html}</div></div>
          <div class="pv-card"><div class="pv-ct">Insights</div><div class="pv-ins">
            <div class="pv-in"><b>Northwind lead time 38% above median</b>412 orders · threshold 25%</div>
            <div class="pv-in ok"><b>Fill rate stable across warehouses</b>Spread within 1.2 pts</div></div></div>
        </div>
      </div>
    </div>
  </div>
  <div class="pv-cap">Interface preview with synthetic sample figures</div>
</div>"""


def _hero() -> str:
    trust = [("check", "CSV and Excel"), ("lock", "Private accounts"), ("server", "No external AI service required")]
    trust_html = "".join(f"<span>{icon(i, 15)}{t}</span>" for i, t in trust)
    return f"""
<section class="lp-hero" id="lp-home">
  <div>
    <span class="lp-pill"><b>Analytics</b>Forecasting, anomalies and insights in one workspace</span>
    <h1 class="lp-h1">Turn Business Data Into <span>Decisions That Drive Growth.</span></h1>
    <p class="lp-lead">Transform raw business data into meaningful insights with intelligent analytics, predictive forecasting,
      automated reporting, and AI-powered business intelligence.</p>
    <div class="lp-actions"><a class="pb-btn primary" href="register" target="_self">Get Started {icon("arrow", 17)}</a>
      <a class="pb-btn" href="#lp-features">Explore Platform</a></div>
    <div class="lp-trust">{trust_html}</div>
  </div>
  {_preview()}
</section>"""


def _features() -> str:
    cards = "".join(f'<div class="lp-card"><div class="lp-ic">{icon(i)}</div><div class="lp-ct">{t}</div><div class="lp-cd">{d}</div></div>'
                    for i, t, d in FEATURES)
    return f"""
<section class="lp-sec" id="lp-features">
  <div class="lp-eyebrow">Features</div>
  <h2 class="lp-h2">Everything between a raw spreadsheet and a confident decision.</h2>
  <p class="lp-sub">One workflow takes your data from upload to report. Each step shows its working, so you can trust what you share.</p>
  <div class="lp-grid">{cards}</div>
</section>"""


def _solutions() -> str:
    rows = "".join(f'<div class="lp-sol"><div class="lp-ic">{icon(i)}</div><div><div class="lp-ct">{t}</div><div class="lp-cd">{d}</div>'
                   f'<div class="lp-tags">{"".join(f"<span>{tag}</span>" for tag in tags)}</div></div></div>'
                   for i, t, d, tags in SOLUTIONS)
    return f"""
<section class="lp-sec" id="lp-solutions">
  <div class="lp-split">
    <div><div class="lp-eyebrow">Industry solutions</div>
      <h2 class="lp-h2">Built for operations and supply chain teams.</h2>
      <p class="lp-sub">Supply chain is where the platform goes deepest, with purpose-built KPIs for delivery, inventory and suppliers.
        The same engine works on any business dataset.</p></div>
    <div class="lp-sols">{rows}</div>
  </div>
</section>"""


def _how() -> str:
    steps = "".join(f'<div class="lp-step"><div class="lp-step-top"><div class="lp-ic">{icon(i)}</div><span class="lp-num">0{n}</span></div>'
                    f'<div class="lp-ct">{t}</div><div class="lp-cd">{d}</div></div>'
                    for n, (i, t, d) in enumerate(STEPS, start=1))
    return f"""
<section class="lp-sec" id="lp-how">
  <div class="lp-eyebrow">How it works</div>
  <h2 class="lp-h2">From upload to insight in three steps.</h2>
  <div class="lp-steps">{steps}</div>
</section>"""


def _about() -> str:
    rows = "".join(f'<div class="lp-pr"><div class="lp-ic">{icon(i)}</div><div><div class="lp-ct">{t}</div><div class="lp-cd">{d}</div></div></div>'
                   for i, t, d in PRINCIPLES)
    return f"""
<section class="lp-sec" id="lp-about">
  <div class="lp-split">
    <div><div class="lp-eyebrow">About</div>
      <h2 class="lp-h2">Business intelligence you can verify.</h2>
      <p class="lp-sub">The AI Business Intelligence Platform turns business datasets into analysis that holds up to scrutiny.
        It computes every figure from the data you upload, explains how each result was produced, and says plainly when the
        data cannot support an analysis.</p></div>
    <div class="lp-princ">{rows}</div>
  </div>
</section>
<section class="lp-band">
  <h2 class="lp-h2">Start turning your data into decisions.</h2>
  <p class="lp-sub">Create an account, upload a dataset and see your first insights in minutes.</p>
  <div class="lp-actions"><a class="pb-btn primary" href="register" target="_self">Get Started {icon("arrow", 17)}</a>
    <a class="pb-btn" href="login" target="_self">Log in</a></div>
</section>"""


def render_landing_page() -> None:
    apply_public_theme()
    st.html(_nav() + _hero())
    st.html(_features() + _solutions())
    st.html(_how() + _about())
    footer()
