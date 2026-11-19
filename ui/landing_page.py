"""Public homepage. Static content only: it reads no user data and needs no session.

Every capability described here exists in the product (see README "Features"); nothing is invented. The picture in
the hero is a real screenshot of the Dashboard page, taken with the built-in synthetic sample dataset, and says so.
"""
from __future__ import annotations

import streamlit as st

from config.settings import APP_NAME
from ui.public_ui import ASSETS, apply_public_theme, brand, footer, icon, theme_switch, themed_img

SCREENSHOT = "screen-dashboard"          # assets/screen-dashboard-light.webp and -dark.webp

FEATURES = [
    ("analysis", "Intelligent data analysis",
     "Profiles every column and works out what it means: revenue, dates, suppliers, quantities."),
    ("clean", "Automated data cleaning",
     "Proposes fixes for missing values, duplicates, types and outliers. You approve each one; the original file is kept."),
    ("kpi", "Business KPI monitoring",
     "Only the KPIs your data supports, each with its formula, source columns and limitations."),
    ("forecast", "Predictive forecasting",
     "Seasonal, smoothing, Holt-Winters and ARIMA models, tested on held-out periods. The simplest one that holds up wins."),
    ("anomaly", "Anomaly detection",
     "Unusual records and periods, found with robust statistics and Isolation Forest. Flagged, never deleted."),
    ("truck", "Supplier performance analysis",
     "On-time in full, fill rate and lead time by supplier, and segments based on how suppliers actually perform."),
    ("insight", "Automated business insights",
     "Each insight carries its evidence, the threshold it crossed and a suggested next step."),
    ("report", "Interactive reporting",
     "Explore the results, then export a PDF or HTML report built only from figures computed from your data."),
]

SOLUTIONS = [
    ("network", "Supply chain analytics", "Orders, deliveries and lead times in one view, with demand forecasts tested on your history.",
     "On-time in full · Lead time · Demand forecast"),
    ("boxes", "Inventory optimization", "Stockout exposure and inventory cover, plus unusual stock movements before they become shortages.",
     "Inventory cover · Stockout rate · Anomalies"),
    ("people", "Supplier management", "Compare suppliers on reliability, fill rate and lead time, and group them into segments.",
     "Fill rate · Supplier segments · Scorecards"),
    ("sliders", "Operational analytics", "Which factors move your outcomes, from driver analysis checked on held-out data.",
     "Key drivers · Data quality · Trends"),
    ("pulse", "Business performance monitoring", "Revenue, volume and other core measures over time, broken down by segment.",
     "KPIs · Breakdowns · Reports"),
]

STEPS = [
    ("Upload a dataset", "CSV, TSV or Excel. Encoding, delimiter and worksheet are detected for you."),
    ("Review what it found", "Data quality, column roles, KPIs, forecasts and anomalies, each with the working shown."),
    ("Share the result", "Export a PDF or HTML report built from the figures you reviewed."),
]


def _nav() -> str:
    return f"""
<nav class="lp-nav" aria-label="Main">
  {brand()}
  <div class="lp-links"><a href="#lp-home">Home</a><a href="#lp-features">Features</a><a href="#lp-solutions">Solutions</a><a href="#lp-about">About</a></div>
  <div class="lp-cta">{theme_switch()}<a class="pb-btn sm ghost" href="login" target="_self">Log in</a>
    <a class="pb-btn sm primary" href="register" target="_self">Get started</a></div>
</nav>"""


def _screenshot() -> str:
    if not all((ASSETS / f"{SCREENSHOT}-{t}.webp").exists() for t in ("light", "dark")):
        return ""
    img = themed_img(SCREENSHOT, "webp", "The Dashboard page: KPI tiles, a revenue trend and breakdowns", loading="lazy")
    return (f'<figure class="lp-shot" style="margin-bottom:0">{img}</figure>'
            f'<p class="lp-cap">The Dashboard, shown with the built-in synthetic sample dataset.</p>')


def _hero() -> str:
    return f"""
<section class="lp-hero" id="lp-home">
  <div class="lp-hero-row">
    <h1 class="lp-h1">Turn Business Data Into Decisions That Drive Growth.</h1>
    <div>
      <p class="lp-lead">Transform raw business data into meaningful insights with intelligent analytics, predictive forecasting,
        automated reporting, and AI-powered business intelligence.</p>
      <div class="lp-actions"><a class="pb-btn primary" href="register" target="_self">Get Started {icon("arrow", 17)}</a>
        <a class="pb-btn" href="#lp-features">Explore Platform</a></div>
    </div>
  </div>
  {_screenshot()}
</section>"""


def _features() -> str:
    items = "".join(f'<div class="lp-item">{icon(i, 18)}<div><div class="lp-ct">{t}</div><div class="lp-cd">{d}</div></div></div>'
                    for i, t, d in FEATURES)
    return f"""
<section class="lp-sec" id="lp-features">
  <div><h2 class="lp-h2">From spreadsheet to decision.</h2>
    <p class="lp-sub">Each step shows its working, so you can check a number before you share it.</p></div>
  <div class="lp-list">{items}</div>
</section>"""


def _solutions() -> str:
    items = "".join(f'<div class="lp-item">{icon(i, 18)}<div><div class="lp-ct">{t}</div><div class="lp-cd">{d}</div>'
                    f'<div class="lp-tags">{tags}</div></div></div>' for i, t, d, tags in SOLUTIONS)
    return f"""
<section class="lp-sec" id="lp-solutions">
  <div><h2 class="lp-h2">Built for operations and supply chain teams.</h2>
    <p class="lp-sub">Supply chain gets purpose-built KPIs for delivery, inventory and suppliers. Everything else works on any
      business dataset.</p></div>
  <div class="lp-list">{items}</div>
</section>"""


def _how() -> str:
    steps = "".join(f'<div><div class="lp-num">{n}</div><div class="lp-ct">{t}</div><div class="lp-cd">{d}</div></div>'
                    for n, (t, d) in enumerate(STEPS, start=1))
    return f"""
<section class="lp-sec" id="lp-how">
  <div><h2 class="lp-h2">How it works</h2></div>
  <div class="lp-steps">{steps}</div>
</section>"""


def _about() -> str:
    return f"""
<section class="lp-sec" id="lp-about">
  <div><h2 class="lp-h2">About {APP_NAME}</h2></div>
  <div class="lp-about">
    <p><b>Every figure comes from your data.</b> When a dataset cannot support an analysis, {APP_NAME} says which one and why,
      instead of filling the gap.</p>
    <p><b>Results can be checked.</b> KPIs show their formulas, forecasts their validation, insights their evidence.</p>
    <p><b>Private by default.</b> It runs on your own machine or server with no external AI service, and each account sees only
      its own datasets and reports.</p>
    <div class="lp-actions"><a class="pb-btn primary" href="register" target="_self">Create an account</a>
      <a class="pb-btn" href="login" target="_self">Log in</a></div>
  </div>
</section>"""


def render_landing_page() -> None:
    apply_public_theme()
    st.html(_nav() + _hero())
    st.html(_features() + _solutions())
    st.html(_how() + _about())
    footer()
