"""Drivers page: which columns go with a chosen result (late deliveries, delay, profit, units sold). Association, not cause.

Runs as soon as the page opens (supply-chain-first default target, all columns) and on every setting change; results are
cached per setting and filter and shared with the Home card. The page leads with the answer and how far to trust it,
then shows the top drivers, how the target changes across each one, and every column that was left out and why.
Nothing in the data is changed.
"""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from analytics.drivers import DATE_PARTS, NOTE, available_date_parts, candidate_features, default_target, driver_targets, \
    run_driver_analysis
from core.logging_config import get_logger
from semantic.column_lexicon import DOMAIN_LABELS
from ui import state
from ui.analysis_runner import safe_cached
from ui.analytics_state import base_key, show_figure
from ui.filters import filtered, no_rows_notice, render_filter_bar
from ui.layout import badge, callout, empty_state, lede, meta_line, no_dataset_notice, page_title, section, stat_cards
from ui.progress_bar import ProgressBar
from visualization import charts

log = get_logger("ui.drivers")
MAX_CARDS = 4
MIN_CARD_SHARE = 0.01            # drivers below 1% of the explanatory power get no card
# reasons that only describe the column type; anything else (copies, forecasts, after-the-fact values) is listed first
_ROUTINE = ("identifier", "same value in every row", "distinct values", "text column", "not selected", "missing in")


def _explain(df, profile, model, p):
    """p = (base_key, target, features). Shared with the Home card (same cache entry)."""
    with ProgressBar("Driver analysis") as bar:
        return run_driver_analysis(df, p[1], profile, model, list(p[2]), progress=bar)


def default_params(df, profile, model, target: str | None = None) -> tuple | None:
    """(base_key, target, features) for the default run: the page's first view and the Home card use the same key."""
    targets = driver_targets(df, profile, model)
    target = target if target in targets else default_target(df, profile, model, targets=targets)
    if target is None:
        return None
    num, cat, _ = candidate_features(df, profile, model, target)
    return base_key(), target, tuple(num + cat + available_date_parts(df, model, target))


def render_drivers_page() -> None:
    result = state.get_result()
    if result is None:
        page_title("Drivers", "Which columns go with a result.")
        no_dataset_notice()
        return
    ds = result.dataset
    page_title("Drivers", "Which columns go with a result, such as late deliveries or profit")
    try:
        df_all, profile, model = state.get_analysis_df(), state.get_analysis_profile(), state.get_semantic_model()
    except Exception:
        log.exception("Driver inputs unavailable")
        st.error("The dataset could not be prepared for analysis. Check the Data quality and Column roles pages.")
        return
    if df_all is None or profile is None or model is None or len(df_all) == 0:
        empty_state("Nothing to analyse", "The active dataset has no rows.")
        return
    df = filtered(df_all)
    dom = model.domain.domain
    meta_line([ds.filename + (" (synthetic sample)" if ds.is_sample else ""),
               f"{len(df):,} of {len(df_all):,} rows" if state.get_filter().active else f"{len(df):,} rows"],
              badge(DOMAIN_LABELS.get(dom, dom), "accent" if model.is_supply_chain else "neutral"))
    render_filter_bar(df_all, model)
    df = filtered(df_all)
    if len(df) == 0:
        no_rows_notice()
        return
    targets = driver_targets(df, profile, model)
    if not targets:
        empty_state("No suitable result to explain",
                    "A result needs a numeric measure with enough different values, or a Yes/No column where both answers "
                    "are common.")
        return
    first = default_target(df, profile, model, targets=targets)
    options = [first] + [t for t in targets if t != first]

    with st.container(border=True, key="dr_setup"):
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        target = c1.selectbox("Result to explain", options, key="dr_target",
                              format_func=lambda c: f"{c}  ·  {'yes / no' if targets[c] == 'classification' else 'number'}",
                              help="Supply-chain results (late deliveries, delay) come first, then profit, revenue and quantity.")
        num, cat, _ = candidate_features(df, profile, model, target)
        allf = num + cat + available_date_parts(df, model, target)
        with c2.popover("Columns", icon=":material/tune:", width="stretch"):
            feats = st.multiselect("Columns to compare", allf, default=allf, key=f"dr_feats_{target}")
            st.caption("Columns built from the result (copies, parts of it, a forecast of it, values only known afterwards) "
                       "are left out automatically and listed with the reason. " + NOTE)
    if len(feats) < 2:
        st.caption("Choose at least two columns to compare.")
        return
    p = (base_key(), target, tuple(feats))
    st.session_state["dr_params"] = p
    status, res = safe_cached("driver_result", p[1:], lambda: _explain(df, profile, model, p),
                              "Driver analysis could not be completed for this selection.")
    if status != "ok":
        empty_state("Analysis not possible", res) if status == "insufficient" else st.warning(res)
        return
    _show(res, model)


# ---- formatting ---------------------------------------------------------------------------------
def _unit(res, model) -> str:
    if res.task == "classification":
        return "%"
    role = model.role_of(res.target) if model is not None and res.target in model.columns else None
    return "days" if role == "duration" else ""


def _val(v: float, unit: str) -> str:
    """A target value: a share of 'yes' for a yes/no result, otherwise the number in its unit."""
    if v != v:
        return "–"
    return charts.value_text(100 * v if unit == "%" else v, unit)


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _subject(res) -> str:
    return f"{res.target} (how often it is yes)" if res.task == "classification" else res.target


_FLAG_VALUES = {"0", "1", "true", "false", "yes", "no", "y", "n"}


def _describe(res, feature: str, unit: str) -> str:
    """One plain sentence on how the target moves with this column."""
    g = res.groups.get(feature)
    if g is not None and len(g):
        hi, lo = g.loc[g["target_mean"].idxmax()], g.loc[g["target_mean"].idxmin()]
        banded = feature not in DATE_PARTS.values() and res.importance.set_index("feature").loc[feature, "kind"] == "numeric"
        flag = set(g["group"].astype(str).str.lower()) <= _FLAG_VALUES
        where = (lambda r: f"{feature} {r['group']}") if banded else (lambda r: f"{feature} = {r['group']}") if flag             else (lambda r: str(r["group"]))
        return (f"Highest for {where(hi)} ({_val(hi['target_mean'], unit)}), lowest for {where(lo)} "
                f"({_val(lo['target_mean'], unit)}); {_val(res.overall, unit)} overall.")
    d = str(res.importance.set_index("feature").loc[feature, "direction"])
    if d.startswith("higher values go with"):
        return f"Higher {feature} goes with {'higher' if 'higher target' in d else 'lower'} {res.target}."
    return "No steady up or down pattern; it matters in combination with other columns."


# ---- result -------------------------------------------------------------------------------------
def _show(res, model) -> None:
    unit = _unit(res, model)
    imp = res.importance
    pos = imp[imp["share"] > 0]
    m = res.metrics.set_index("metric")
    held = "later rows held out for testing" if res.split.startswith("chronological") else "rows held out for testing"
    score, base = float(res.metrics.loc[0, "model"]), float(res.metrics.loc[0, "baseline"])

    # ---- the answer
    if pos.empty:
        text = f"None of the columns helps predict {_subject(res)} on {held}."
    elif res.near_perfect:
        text = (f"{res.target} is predicted almost perfectly, mostly from {pos.iloc[0]['feature']}. "
                "That usually means a column is part of how the result is recorded, not a driver of it: check the top columns "
                "below before acting on them.")
    elif not res.reliable:
        text = (f"No column in this data clearly explains {_subject(res)}. On {held} the model does hardly better than "
                f"{'a plain average' if res.task == 'regression' else 'a coin toss'}, so the ranking below is shown for completeness only.")
    else:
        top = [f"{r.feature} ({r.share:.0%})" for r in pos.head(3).itertuples() if r.share >= MIN_CARD_SHARE]
        text = (f"{top[0].rsplit(' (', 1)[0]} is the column most closely tied to {res.target}, with "
                f"{pos.iloc[0]['share']:.0%} of the model's explanatory power")
        text += f", followed by {_and(top[1:])}." if len(top) > 1 else "."
        if res.task == "regression":
            text += f" On {held} the model explains {score:.0%} of the variation; a plain average explains none."
        else:
            acc, acc0 = float(res.metrics.loc[1, "model"]), float(res.metrics.loc[1, "baseline"])
            text += (f" Overall {res.target} is yes in {res.overall:.0%} of rows. On {held} the model is right {acc:.0%} of the time, against {acc0:.0%} for always guessing the most "
                     "common answer.")
    lede(text)

    # ---- numbers behind it
    tone = ("Too perfect", "bad") if res.near_perfect else ("Reliable", "good") if res.reliable else ("Weak", "bad")
    if res.task == "regression":
        mae, mae0 = float(m.iloc[1]["model"]), float(m.iloc[1]["baseline"])
        fit = {"label": "Variation explained (R²)", "value": f"{max(score, 0):.0%}" if score > -1 else "below 0%", "chip": tone,
               "sub": "a plain average: 0%", "foot": "Share of the ups and downs in held-out rows the model gets right.",
               "tip": f"R-squared {score:.3f}, plain average {base:.3f}"}
        gain = 1 - mae / mae0 if mae0 > 0 else 0.0
        err = {"label": "Typical error", "value": _val(mae, unit),
               "chip": (f"{abs(gain):.0%} {'smaller' if gain >= 0 else 'larger'}", "good" if gain > 0.05 else "flat"),
               "sub": f"than the average's {_val(mae0, unit)}", "foot": "Mean absolute error on held-out rows."}
    else:
        acc, acc0 = float(m.iloc[1]["model"]), float(m.iloc[1]["baseline"])
        fit = {"label": "Ranking quality (ROC AUC)", "value": f"{score:.2f}", "chip": tone, "sub": "coin toss: 0.50",
               "foot": "How often a 'yes' row scores above a 'no' row (1.00 is perfect)."}
        err = {"label": "Right on held-out rows", "value": f"{acc:.0%}",
               "chip": (f"{100 * (acc - acc0):+.0f} pts", "good" if acc - acc0 > 0.02 else "flat"),
               "sub": f"vs {acc0:.0%} always guessing", "foot": "Accuracy at a 50% cut-off."}
    n_used, n_left = len(imp), len([r for r in res.excluded.values() if r != "not selected"])
    cards = [fit, err]
    if not pos.empty:
        cards.append({"label": "Strongest link" if res.reliable else "Top of the ranking", "value": str(pos.iloc[0]["feature"]),
                      "chip": (f"{pos.iloc[0]['share']:.0%} of power", "flat"), "foot": _describe(res, pos.iloc[0]["feature"], unit)})
    cards.append({"label": "Columns compared", "value": f"{n_used}", "sub": f"{n_left} left out",
                  "foot": f"{res.n_train:,} rows to learn from, {res.n_test:,} to test."})
    stat_cards(cards)

    warn = [w for w in res.warnings if "almost perfectly" in w or "holds" in w]
    info = [w for w in res.warnings if w not in warn and "little better than a trivial guess" not in w]
    if warn:
        callout("Check before acting", " ".join(warn), "warn")
    month_reason = res.excluded.get(DATE_PARTS["month"], "")
    if "never occur" in month_reason:
        info.append("Month was left out: the data covers less than a year before the test period, so a seasonal pattern "
                    "cannot be checked on later rows.")
    if info:
        callout("Good to know", " ".join(info), "info")

    if pos.empty:
        _details(res, unit)
        return

    # ---- top drivers
    section(f"What goes with {res.target}", "Share of the model's explanatory power, measured on held-out rows")
    top = pos[pos["share"] >= MIN_CARD_SHARE].head(MAX_CARDS if res.reliable else 0)    # no cards for a weak model
    html = []
    for i, r in enumerate(top.itertuples(), 1):
        html.append(
            f'<div class="kpi compact seg"><div class="kpi-label"><span class="seg-dot" style="background:'
            f'{charts.PALETTE[0] if i == 1 else charts.MUTED}"></span>#{i} · {"category" if r.kind == "category" else "number"}</div>'
            f'<div class="kpi-value">{escape(str(r.feature))}</div><div class="kpi-delta"><span class="chip '
            f'{"good" if res.reliable and not res.near_perfect else "flat"}">{r.share:.0%} of explanatory power</span></div>'
            f'<div class="kpi-foot">{escape(_describe(res, r.feature, unit))}</div></div>')
    if html:
        st.html(f'<div class="kpi-grid" style="grid-template-columns:repeat({min(len(html), 4)}, minmax(0, 1fr))">'
                + "".join(html) + "</div>")
    with st.container(border=True, key="dr_rank"):
        t = pos.head(12).assign(pct=lambda d: 100 * d["share"])
        fig = charts.bar_chart(t, "feature", "pct", "Share of explanatory power", "Percent", unit="%")
        if fig is not None:
            show_figure(fig)
        rest = len(imp) - len(pos)
        st.markdown('<div class="view-insight">How much the model\'s accuracy drops when a column is shuffled, as a share '
                    f'of the total drop.{f" {rest} column(s) made no measurable difference." if rest else ""}</div>',
                    unsafe_allow_html=True)

    # ---- how the target moves with each driver
    if res.groups:
        section(f"How {res.target} changes", f"Average {'share of yes' if unit == '%' else res.target} per group; the dashed line is overall")
        tabs = st.tabs([str(f) for f in res.groups])
        for tab, (feature, g) in zip(tabs, res.groups.items()):
            with tab:
                t = g.assign(v=g["target_mean"] * (100 if unit == "%" else 1), label=g["group"].astype(str))
                ref = res.overall * (100 if unit == "%" else 1)
                fig = charts.bar_chart(t, "label", "v", f"{'Share of yes' if unit == '%' else res.target} by {feature}",
                                       "Share of yes" if unit == "%" else res.target, unit=unit, ref=ref)
                if fig is not None:
                    show_figure(fig)
                lo, hi = int(g["rows"].min()), int(g["rows"].max())
                size = f"Each group has {lo:,} rows." if lo == hi else f"Groups have {lo:,} to {hi:,} rows."
                st.markdown(f'<div class="view-insight">{escape(_describe(res, feature, unit))} {size}</div>',
                            unsafe_allow_html=True)
    _details(res, unit)


def _details(res, unit: str) -> None:
    t1, t2, t3 = st.tabs(["All columns", f"Left out ({len(res.excluded)})", "How this was checked"])
    with t1:
        out = res.importance.assign(Share=100 * res.importance["share"], Type=res.importance["kind"].map(
            {"category": "Category", "numeric": "Number"}))
        out["Direction (association only)"] = [_describe(res, f, unit) for f in out["feature"]]
        st.dataframe(out[["feature", "Type", "Share", "importance", "std", "Direction (association only)"]].rename(
            columns={"feature": "Column", "importance": "Importance", "std": "Spread"}), hide_index=True,
            column_config={"Share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                           "Importance": st.column_config.NumberColumn(format="%.4f"),
                           "Spread": st.column_config.NumberColumn(format="%.4f", help="Standard deviation over repeated shuffles")})
    with t2:
        if res.excluded:
            rows = sorted(res.excluded.items(), key=lambda kv: any(w in kv[1] for w in _ROUTINE))
            st.dataframe(pd.DataFrame(rows, columns=["Column", "Why it was left out"]), hide_index=True)
            st.caption("Columns built from the result would make the model look good for the wrong reason, and columns "
                       "that repeat another would split its importance, so both are left out. Routine reasons are listed last.")
        else:
            st.caption("Every column was used.")
    with t3:
        st.markdown(f"**Split.** {res.split[:1].upper() + res.split[1:]}. {res.n_train:,} rows to learn from, "
                    f"{res.n_test:,} to test ({res.n_rows:,} used in all).")
        mt = res.metrics.rename(columns={"metric": "Measure", "model": "Model", "baseline": "Simple guess"})
        st.dataframe(mt, hide_index=True, column_config={"Model": st.column_config.NumberColumn(format="%.3f"),
                                                         "Simple guess": st.column_config.NumberColumn(format="%.3f")})
        if not res.correlations.empty:
            st.markdown("**Rank correlation with the result** (numeric columns; -1 to 1, 0 is no steady link)")
            st.dataframe(res.correlations.rename(columns={"feature": "Column", "spearman": "Rank correlation"}), hide_index=True,
                         column_config={"Rank correlation": st.column_config.NumberColumn(format="%.3f")})
        how = "the earlier rows and scored on the later ones" if res.split.startswith("chronological") else \
            "most rows and scored on the rest"
        st.caption(f"A random forest is trained on {how}; importance is the drop in that score when one column is "
                   "shuffled. " + res.note)
