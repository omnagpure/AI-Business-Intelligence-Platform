"""Report options and assembly (no Streamlit): PDF by default, HTML only when asked, steps and progress, failures isolated, charts drawn once."""
import pytest

from analytics.kpis import compute_kpis
from core.exceptions import InsufficientDataError
from core.progress import split_skip
from ingestion.pipeline import ingest_sample
from reporting import charts
from reporting.assembly import ReportContext, Sources, assemble_report, step_names
from semantic.semantic_model import build_semantic_model
from tests.test_reporting import _forecast


class Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, step, total, message):
        self.calls.append((step, total, message))


@pytest.fixture(scope="module")
def world():
    r = ingest_sample()
    df, p = r.dataset.df, r.profile
    m = build_semantic_model(df, p, {}, None, None)
    ctx = ReportContext(r.dataset.filename, True, r.dataset.fingerprint, p, r.quality, len(df), m)
    return ctx, df, m


def _sources(df, m, **over):
    calls = {"kpis": 0}

    def kpis():
        calls["kpis"] += 1
        return "ok", compute_kpis(df, m)
    adv = {"forecast": lambda progress=None: ("ok", _forecast()),
           "anomalies": lambda progress=None: ("insufficient", "No numeric column is suitable for anomaly screening."),
           "segmentation": lambda progress=None: ("insufficient", "No entity column."),
           "drivers": lambda progress=None: ("insufficient", "No target.")}
    kw = dict(kpis=kpis, findings=lambda: ("insufficient", "no patterns"), advanced=adv)
    kw.update(over)
    return Sources(**kw), calls


def test_pdf_is_built_by_default_and_html_only_when_asked(world):
    ctx, df, m = world
    src, _ = _sources(df, m)
    pdf_only = assemble_report(ctx, src)
    assert set(pdf_only.files) == {"pdf"} and pdf_only.files["pdf"].startswith(b"%PDF")
    both = assemble_report(ctx, src, with_html=True)
    assert set(both.files) == {"pdf", "html"} and b"</html>" in both.files["html"]


def test_steps_are_named_in_order_and_totals_match_the_options(world):
    ctx, df, m = world
    assert step_names(True, False) == ["Metrics", "Insights", "Forecast", "Anomalies", "Segments", "Drivers", "PDF"]
    assert step_names(True, True)[-2:] == ["PDF", "HTML"]
    src, _ = _sources(df, m)
    for html, total in ((False, 7), (True, 8)):
        rec = Rec()
        assemble_report(ctx, src, with_html=html, progress=rec)
        assert {c[1] for c in rec.calls} == {total} and rec.calls[-1] == (total, total, "Done")
        starts = [c[2] for c in rec.calls if split_skip(c[2]) is None and c[2] != "Done" and ":" not in c[2]]
        assert starts[:2] == ["Metrics", "Insights"] and starts[-1] == ("HTML" if html else "PDF")
        steps = [c[0] for c in rec.calls]
        assert steps == sorted(steps)


def test_failed_analyses_are_skipped_with_the_reason_and_the_report_still_builds(world):
    ctx, df, m = world
    src, _ = _sources(df, m)
    rec = Rec()
    out = assemble_report(ctx, src, progress=rec)
    names = [n for n, _r in out.skipped]
    assert names == ["Insights", "Anomalies", "Segments", "Drivers"]
    assert ("Anomalies", "No numeric column is suitable for anomaly screening.") in out.skipped
    assert out.files["pdf"].startswith(b"%PDF") and out.data.section("forecast").status == "ok"
    assert out.data.section("anomalies").status == "skipped" and "anomaly screening" in out.data.section("anomalies").reason
    assert [c[0] for c in rec.calls] == sorted(c[0] for c in rec.calls) and rec.calls[-1][0] == rec.calls[-1][1]


def test_one_failing_format_never_blocks_the_other(world):
    ctx, df, m = world
    src, _ = _sources(df, m)

    def broken(_d):
        raise RuntimeError("renderer exploded")
    out = assemble_report(ctx, src, with_html=True, renderers={"html": broken})
    assert "pdf" in out.files and "html" not in out.files
    assert out.errors == {"html": "The HTML file could not be created."} and "exploded" not in str(out.errors)
    out2 = assemble_report(ctx, src, with_html=True, renderers={"pdf": broken})
    assert "html" in out2.files and "pdf" not in out2.files and set(out2.errors) == {"pdf"}


def test_advanced_off_skips_the_four_analyses_with_one_clear_reason(world):
    ctx, df, m = world
    src, _ = _sources(df, m)
    rec = Rec()
    out = assemble_report(ctx, src, advanced=False, progress=rec)
    reasons = {n: r for n, r in out.skipped}
    assert all(reasons[n] == "Advanced analyses were switched off for this report." for n in ("Forecast", "Anomalies", "Segments", "Drivers"))
    assert out.data.section("forecast").status == "skipped" and out.files["pdf"].startswith(b"%PDF")


def test_results_the_user_already_ran_are_reused_not_recomputed(world):
    ctx, df, m = world
    ran = []
    src, _ = _sources(df, m, advanced={k: (lambda progress=None, k=k: ran.append(k) or ("ok", None)) for k in ("forecast", "anomalies", "segmentation", "drivers")})
    out = assemble_report(ctx, src, user_results={"forecast": _forecast()})
    assert "forecast" not in ran and set(ran) == {"anomalies", "segmentation", "drivers"}
    assert out.used_user_settings == ("Forecasting",)
    assert any("Forecasting" in line for line in out.data.methodology)


def test_inner_analysis_progress_is_shown_under_its_step_without_moving_the_count(world):
    ctx, df, m = world

    def forecast(progress=None):
        progress(2, 7, "Comparing models, 3 of 7")
        return "ok", _forecast()
    src, _ = _sources(df, m)
    src = Sources(src.kpis, src.findings, {**src.advanced, "forecast": forecast})
    rec = Rec()
    assemble_report(ctx, src, progress=rec)
    inner = [c for c in rec.calls if c[2] == "Forecast: Comparing models, 3 of 7"]
    assert len(inner) == 1 and inner[0][1] == 7 and inner[0][0] == 2          # Metrics and Insights finished; Forecast not yet


def test_charts_are_drawn_once_and_shared_by_pdf_and_html(world):
    ctx, df, m = world
    src, _ = _sources(df, m)
    count = {"n": 0}
    real = charts._png

    def counting(fig):
        count["n"] += 1
        return real(fig)
    charts._png = counting
    try:
        pdf_only = assemble_report(ctx, src)
        n_pdf = count["n"]
        count["n"] = 0
        both = assemble_report(ctx, src, with_html=True)
        n_both = count["n"]
    finally:
        charts._png = real
    assert n_pdf >= 1 and n_both == n_pdf                    # building the HTML as well draws no extra chart
    assert len(both.data.sections) == len(pdf_only.data.sections)


def test_kpis_come_from_the_supplied_source_exactly_once(world):
    ctx, df, m = world
    src, calls = _sources(df, m)
    assemble_report(ctx, src, with_html=True)
    assert calls["kpis"] == 1


def test_a_failing_metrics_source_is_reported_and_the_pdf_still_builds(world):
    ctx, df, m = world

    def kpis():
        raise InsufficientDataError("nothing to measure")
    src, _ = _sources(df, m, kpis=kpis)
    out = assemble_report(ctx, src)
    assert ("Metrics", "nothing to measure") in out.skipped and out.data.section("kpis").status == "skipped"
    assert out.files["pdf"].startswith(b"%PDF")
