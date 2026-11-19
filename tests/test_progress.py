"""Progress callbacks (core/progress.py and the analyses that report): order, totals, failures still advance, no Streamlit."""
import numpy as np
import pandas as pd

from analytics import anomaly_detection as ad
from analytics.drivers import run_driver_analysis
from analytics.forecasting import _MODELS, forecast_series
from analytics.segmentation import build_entity_features, run_segmentation
from core import estimates
from core.exceptions import InsufficientDataError
from core.progress import StepRunner, loop_items, notify, skip_message, split_skip
from ingestion.pipeline import ingest_bytes
from ingestion.profiler import profile_dataframe
from semantic.semantic_model import build_semantic_model
from tests.test_drivers import ROLES as DRIVER_ROLES, make as make_drivers
from tests.test_segmentation import ROLES as SEG_ROLES, supplier_df


class Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, step, total, message):
        self.calls.append((step, total, message))


def test_step_runner_reports_start_then_done_in_order():
    rec = Rec()
    run = StepRunner(3, rec)
    for name in ("A", "B", "C"):
        assert run.run(name, lambda n=name: n.lower()) == (True, name.lower(), "")
    run.done()
    assert rec.calls == [(0, 3, "A"), (1, 3, "B"), (2, 3, "C"), (3, 3, "Done")]


def test_a_failing_step_is_reported_as_skipped_still_advances_and_the_run_continues():
    rec = Rec()
    run = StepRunner(3, rec)
    run.run("A", lambda: 1)

    def boom():
        raise InsufficientDataError("Needs a date column.")
    ok, value, reason = run.run("B", boom)
    assert (ok, value, reason) == (False, None, "Needs a date column.")
    assert run.run("C", lambda: 3)[0] is True
    run.done()
    assert rec.calls == [(0, 3, "A"), (1, 3, "B"), (2, 3, skip_message("B", "Needs a date column.")), (2, 3, "C"), (3, 3, "Done")]
    assert run.skipped == [("B", "Needs a date column.")]
    steps = [c[0] for c in rec.calls]
    assert steps == sorted(steps)                                              # the count never goes down


def test_unexpected_errors_show_a_generic_reason_never_a_traceback_or_the_exception_text():
    rec = Rec()
    run = StepRunner(1, rec)
    ok, _v, reason = run.run("X", lambda: 1 / 0, "The X analysis could not be completed.")
    assert not ok and reason == "The X analysis could not be completed."
    assert "division" not in rec.calls[-1][2] and "Traceback" not in rec.calls[-1][2]


def test_run_status_handles_the_cached_wrapper_form_and_skip_and_reuse():
    rec = Rec()
    run = StepRunner(4, rec)
    assert run.run_status("K", lambda: ("ok", 5))[:2] == (True, 5)
    assert run.run_status("L", lambda: ("insufficient", "Too few rows."))[0] is False
    run.skip("M", "switched off")
    run.reuse("N", "your result from the Drivers page")
    assert [c[0] for c in rec.calls] == [0, 1, 2, 3, 3] and run.finished == 4
    assert split_skip(rec.calls[2][2]) == ("L", "Too few rows.")
    assert rec.calls[-1][2] == "N (your result from the Drivers page)"


def test_a_broken_callback_never_breaks_the_work_and_none_is_fine():
    def bad(*_a):
        raise RuntimeError("display crashed")
    run = StepRunner(1, bad)
    assert run.run("A", lambda: 7)[1] == 7
    notify(None, 0, 1, "x")
    assert StepRunner(1).run("A", lambda: 2)[0] is True


def test_loop_items_reports_k_of_n_before_each_item():
    rec = Rec()
    assert list(loop_items(rec, ["a", "b"], "Doing")) == ["a", "b"]
    assert rec.calls == [(0, 2, "Doing, 1 of 2"), (1, 2, "Doing, 2 of 2")]


def _daily(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2023-01-01", periods=n)
    return pd.DataFrame({"date": d, "v": 100 + 10 * np.sin(np.arange(n) / 7) + rng.normal(0, 3, n)})


def test_forecast_reports_one_message_per_model_in_order():
    rec = Rec()
    forecast_series(_daily(), "date", "v", "D", "sum", 14, progress=rec)
    n = len(_MODELS)
    assert [c[:2] for c in rec.calls] == [(i, n) for i in range(n)]
    assert [c[2] for c in rec.calls] == [f"Comparing models, {i + 1} of {n}" for i in range(n)]


def test_anomaly_detection_reports_every_method_and_column():
    df = pd.DataFrame({"a": np.random.default_rng(1).normal(0, 1, 200), "b": np.random.default_rng(2).normal(5, 2, 200)})
    rec = Rec()
    ad.run_anomaly_detection(df, ["a", "b"], ("mad", "iqr", "isolation_forest"), None, progress=rec)
    assert [c[:2] for c in rec.calls] == [(i, 5) for i in range(5)]
    assert "Screening a" in rec.calls[0][2] and "Isolation" in rec.calls[-1][2]


def test_a_failing_column_still_advances_the_anomaly_count():
    df = pd.DataFrame({"a": ["x"] * 100, "b": np.random.default_rng(2).normal(5, 2, 100)})   # text column cannot be screened numerically
    rec = Rec()
    try:
        ad.run_anomaly_detection(df, ["a", "b"], ("mad",), None, progress=rec)
    except InsufficientDataError:
        pass
    assert [c[0] for c in rec.calls] == [0, 1]


def test_segmentation_reports_each_k_tried():
    df = supplier_df()
    prof = profile_dataframe(df)
    table = build_entity_features(df, build_semantic_model(df, prof, SEG_ROLES), "supplier")
    rec = Rec()
    res = run_segmentation(table, None, None, "robust", progress=rec)
    assert len(rec.calls) >= 1 and all(c[1] == rec.calls[0][1] for c in rec.calls)
    assert [c[0] for c in rec.calls] == list(range(len(rec.calls)))
    assert rec.calls[0][2].startswith("Trying 2 segments") and res.k >= 2


def test_driver_analysis_reports_stages_then_each_column_then_finishes():
    df = make_drivers()
    prof = profile_dataframe(df)
    rec = Rec()
    r = run_driver_analysis(df, "y", prof, build_semantic_model(df, prof, DRIVER_ROLES), progress=rec)
    steps = [c[0] for c in rec.calls]
    assert steps == sorted(steps) and rec.calls[0][2] == "Preparing the data" and rec.calls[-1][:2] == (4, 4)
    cols = [c[2] for c in rec.calls if c[2].startswith("Checking which columns matter")]
    n = len(r.importance)
    assert cols[0].endswith(f"0 of {n} done") and cols[-1].endswith(f"{n} of {n} done")
    assert any("no partial progress" in c[2] for c in rec.calls)                 # a single fit is named, not given a fake percentage


def test_driver_progress_is_identical_with_one_or_many_threads_and_results_do_not_change():
    import os
    df = make_drivers()
    prof = profile_dataframe(df)
    m = build_semantic_model(df, prof, DRIVER_ROLES)
    got = {}
    old = os.environ.get("BI_WORKERS")
    try:
        for w in ("1", "3"):
            os.environ["BI_WORKERS"] = w
            rec = Rec()
            got[w] = (run_driver_analysis(df, "y", prof, m, progress=rec), [c[2] for c in rec.calls])
    finally:
        os.environ.pop("BI_WORKERS", None) if old is None else os.environ.__setitem__("BI_WORKERS", old)
    assert got["1"][0].importance.equals(got["3"][0].importance) and got["1"][0].metrics.equals(got["3"][0].metrics)
    assert got["1"][1] == got["3"][1]


def test_ingest_reports_three_stages_and_columns():
    rec = Rec()
    csv = pd.DataFrame({"a": range(50), "b": list("xy") * 25}).to_csv(index=False).encode()
    ingest_bytes(csv, "t.csv", max_bytes=10**7, progress=rec)
    msgs = [c[2] for c in rec.calls]
    assert msgs[0].startswith("Reading") and "Profiling columns, 1 of 2" in msgs and "Profiling columns, 2 of 2" in msgs
    assert msgs[-2:] == ["Checking data quality", "Done"] and rec.calls[-1][:2] == (3, 3)
    assert all(c[1] == 3 for c in rec.calls)


def test_estimates_are_hidden_for_small_files_and_say_they_are_rough_for_big_ones():
    assert estimates.describe_estimate("report", 2000) is None
    text = estimates.describe_estimate("report", 180_000)
    assert "180,000 rows" in text and "rough guide" in text and "to" in text
    assert estimates.describe_estimate("nope", 10**6) is None
