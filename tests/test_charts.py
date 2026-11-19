"""Chart data preparation (pure pandas) plus smoke tests for the builders when Plotly is installed."""
import numpy as np
import pandas as pd
import pytest

from analytics.formatting import format_value
from visualization import charts


def test_clean_xy_drops_nan_and_infinite():
    df = pd.DataFrame({"x": [1, 2, np.inf, 4, 5], "y": [1, np.nan, 3, -np.inf, 5]})
    f = charts.clean_xy(df, "x", "y", x_numeric=True)
    assert list(f["x"]) == [1, 5] and list(f["y"]) == [1, 5]


def test_bin_categories_caps_hundreds_of_categories():
    labels = pd.Series([f"c{i}" for i in range(500)] * 2)
    r = charts.bin_categories(labels, None, 10)
    assert len(r) == 11 and r["label"].iloc[-1] == "Other" and r["value"].sum() == 1000
    vals = pd.Series(np.arange(1000.0))
    r2 = charts.bin_categories(pd.Series([f"c{i % 50}" for i in range(1000)]), vals, 10)
    assert len(r2) == 11 and r2["value"].sum() == pytest.approx(vals.sum())


def test_sample_points_is_capped_and_reproducible():
    f = pd.DataFrame({"x": range(10000), "y": range(10000)})
    a, sa = charts.sample_points(f, 500)
    b, _ = charts.sample_points(f, 500)
    assert len(a) == 500 and sa and a.equals(b)
    assert charts.sample_points(f.head(10), 500)[1] is False


def test_format_value():
    assert format_value(None, "amount") == "Not calculable"
    assert format_value(float("nan"), "percent") == "Not calculable"
    assert format_value(12.345, "percent") == "12.3%"
    assert format_value(1234567, "count") == "1,234,567"
    assert format_value(36784735.01, "amount") == "36.78M" and format_value(559.449, "amount") == "559.45"


def test_builders_return_none_without_plottable_data_and_figures_otherwise():
    pytest.importorskip("plotly")
    bad = pd.DataFrame({"x": [np.nan, np.inf], "y": [np.nan, np.inf]})
    assert charts.scatter(bad, "x", "y", "t") is None
    assert charts.histogram(bad["x"], "t") is None
    assert charts.heatmap(pd.DataFrame(), "t") is None
    t = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=4, freq="MS"), "v": [1.0, np.inf, 3.0, 4.0]})
    fig = charts.line_chart(t, "d", "v", "Trend", partial=pd.Series([False, False, False, True]))
    assert fig is not None and len(fig.data) == 2 and not np.isinf(np.asarray(fig.data[0].y, dtype=float)).any()
    many = pd.DataFrame({"label": [f"c{i}" for i in range(200)], "value": np.arange(200.0)})
    assert len(charts.bar_chart(many, "label", "value", "t").data[0].y) == 11
    assert charts.share_chart(many, "label", "value", "t").data[0].type == "bar"     # no 200-slice pie
    few = pd.DataFrame({"label": list("abc"), "value": [1.0, 2.0, 3.0]})
    assert charts.share_chart(few, "label", "value", "t").data[0].type == "pie"
    m = pd.DataFrame(np.eye(3), columns=list("abc"), index=list("abc"))
    assert charts.heatmap(m, "t").data[0].zmin == -1
