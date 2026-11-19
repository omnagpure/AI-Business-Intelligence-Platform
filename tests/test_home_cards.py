import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from streamlit.testing.v1 import AppTest

from ui import home_cards as hc

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _cards(html):
    return {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(
        r'card-k">([^<]*)</div><div class="card-t[^"]*">([^<]*)</div><div class="card-d">([^<]*)', html)}


def test_home_shows_five_cards_with_real_results_or_reasons():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    cards = _cards(" ".join(m.value for m in at.markdown))
    assert set(cards) == {"Forecast", "Anomalies", "Segments", "Drivers", "Explorer"}
    assert cards["Explorer"][0] == "Browse all 405 rows"
    assert all(head and head != "Could not be completed" for head, _ in cards.values())
    assert all(detail for _, detail in cards.values())          # a result line or a reason, never blank
    assert {"home_open_forecasting", "home_open_anomalies", "home_open_segmentation",
            "home_open_drivers", "home_open_explorer"} <= {b.key for b in at.button}


def test_forecast_summary_describes_direction_from_the_numbers():
    hist = pd.DataFrame({"period": pd.date_range("2024-01-01", periods=6, freq="MS"), "value": [100] * 6})
    fc = pd.DataFrame({"period": pd.date_range("2024-07-01", periods=3, freq="MS"), "forecast": [110, 110, 110]})
    r = SimpleNamespace(freq="M", agg="sum", value_col="revenue", history=hist, forecast=fc, horizon=3, chosen="a", baseline="b",
                        scores=(SimpleNamespace(key="a", label="Seasonal (x)"), SimpleNamespace(key="b", label="Naive")))
    card = hc.summarise_forecast(r)
    assert "rise about 10%" in card.headline and "Sum of revenue" in card.headline
    r.forecast = fc.assign(forecast=[100, 100, 100])
    assert "stay about flat" in hc.summarise_forecast(r).headline


def test_driver_summary_is_honest_when_the_model_is_not_reliable():
    weak = SimpleNamespace(reliable=False, importance=pd.DataFrame({"feature": ["a"]}))
    assert "No column clearly explains" in hc.summarise_drivers(weak, "revenue").headline
    ok = SimpleNamespace(reliable=True, importance=pd.DataFrame({"feature": ["lead_time", "x"]}))
    card = hc.summarise_drivers(ok, "delay")
    assert "lead_time" in card.headline and "not proof of cause" in card.detail


def test_anomaly_summary_handles_zero_findings():
    rep = SimpleNamespace(findings=pd.DataFrame({"row": []}))
    assert "Nothing unusual" in hc.summarise_anomalies(rep, 3).headline
