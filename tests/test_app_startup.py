from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_starts_with_empty_state():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("No dataset loaded" in m.value for m in at.markdown)


def test_sample_data_button_loads_and_shows_real_counts():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button[0].click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Rows"] == "405" and metrics["Columns"] == "8"
    assert any("synthetic" in w.value.lower() for w in at.warning)
