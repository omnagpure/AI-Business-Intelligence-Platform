from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_starts_with_empty_state():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Try the sample data" in html and "Use your own data" in html
    assert [b.label for b in at.button][:2] == ["Load sample data", "Upload a file"]


def test_sample_data_button_loads_and_shows_real_counts():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button[0].click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Rows"] == "405" and metrics["Columns"] == "8"
    assert any("synthetic" in w.value.lower() for w in at.warning)


def test_home_after_loading_shows_next_step_and_context_line():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button[0].click().run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Next step" in html and "ctx-line" in html and "405 rows" in html
    assert any(b.key == "home_next" for b in at.button)
    assert 'class="pg-head"' in html and "<h1" not in html.split("</style>")[-1]
