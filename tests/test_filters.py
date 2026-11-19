from datetime import date

from streamlit.testing.v1 import AppTest


def _app():
    import streamlit as st

    from ingestion.pipeline import ingest_sample
    from ui import state
    from ui.analytics_state import base_key
    from ui.filters import begin_run
    from ui.layout import configure_page
    configure_page()
    begin_run()
    if state.get_result() is None:
        state.set_result(ingest_sample())
    if st.session_state.get("_reload"):
        st.session_state["_reload"] = False
        state.set_result(ingest_sample())
    page = st.session_state.get("_page", "explorer")
    if page == "keys":
        k_off = base_key()                       # scope off: a page that ignores the filter
        state.set_filter_scope(True)
        st.session_state["keys"] = (k_off, base_key())
        return
    from ui.dashboard_page import render_dashboard_page
    from ui.explorer_page import render_explorer_page
    from ui.forecasting_page import render_forecasting_page
    {"explorer": render_explorer_page, "dashboard": render_dashboard_page, "forecasting": render_forecasting_page}[page]()


def _rows(at):
    return [c.value for c in at.caption if c.value.startswith(("Showing", "All "))]


def _filter(at, column="product_category"):
    [s for s in at.selectbox if s.label == "Filter by"][0].select(column).run()
    ms = [m for m in at.multiselect if m.label.endswith("is one of")][0]
    ms.select(ms.options[0]).run()
    return at


def _at():
    return AppTest.from_function(_app, default_timeout=120).run()


def test_bar_is_visible_without_expanding_anything():
    at = _at()
    assert not at.exception and _rows(at) == ["All 405 rows"]
    assert [s.label for s in at.selectbox][0] == "Filter by" and at.date_input[0].label.startswith("Date range")


def test_filter_applies_and_follows_the_user_to_other_pages():
    at = _filter(_at())
    assert not at.exception and _rows(at) == ["Showing 123 of 405 rows"]
    assert any(b.label == "product_category: Electronics" for b in at.button)
    for page in ("forecasting", "dashboard"):
        at.session_state["_page"] = page
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert _rows(at) == ["Showing 123 of 405 rows"]


def test_chip_and_reset_remove_filters():
    at = _filter(_at())
    [b for b in at.button if b.label == "product_category: Electronics"][0].click().run()
    assert _rows(at) == ["All 405 rows"]
    at = _filter(_at())
    [b for b in at.button if b.label == "Reset"][0].click().run()
    assert _rows(at) == ["All 405 rows"] and not [b for b in at.button if b.label.startswith("product_category:")]


def test_date_range_filters_rows():
    at = _at()
    lo = at.date_input[0].min
    at.date_input[0].set_value((lo, date(lo.year, lo.month, 28))).run()
    assert not at.exception
    shown = _rows(at)[0]
    assert shown.startswith("Showing") and "of 405 rows" in shown
    assert any(b.label.startswith("order_date:") for b in at.button)


def test_a_new_dataset_clears_the_filter():
    at = _filter(_at())
    at.session_state["_reload"] = True
    at.run()
    assert _rows(at) == ["All 405 rows"]


def test_filter_joins_the_cache_key_only_on_pages_that_apply_it():
    at = _at()
    at.session_state["_page"] = "keys"
    at.run()
    off, on = at.session_state["keys"]
    assert off == on                                      # no filter: results are shared with pages that ignore it
    at.session_state["_page"] = "explorer"
    at.run()
    at = _filter(at)
    at.session_state["_page"] = "keys"
    at.run()
    off, on = at.session_state["keys"]
    assert off != on                                      # filtered pages never reuse unfiltered results
