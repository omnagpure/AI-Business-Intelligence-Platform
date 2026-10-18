from config.settings import load_settings


def test_defaults(monkeypatch):
    for k in ("BI_MAX_UPLOAD_MB", "BI_PREVIEW_ROWS"):
        monkeypatch.delenv(k, raising=False)
    s = load_settings()
    assert s.max_upload_mb == 200 and s.preview_rows == 100
    assert s.max_upload_bytes == 200 * 1024 * 1024


def test_env_override_and_invalid(monkeypatch):
    monkeypatch.setenv("BI_MAX_UPLOAD_MB", "5")
    monkeypatch.setenv("BI_PREVIEW_ROWS", "abc")
    s = load_settings()
    assert s.max_upload_mb == 5
    assert s.preview_rows == 100  # invalid value falls back to default
