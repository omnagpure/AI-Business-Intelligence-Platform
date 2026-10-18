import io

import pandas as pd
import pytest

from core.exceptions import DataValidationError, FileTooLargeError, UnsupportedFileFormatError
from ingestion.file_loader import list_excel_sheets, load_dataset

MAX = 10 * 1024 * 1024


def load(data: bytes, name="t.csv", **kw):
    return load_dataset(data, name, max_bytes=MAX, **kw)


def xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for n, d in sheets.items():
            d.to_excel(w, sheet_name=n, index=False)
    return buf.getvalue()


def test_csv_basic():
    ds = load(b"a,b\n1,x\n2,y\n")
    assert ds.df.shape == (2, 2) and ds.file_type == "text" and ds.delimiter == ","
    assert list(ds.df.columns) == ["a", "b"]


def test_csv_cp1252_decoded_correctly():
    ds = load("name\ncafé\n".encode("cp1252"))
    assert ds.encoding == "cp1252" and ds.df["name"].iloc[0] == "café"
    assert any("not UTF-8" in w for w in ds.warnings)


def test_csv_utf8_bom_header_clean():
    ds = load("\ufeffa,b\n1,2\n".encode("utf-8"))
    assert list(ds.df.columns) == ["a", "b"]


def test_csv_semicolon_and_tab_delimiters():
    assert load(b"a;b\n1;2\n3;4\n").df.shape == (2, 2)
    assert load(b"a\tb\n1\t2\n3\t4\n").df.shape == (2, 2)


def test_csv_leading_comment_lines_skipped():
    ds = load(b"# source: x\n# rows: 2\na,b\n1,2\n3,4\n")
    assert ds.skipped_header_lines == 2 and list(ds.df.columns) == ["a", "b"] and len(ds.df) == 2


def test_duplicate_column_names_warned():
    ds = load(b"a,a\n1,2\n")
    assert any("Duplicate column names" in w for w in ds.warnings)


def test_empty_and_header_only_rejected():
    for raw in (b"", b"\n\n", b"a,b\n"):
        with pytest.raises(DataValidationError):
            load(raw)


def test_unsupported_extension():
    with pytest.raises(UnsupportedFileFormatError):
        load(b"a,b\n1,2", "x.exe")


def test_binary_with_csv_extension_rejected():
    with pytest.raises(UnsupportedFileFormatError):
        load(b"PK\x03\x04" + b"\x00" * 50, "fake.csv")
    with pytest.raises(UnsupportedFileFormatError):
        load(b"%PDF-1.4\x00\x00", "fake.csv")


def test_fake_xlsx_rejected():
    with pytest.raises(UnsupportedFileFormatError):
        load(b"a,b\n1,2\n", "fake.xlsx")


def test_zip_that_is_not_workbook_rejected():
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("hello.txt", "hi")
    with pytest.raises(UnsupportedFileFormatError):
        load(buf.getvalue(), "x.xlsx")


def test_size_limit():
    with pytest.raises(FileTooLargeError):
        load_dataset(b"a,b\n1,2\n", "t.csv", max_bytes=3)


def test_excel_sheets_and_selection():
    data = xlsx_bytes({"First": pd.DataFrame({"a": [1, 2]}), "Second": pd.DataFrame({"b": [1, 2, 3]})})
    assert list_excel_sheets(data) == ["First", "Second"]
    assert load(data, "w.xlsx").sheet == "First"
    ds = load(data, "w.xlsx", sheet="Second")
    assert ds.sheet == "Second" and ds.df.shape == (3, 1) and list(ds.df.columns) == ["b"]


def test_excel_unknown_sheet_falls_back_to_first():
    data = xlsx_bytes({"Only": pd.DataFrame({"a": [1]})})
    assert load(data, "w.xlsx", sheet="Nope").sheet == "Only"


def test_fingerprint_changes_with_content_and_sheet():
    a = load(b"a\n1\n").fingerprint
    assert a == load(b"a\n1\n").fingerprint
    assert a != load(b"a\n2\n").fingerprint


def test_tsv_and_txt_extensions():
    assert load(b"a\tb\n1\t2\n", "x.tsv").df.shape == (1, 2)
    assert load(b"a|b\n1|2\n3|4\n", "x.txt").df.shape == (2, 2)


def test_csv_title_rows_above_header_skipped():
    raw = b"Inventory report,,,\nGenerated 2024-05-01,,,\nsku,warehouse,on_hand,lead_days\nA1,W1,10,5\nA2,W2,20,7\n"
    ds = load(raw)
    assert list(ds.df.columns) == ["sku", "warehouse", "on_hand", "lead_days"] and len(ds.df) == 2
    assert any("title row" in w for w in ds.warnings)


def test_excel_title_rows_above_header_skipped():
    rows = [["Stock levels", None, None], ["Run date", "2024-05-01", None],
            ["sku", "on_hand", "reorder_point"], ["A1", 10, 5], ["A2", 20, 8]]
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, header=False, index=False, engine="openpyxl")
    ds = load(buf.getvalue(), "s.xlsx")
    assert list(ds.df.columns) == ["sku", "on_hand", "reorder_point"] and len(ds.df) == 2
    assert ds.df["on_hand"].tolist() == [10, 20]


def test_normal_header_not_skipped():
    ds = load(b"a,b,c\n1,2,3\n")
    assert ds.skipped_header_lines == 0 and list(ds.df.columns) == ["a", "b", "c"]


def test_decimal_comma_semicolon_csv():
    ds = load(b"sku;cost;qty\nA;12,5;3\nB;7,25;4\n")
    assert ds.df["cost"].tolist() == [12.5, 7.25]
    assert any("decimal comma" in w for w in ds.warnings)


def test_comma_thousands_not_mistaken_for_decimal():
    ds = load(b'sku,cost\nA,"1,200"\nB,"3,400"\n')
    assert ds.df["cost"].tolist() == ["1,200", "3,400"]  # kept as text for the cleaning step


def test_fake_xls_rejected():
    with pytest.raises(UnsupportedFileFormatError):
        load(b"<html><table></table></html>", "export.xls")


def test_legacy_xls_roundtrip():
    xlwt = pytest.importorskip("xlwt")
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Stock")
    for r, row in enumerate([["sku", "qty"], ["A", 5], ["B", 7]]):
        for c, v in enumerate(row):
            ws.write(r, c, v)
    buf = io.BytesIO()
    wb.save(buf)
    ds = load(buf.getvalue(), "old.xls")
    assert ds.sheet == "Stock" and ds.df["qty"].tolist() == [5, 7]


def test_utf16_text_file():
    ds = load("a,b\n1,2\n3,4\n".encode("utf-16"), "u.csv")
    assert ds.df.shape == (2, 2) and ds.encoding == "utf-16"
