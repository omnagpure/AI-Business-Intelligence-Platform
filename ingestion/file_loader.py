"""Load CSV / XLSX bytes into a DataFrame with validation. No Streamlit imports."""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zipfile
from dataclasses import dataclass, field

import pandas as pd

from config.settings import EXCEL_EXTENSIONS, SUPPORTED_EXTENSIONS
from core.exceptions import DataValidationError, UnsupportedFileFormatError
from core.logging_config import get_logger
from core.validation import sanitize_filename, validate_extension, validate_size

log = get_logger("ingestion.loader")

_XLSX_MAGIC = b"PK\x03\x04"
_XLS_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_DUP_SUFFIX = re.compile(r"^(?P<base>.+)\.(?P<n>\d+)$")


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    filename: str
    file_type: str                       # "text" | "excel"
    fingerprint: str                     # sha256 of content + sheet
    sheet_names: list[str] = field(default_factory=list)
    sheet: str | None = None
    encoding: str | None = None
    delimiter: str | None = None
    skipped_header_lines: int = 0
    warnings: list[str] = field(default_factory=list)
    size_bytes: int = 0
    is_sample: bool = False


def fingerprint_bytes(data: bytes, sheet: str | None = None) -> str:
    h = hashlib.sha256(data)
    h.update((sheet or "").encode("utf-8"))
    return h.hexdigest()


def detect_file_type(data: bytes, filename: str) -> str:
    """Return 'text' or 'excel' based on extension AND content; reject mismatches."""
    ext = validate_extension(filename)
    if ext in EXCEL_EXTENSIONS:
        if ext == ".xls":
            if data[:8] != _XLS_MAGIC:
                raise UnsupportedFileFormatError(
                    "This file has an .xls extension but is not a valid legacy Excel workbook. "
                    "Some systems export HTML or CSV with an .xls name; re-save it as .xlsx or .csv.")
            return "excel"
        if data[:4] != _XLSX_MAGIC:
            raise UnsupportedFileFormatError(
                f"This file has a {ext} extension but is not a valid Excel workbook.")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "xl/workbook.xml" not in zf.namelist():
                    raise UnsupportedFileFormatError("The file is a zip archive, not an Excel workbook.")
        except zipfile.BadZipFile as exc:
            raise UnsupportedFileFormatError("The Excel file is corrupted.") from exc
        return "excel"
    looks_binary = (data[:4] == _XLSX_MAGIC or data[:8] == _XLS_MAGIC or data[:5] == b"%PDF-"
                    or data[:2] == b"MZ" or b"\x00" in data[:8192])
    if looks_binary and data[:2] not in (b"\xff\xfe", b"\xfe\xff"):  # UTF-16 text contains NULs
        raise UnsupportedFileFormatError(
            f"This file has a {ext} extension but its content is binary, not text.")
    return "text"


def decode_text(data: bytes) -> tuple[str, str]:
    """Decode bytes: UTF-16 by BOM, then UTF-8, then Windows-1252, then Latin-1 (never fails).

    Note: a non-UTF-8 file is assumed to be Western European (cp1252). Files in other legacy
    encodings may decode with wrong characters; the loader reports the encoding used."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16"), "utf-16"
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise DataValidationError("Could not determine the text encoding of this CSV file.")


def _leading_comment_lines(lines: list[str]) -> int:
    n = 0
    for line in lines:
        if line.lstrip().startswith("#") or not line.strip():
            n += 1
        else:
            break
    return n


def _is_header_cell(v) -> bool:
    if not isinstance(v, str) or not v.strip():
        return False
    if re.match(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", v):  # looks like a date, so it is data
        return False
    try:
        float(v.replace(",", ""))
        return False
    except ValueError:
        return True


def find_header_index(rows: list[list]) -> int:
    """Index of the first row that looks like a header: wide enough (relative to the widest row)
    and made only of non-numeric text. Title/summary rows above the header are skipped."""
    def filled(r):
        return [c for c in r if c is not None and str(c).strip() not in ("", "nan", "NaN") and not str(c).startswith("Unnamed:")]
    widths = [len(filled(r)) for r in rows]
    if not widths or max(widths) < 2:
        return 0
    need = max(2, math.ceil(0.8 * max(widths)))
    for i, r in enumerate(rows):
        f = filled(r)
        if len(f) >= need and all(_is_header_cell(c) for c in f):
            return i
    return 0


def _use_decimal_comma(sample: str, delim: str) -> bool:
    """European-style files: ';' delimiter, numbers like 12,5 and no 12.5 anywhere."""
    if delim == ",":
        return False
    return bool(re.search(r"\d,\d", sample)) and not re.search(r"\d\.\d", sample)


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _duplicate_name_warnings(columns: list[str]) -> list[str]:
    cols = set(map(str, columns))
    renamed = []
    for c in map(str, columns):
        m = _DUP_SUFFIX.match(c)
        if m and m.group("base") in cols:
            renamed.append(m.group("base"))
    if not renamed:
        return []
    names = ", ".join(sorted(set(renamed))[:5])
    return [f"Duplicate column names were found and renamed with numeric suffixes (e.g. {names})."]


def _finalize(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    if df.shape[1] == 0 or df.shape[0] == 0:
        raise DataValidationError(
            "The file contains no data rows. Check that it has a header row and at least one record."
        )
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        warnings.append(f"{len(unnamed)} column(s) have no header name.")
    warnings.extend(_duplicate_name_warnings(list(df.columns)))
    df.columns = [str(c) for c in df.columns]
    return df


def list_excel_sheets(data: bytes, filename: str = "x.xlsx") -> list[str]:
    detect_file_type(data, filename)
    try:
        with pd.ExcelFile(io.BytesIO(data), engine=_excel_engine(filename)) as xf:
            return list(xf.sheet_names)
    except Exception as exc:  # openpyxl raises many types
        log.warning("Excel sheet listing failed: %s", type(exc).__name__)
        raise DataValidationError("Could not read the Excel workbook.") from exc


def _load_csv(data: bytes, filename: str, warnings: list[str]):
    text, enc = decode_text(data)
    if enc not in ("utf-8", "utf-8-sig"):
        warnings.append(f"File is not UTF-8; decoded as {enc}.")
    lines = text.splitlines()
    skip = _leading_comment_lines(lines[:50])
    if skip:
        warnings.append(f"Skipped {skip} comment/blank line(s) above the header.")
    body = lines[skip : skip + 30]
    sample = "\n".join(body)
    delim = "\t" if filename.lower().endswith(".tsv") else _sniff_delimiter(sample)
    rows = list(csv.reader(body, delimiter=delim))
    offset = find_header_index(rows)
    if offset:
        warnings.append(f"Skipped {offset} title row(s) above the header.")
    decimal = "," if _use_decimal_comma("\n".join(lines[skip + offset : skip + offset + 200]), delim) else "."
    if decimal == ",":
        warnings.append("Numbers use a decimal comma; parsed accordingly.")
    try:
        df = pd.read_csv(io.StringIO(text), sep=delim, skiprows=skip + offset, low_memory=False,
                         on_bad_lines="warn", decimal=decimal)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError("The file is malformed and could not be parsed.",
                                  detail=str(exc)[:300]) from exc
    return df, enc, delim, skip + offset


def _excel_engine(filename: str) -> str:
    return "xlrd" if filename.lower().endswith(".xls") else "openpyxl"


def load_dataset(data: bytes, filename: str, *, max_bytes: int, sheet: str | None = None) -> LoadedDataset:
    """Validate and parse an uploaded file. The returned DataFrame is the untouched original."""
    safe_name = sanitize_filename(filename)
    validate_size(len(data), max_bytes)
    ftype = detect_file_type(data, safe_name)
    warnings: list[str] = []
    sheet_names: list[str] = []
    enc = delim = None
    skipped = 0

    if ftype == "text":
        df, enc, delim, skipped = _load_csv(data, safe_name, warnings)
    else:
        engine = _excel_engine(safe_name)
        sheet_names = list_excel_sheets(data, safe_name)
        if not sheet_names:
            raise DataValidationError("The workbook has no worksheets.")
        sheet = sheet if sheet in sheet_names else sheet_names[0]
        try:
            df = pd.read_excel(io.BytesIO(data), sheet_name=sheet, engine=engine)
            head_rows = [list(df.columns)] + df.head(29).values.tolist()
            skipped = find_header_index(head_rows)
            if skipped:  # title rows above the header: re-read with the detected header row
                df = pd.read_excel(io.BytesIO(data), sheet_name=sheet, engine=engine, header=skipped)
        except Exception as exc:
            log.warning("Excel read failed: %s", type(exc).__name__)
            raise DataValidationError(f"Could not read worksheet '{sheet}'.") from exc
        if skipped:
            warnings.append(f"Skipped {skipped} title row(s) above the header.")
        if len(sheet_names) > 1:
            warnings.append(f"Workbook has {len(sheet_names)} sheets; showing '{sheet}'.")

    df = _finalize(df, warnings)
    log.info("Loaded %s: %d rows x %d cols", ftype, len(df), df.shape[1])
    return LoadedDataset(
        df=df, filename=safe_name, file_type=ftype,
        fingerprint=fingerprint_bytes(data, sheet), sheet_names=sheet_names, sheet=sheet,
        encoding=enc, delimiter=delim, skipped_header_lines=skipped,
        warnings=warnings, size_bytes=len(data),
    )


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "LoadedDataset",
    "detect_file_type",
    "fingerprint_bytes",
    "list_excel_sheets",
    "load_dataset",
]
