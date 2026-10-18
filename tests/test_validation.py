import pytest

from core.exceptions import DataValidationError, FileTooLargeError, UnsupportedFileFormatError
from core.validation import sanitize_filename, validate_extension, validate_size


def test_extension_ok_case_insensitive():
    assert validate_extension("Data.CSV") == ".csv"
    assert validate_extension("a.xlsx") == ".xlsx"
    for ext in (".tsv", ".txt", ".xls", ".xlsm"):
        assert validate_extension("a" + ext) == ext


@pytest.mark.parametrize("name", ["a.exe", "a.json", "noext", "a.csv.exe", "a.pdf"])
def test_extension_rejected(name):
    with pytest.raises(UnsupportedFileFormatError):
        validate_extension(name)


def test_size_limits():
    validate_size(10, 100)
    with pytest.raises(DataValidationError):
        validate_size(0, 100)
    with pytest.raises(FileTooLargeError):
        validate_size(101, 100)


def test_sanitize_blocks_traversal():
    assert sanitize_filename("../../etc/passwd.csv") == "passwd.csv"
    assert sanitize_filename("C:\\x\\y\\sales data.csv") == "sales data.csv"
    with pytest.raises(DataValidationError):
        sanitize_filename("..")
