from core.exceptions import BIError, FileTooLargeError


def test_user_message_separate_from_detail():
    e = FileTooLargeError("too big", detail="size=999")
    assert isinstance(e, BIError)
    assert e.user_message == "too big" and e.detail == "size=999"
