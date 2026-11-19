"""Ownership is enforced in the data layer: another user's dataset or report id behaves like an id that does not exist."""
from __future__ import annotations

import uuid

import pytest

from auth import service
from auth.db import data_dir
from storage import repository as repo
from storage.repository import NotFound

PW = "Valid-pass-2024"


def _user():
    return service.register("Owner Person", f"o{uuid.uuid4().hex[:10]}@example.com", PW, PW)


def _dataset(user, data=b"a,b\n1,2\n", fp=None):
    return repo.save_dataset(user.id, data, "orders.csv", None, fp or uuid.uuid4().hex, 1, 2)


def test_users_see_only_their_own_datasets():
    alice, bob = _user(), _user()
    rec = _dataset(alice)
    assert [r.id for r in repo.list_datasets(alice.id)] == [rec.id]
    assert repo.list_datasets(bob.id) == []
    with pytest.raises(NotFound):
        repo.get_dataset(bob.id, rec.id)
    with pytest.raises(NotFound):
        repo.read_dataset(bob.id, rec.id)
    with pytest.raises(NotFound):
        repo.delete_dataset(bob.id, rec.id)
    assert repo.read_dataset(alice.id, rec.id)[1] == b"a,b\n1,2\n"      # untouched by Bob's attempts


def test_same_file_is_stored_once_per_user_but_separately_for_each_user():
    alice, bob = _user(), _user()
    a1, a2 = _dataset(alice, fp="same"), _dataset(alice, fp="same")
    b1 = _dataset(bob, fp="same")
    assert a1.id == a2.id and b1.id != a1.id
    assert len(repo.list_datasets(alice.id)) == len(repo.list_datasets(bob.id)) == 1
    assert repo.find_dataset(bob.id, "same").id == b1.id


def test_files_live_under_the_owner_folder_with_generated_names():
    alice = _user()
    rec = repo.save_dataset(alice.id, b"x", "../../etc/passwd", None, uuid.uuid4().hex, 1, 1)
    files = list((data_dir() / "users" / str(alice.id) / "datasets").iterdir())
    assert [f.name for f in files] == [f"{rec.id}.bin"]
    assert repo.get_dataset(alice.id, rec.id).filename == "../../etc/passwd"   # kept as a label only


def test_delete_removes_record_and_file():
    alice = _user()
    rec = _dataset(alice)
    path = data_dir() / "users" / str(alice.id) / "datasets" / f"{rec.id}.bin"
    assert path.exists()
    repo.delete_dataset(alice.id, rec.id)
    assert not path.exists() and repo.list_datasets(alice.id) == []


def test_reports_are_private_and_cannot_reference_another_users_dataset():
    alice, bob = _user(), _user()
    ds = _dataset(alice)
    rep = repo.save_report(alice.id, ds.id, "orders.csv", "Q3 review", "pdf", b"%PDF-1.4")
    assert [r.id for r in repo.list_reports(alice.id)] == [rep.id] and repo.list_reports(bob.id) == []
    with pytest.raises(NotFound):
        repo.read_report(bob.id, rep.id)
    with pytest.raises(NotFound):
        repo.delete_report(bob.id, rep.id)
    with pytest.raises(NotFound):
        repo.save_report(bob.id, ds.id, "orders.csv", "stolen", "pdf", b"x")
    assert repo.read_report(alice.id, rep.id)[1] == b"%PDF-1.4"


def test_deleting_a_dataset_keeps_its_reports():
    alice = _user()
    ds = _dataset(alice)
    rep = repo.save_report(alice.id, ds.id, "orders.csv", "Kept", "html", b"<html></html>")
    repo.delete_dataset(alice.id, ds.id)
    rec, data = repo.read_report(alice.id, rep.id)
    assert rec.dataset_id is None and data == b"<html></html>"


@pytest.mark.parametrize("bad", [0, -1, None, "1", True])
def test_a_user_id_is_required(bad):
    with pytest.raises(PermissionError):
        repo.list_datasets(bad)


def test_deleting_an_account_removes_its_records():
    from auth.db import connect
    alice = _user()
    _dataset(alice)
    with connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (alice.id,))
    with connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM datasets WHERE user_id = ?", (alice.id,)).fetchone()[0] == 0
