import pytest

from backend.data.store import Database


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))
    yield db
    db.close()
