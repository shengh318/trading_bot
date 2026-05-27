import pytest

from backend.api import deps
from backend.data.store import Database


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))
    yield db
    db.close()


@pytest.fixture
def api_db(db):
    """Inject the test DB into the app so API routes read/write it."""
    original = deps.db
    deps.db = db
    yield db
    deps.db = original
