from threading import Lock

from backend.data.store import Database

db: Database | None = None
_db_lock = Lock()


def get_db() -> Database:
    global db
    if db is None:
        with _db_lock:
            if db is None:
                db = Database()
    return db
