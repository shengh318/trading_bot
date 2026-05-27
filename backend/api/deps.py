from backend.data.store import Database

db: Database | None = None


def get_db() -> Database:
    global db
    if db is None:
        db = Database()
    return db
