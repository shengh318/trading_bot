from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import validate_config
from backend.data.store import Database


db: Database | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    db = Database()
    yield
    db.close()


app = FastAPI(title="TraderBot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    errors = validate_config()
    return {
        "status": "ok" if not errors else "misconfigured",
        "errors": errors,
    }
