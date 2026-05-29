import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import deps
from backend.api.routes import router
from backend.api.websocket import ws_router
from backend.api.live_routes import live_router
from backend.api.live_websocket import live_ws_router
from backend.api.alpaca_routes import alpaca_router
from backend.api.ml_routes import router as ml_router
from backend.api.pairs_routes import router as pairs_router
from backend.config import validate_config
from backend.data.store import Database


@asynccontextmanager
async def lifespan(app: FastAPI):
    deps.db = Database()
    yield
    deps.db.close()


app = FastAPI(title="TraderBot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(ws_router)
app.include_router(live_router)
app.include_router(live_ws_router)
app.include_router(alpaca_router)
app.include_router(ml_router)
app.include_router(pairs_router)


@app.get("/api/health")
def health():
    errors = validate_config()
    return {
        "status": "ok" if not errors else "misconfigured",
        "errors": errors,
    }
