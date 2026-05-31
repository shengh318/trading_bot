import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_t_start = time.time()


def _log(msg: str) -> None:
    elapsed = time.time() - _t_start
    print(f"[startup  {elapsed:6.2f}s] {msg}", flush=True)


_log("Importing deps")
from backend.api import deps

_log("Importing routes")
from backend.api.routes import router

_log("Importing websocket")
from backend.api.websocket import ws_router

_log("Importing live_routes")
from backend.api.live_routes import live_router

_log("Importing live_websocket")
from backend.api.live_websocket import live_ws_router

_log("Importing alpaca_routes")
from backend.api.alpaca_routes import alpaca_router

_log("Importing ml_routes")
from backend.api.ml_routes import router as ml_router

_log("Importing pairs_routes")
from backend.api.pairs_routes import router as pairs_router

_log("Importing tsmom_routes")
from backend.api.tsmom_routes import router as tsmom_router

_log("Importing config/store")
from backend.config import validate_config
from backend.data.store import Database


_log("All imports complete — building app")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _log("Opening database")
    deps.db = Database()
    _log("App ready — accepting requests")
    yield
    deps.db.close()
    _log("Database closed")


app = FastAPI(title="TraderBot", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_log("Registering routers")
app.include_router(router)
app.include_router(ws_router)
app.include_router(live_router)
app.include_router(live_ws_router)
app.include_router(alpaca_router)
app.include_router(ml_router)
app.include_router(pairs_router)
app.include_router(tsmom_router)

_log(f"Startup complete ({time.time() - _t_start:.2f}s)")


@app.get("/api/health")
def health():
    errors = validate_config()
    return {
        "status": "ok" if not errors else "misconfigured",
        "errors": errors,
    }
