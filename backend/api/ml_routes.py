"""ML training API — list, retrain, and manage models."""

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.api.models import MlModelInfo, MlRetrainRequest, MlRetrainResponse
from backend.ml.model import delete_model, list_models, load_model

router = APIRouter()

PID_FILE = Path(__file__).parent.parent / "ml" / "logs" / "active_pids.json"


def _load_pids() -> dict[str, int]:
    if PID_FILE.exists():
        try:
            data = json.loads(PID_FILE.read_text())
            return {k: int(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save_pids(pids: dict[str, int]) -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(json.dumps(pids))


_active_pids: dict[str, int] = _load_pids()


def _build_cmd(req: MlRetrainRequest) -> list[str]:
    cmd = [sys.executable, "-m", "backend.ml.train"]
    cmd += ["--symbols", req.symbols]
    cmd += ["--years", str(req.years)]
    cmd += ["--name", req.name]
    cmd += ["--model-types", req.model_types]
    if req.beat_baselines:
        cmd.append("--beat-baselines")
    if req.grid_search:
        cmd.append("--grid-search")
    if req.walk_forward:
        cmd += ["--walk-forward", str(req.walk_forward)]
    if req.stacking:
        cmd.append("--stacking")
    if req.meta_labeling:
        cmd.append("--meta-labeling")
    if req.regularize:
        cmd.append("--regularize")
    if req.prune > 0:
        cmd += ["--prune", str(req.prune)]
    if req.kelly:
        cmd.append("--kelly")
    if req.auto_threshold:
        cmd.append("--auto-threshold")
    if req.labeling != "next_bar":
        cmd += ["--labeling", req.labeling]
    if req.forecast_horizon != 1:
        cmd += ["--forecast-horizon", str(req.forecast_horizon)]
    if req.context_symbols:
        cmd += ["--context-symbols", req.context_symbols]
    if req.multi_horizon:
        cmd += ["--multi-horizon", req.multi_horizon]
    if req.regime_aware:
        cmd.append("--regime-aware")
    if req.embargo != 5:
        cmd += ["--embargo", str(req.embargo)]
    if req.cutoff_date:
        cmd += ["--cutoff-date", req.cutoff_date]
    if req.val_split != 0.8:
        cmd += ["--val-split", str(req.val_split)]
    if req.triple_barrier_pct != 0.02:
        cmd += ["--triple-barrier-pct", str(req.triple_barrier_pct)]
    if req.triple_barrier_max_bars != 10:
        cmd += ["--triple-barrier-max-bars", str(req.triple_barrier_max_bars)]
    if req.n_estimators != 200:
        cmd += ["--n-estimators", str(req.n_estimators)]
    if req.max_depth != 10:
        cmd += ["--max-depth", str(req.max_depth)]
    if req.learning_rate != 0.1:
        cmd += ["--learning-rate", str(req.learning_rate)]
    if req.confidence_threshold != 0.55:
        cmd += ["--confidence-threshold", str(req.confidence_threshold)]
    if req.base_buy_size != 1000.0:
        cmd += ["--base-buy-size", str(req.base_buy_size)]
    if req.model_dir:
        cmd += ["--model-dir", req.model_dir]
    return cmd


@router.get("/api/ml/models", response_model=list[MlModelInfo])
def get_ml_models():
    """List all trained models with version info."""
    return list_models()


@router.get("/api/ml/models/{name}", response_model=MlModelInfo)
def get_ml_model(name: str):
    """Get info for the latest version of a model."""
    models = list_models()
    for m in models:
        if m["name"] == name:
            return m
    raise HTTPException(status_code=404, detail=f"Model '{name}' not found")


@router.post("/api/ml/retrain", response_model=MlRetrainResponse)
def retrain_model(req: MlRetrainRequest):
    """Start training in a background process. Poll /api/ml/models for status."""
    name = req.name or f"model_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    req.name = name

    cmd = _build_cmd(req)
    log_dir = Path(__file__).parent.parent / "ml" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{name}.log"
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    _active_pids[name] = proc.pid
    _save_pids(_active_pids)
    return MlRetrainResponse(
        status="started",
        pid=proc.pid,
        message=f"Training '{name}' started (PID {proc.pid})",
    )


@router.get("/api/ml/retrain/status/{name}", response_model=MlRetrainResponse)
def retrain_status(name: str):
    """Check if training is still running for a given model name."""
    pid = _active_pids.get(name)
    if pid is None:
        models = list_models()
        for m in models:
            if m["name"] == name:
                return MlRetrainResponse(status="completed", message=f"Model '{name}' trained")
        return MlRetrainResponse(status="unknown", message=f"No training found for '{name}'")

    try:
        os.kill(pid, 0)
        return MlRetrainResponse(status="running", pid=pid, message=f"Training '{name}' (PID {pid})")
    except OSError:
        _active_pids.pop(name, None)
        _save_pids(_active_pids)
        return MlRetrainResponse(status="completed", message=f"Training '{name}' finished")


@router.delete("/api/ml/models/{name}")
def delete_ml_model(name: str, version: int | None = None):
    """Delete a model, optionally a specific version."""
    delete_model(name, version=version)
    return {"status": "deleted", "name": name, "version": version}
