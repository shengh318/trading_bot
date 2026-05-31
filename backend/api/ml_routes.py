"""ML training API — list, retrain, and manage models."""

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

from backend.api.models import (
    CorrelationDataResponse,
    CorrelationPoint,
    MlModelInfo,
    MlRetrainRequest,
    MlRetrainResponse,
)
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
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {e}")
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


# ── Correlation data ──────────────────────────────────────────────────────


def _safe(val: float, default: float = 0.0) -> float:
    """Return *val* if it's a finite number, otherwise *default*."""
    if val is None:
        return default
    try:
        return val if np.isfinite(val) else default
    except (TypeError, ValueError):
        return default


@router.get("/api/correlation/data", response_model=CorrelationDataResponse)
def get_correlation_data(
    symbol_a: str = Query("NVDA", description="First symbol"),
    symbol_b: str = Query("SPY", description="Second symbol"),
    years: int = Query(5, description="Years of history"),
    windows: str = Query("20,60,120", description="Comma-separated rolling windows"),
):
    """Download two symbols and compute rolling correlations + summary statistics."""
    try:
        window_list = [int(w.strip()) for w in windows.split(",") if w.strip()]
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid window value; must be comma-separated integers")
    end = datetime.now()
    start = end - timedelta(days=int(years * 365.25) + 10)

    raw = yf.download([symbol_a, symbol_b], start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty or "Close" not in raw.columns:
        raise HTTPException(status_code=502, detail="No data returned from Yahoo Finance")

    closes = raw["Close"].dropna(how="all")
    if closes.empty:
        raise HTTPException(status_code=502, detail="No close price data available")

    prices = closes.ffill()
    if symbol_a not in prices.columns or symbol_b not in prices.columns:
        raise HTTPException(status_code=404, detail=f"One or both symbols not found in data")

    pa = prices[symbol_a]
    pb = prices[symbol_b]

    if pa.isna().all() or pb.isna().all():
        raise HTTPException(status_code=502, detail="No price data available for one or both symbols")

    returns_a = pa.pct_change(fill_method=None).dropna()
    returns_b = pb.pct_change(fill_method=None).dropna()
    common_idx = returns_a.index.intersection(returns_b.index)
    returns_a = returns_a.loc[common_idx]
    returns_b = returns_b.loc[common_idx]

    correlations: dict[str, list[dict]] = {}
    for w in window_list:
        corr = returns_a.rolling(w).corr(returns_b)
        correlations[str(w)] = [
            {"time": str(idx.date()), "value": round(float(v), 4)}
            for idx, v in corr.items()
            if not (pd.isna(v) or pd.isna(idx))
        ]

    norm_a = pa / pa.dropna().iloc[0] * 100
    norm_b = pb / pb.dropna().iloc[0] * 100
    common_px = norm_a.index.intersection(norm_b.index)
    cumulative_returns = {
        symbol_a: [
            {"time": str(idx.date()), "value": round(float(norm_a.loc[idx]), 4)}
            for idx in common_px
            if not pd.isna(idx) and not pd.isna(norm_a.loc[idx])
        ],
        symbol_b: [
            {"time": str(idx.date()), "value": round(float(norm_b.loc[idx]), 4)}
            for idx in common_px
            if not pd.isna(idx) and not pd.isna(norm_b.loc[idx])
        ],
    }

    # Statistics
    from scipy.stats import pearsonr, spearmanr, kendalltau
    aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
    if len(aligned) < 5:
        raise HTTPException(status_code=422, detail="Not enough data after alignment")

    r_vals = aligned.iloc[:, 0].values
    s_vals = aligned.iloc[:, 1].values
    pr, pp = pearsonr(r_vals, s_vals)
    sr, sp = spearmanr(r_vals, s_vals)
    kt, kp = kendalltau(r_vals, s_vals)

    long_corr = returns_a.rolling(60).corr(returns_b).dropna()
    corr_std = float(long_corr.std()) if len(long_corr) > 0 else 0.0

    split = int(len(aligned) * 0.6)
    in_sample = aligned.iloc[:split]
    out_sample = aligned.iloc[split:]
    oos_drop = 0.0
    if len(in_sample) > 10 and len(out_sample) > 10:
        r_in, _ = pearsonr(in_sample.iloc[:, 0].values, in_sample.iloc[:, 1].values)
        r_out, _ = pearsonr(out_sample.iloc[:, 0].values, out_sample.iloc[:, 1].values)
        if np.isfinite(r_in) and np.isfinite(r_out):
            oos_drop = round(abs(r_in - r_out), 4)

    statistics = {
        "pearson_r": round(_safe(float(pr)), 4),
        "pearson_p": round(_safe(float(pp)), 6),
        "spearman_r": round(_safe(float(sr)), 4),
        "spearman_p": round(_safe(float(sp)), 6),
        "kendall_tau": round(_safe(float(kt)), 4),
        "kendall_p": round(_safe(float(kp)), 6),
        "rolling_corr_std": round(_safe(corr_std), 4),
        "oos_corr_drop": _safe(oos_drop),
    }

    return CorrelationDataResponse(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        correlations=correlations,
        cumulative_returns=cumulative_returns,
        statistics=statistics,
    )
