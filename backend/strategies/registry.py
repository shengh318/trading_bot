from typing import Any, Optional

from backend.strategies.base import Strategy
from backend.strategies.auto_coint_strategy import AutoCointStrategy
from backend.strategies.corr_coint_strat import CorrCointStrategy
from backend.strategies.ml_strategy import MLStrategy
from backend.strategies.sma_crossover import SmaCrossover
from backend.strategies.simple_strat_1 import SimpleStrat1

_REGISTRY: dict[str, dict] = {
    "ML Strategy": {
        "class": MLStrategy,
        "description": "Pre-trained ML model (RF/GBT/XGBoost/LightGBM) loaded from disk. Predicts next-bar direction from 38 technical indicators. Params auto-loaded from model metadata (confidence threshold, Kelly) — override here if needed.",
        "params": [
            {"name": "model_name", "type": "str", "default": "multi_symbol_model"},
            {"name": "confidence_threshold", "type": "float", "default": 0.55},
            {"name": "base_buy_size", "type": "float", "default": 1000.0},
            {"name": "use_kelly", "type": "bool", "default": False},
            {"name": "max_hold_bars", "type": "int", "default": 30},
            {"name": "trailing_stop_pct", "type": "float", "default": 0.05},
        ],
    },
    "SmaCrossover": {
        "class": SmaCrossover,
        "description": "Buy when short SMA crosses above long SMA, sell when it crosses below.",
        "params": [
            {"name": "short_window", "type": "int", "default": 10},
            {"name": "long_window", "type": "int", "default": 50},
        ],
    },
    "Simple Strat 1": {
        "class": SimpleStrat1,
        "description": "Mean reversion with DCA. Buys fixed $ amount when price drops from the day's open, sells entire position on green days, with stop loss.",
        "params": [
            {"name": "buy_size", "type": "float", "default": 100.0},
            {"name": "entry_drop", "type": "float", "default": 1.0},
            {"name": "profit_target", "type": "float", "default": 20.0},
            {"name": "sell_portion", "type": "float", "default": 100.0},
            {"name": "stop_loss", "type": "float", "default": 3.0},
            {"name": "max_buys", "type": "int", "default": 1},
        ],
    },
    "CorrCointStrategy": {
        "class": CorrCointStrategy,
        "description": "Pairs mean-reversion trading using correlation + cointegration. Enters long/short spread when z-score deviates, exits on mean reversion, partial profit-taking, stop-loss, or time limit. Supports graduated entry, volatility scaling, P&L stop, half-life weighting, hedge ratio drift rebalancing, and cooldown.",
        "params": [
            {"name": "symbol_b", "type": "str", "default": ""},
            {"name": "hedge_ratio", "type": "float", "default": 1.0},
            {"name": "half_life", "type": "float", "default": 20.0},
            {"name": "z_entry_strong", "type": "float", "default": 2.0},
            {"name": "z_entry_weak", "type": "float", "default": 1.5},
            {"name": "z_exit_full", "type": "float", "default": 0.0},
            {"name": "z_exit_partial", "type": "float", "default": 0.5},
            {"name": "z_stop", "type": "float", "default": 2.5},
            {"name": "max_holding_days", "type": "int", "default": 40},
            {"name": "volatility_adjust", "type": "bool", "default": True},
            {"name": "min_corr_gate", "type": "float", "default": 0.5},
            {"name": "max_consecutive_losses", "type": "int", "default": 3},
            {"name": "cooldown_days", "type": "int", "default": 10},
            {"name": "base_pair_capital", "type": "float", "default": 10000.0},
            {"name": "max_loss_pct", "type": "float", "default": 5.0},
            {"name": "hedge_ratio_window", "type": "int", "default": 60},
        ],
    },
    "AutoCointStrategy": {
        "class": AutoCointStrategy,
        "description": "Auto-discovers the best cointegrated pair from a comma-separated list of symbols. Downloads prices, computes Pearson correlations, runs EG cointegration on top candidates, then trades the best pair using z-score mean reversion with stop-loss and time-based exit. No prior pair discovery needed.",
        "params": [
            {"name": "symbols", "type": "str", "default": "NVDA,AMD"},
            {"name": "min_corr", "type": "float", "default": 0.7},
            {"name": "z_entry", "type": "float", "default": 2.0},
            {"name": "z_exit", "type": "float", "default": 0.0},
            {"name": "stop_loss", "type": "float", "default": 3.0},
            {"name": "max_holding_days", "type": "int", "default": 40},
            {"name": "base_pair_capital", "type": "float", "default": 10000.0},
            {"name": "min_trading_days", "type": "int", "default": 504},
        ],
    },
}


def list_strategies() -> list[dict]:
    return [
        {
            "name": name,
            "description": info["description"],
            "params": info["params"],
        }
        for name, info in _REGISTRY.items()
    ]


def get_strategy(name: str, params: Optional[dict[str, Any]] = None) -> Strategy:
    info = _REGISTRY.get(name)
    if info is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(_REGISTRY.keys())}")

    resolved_params = {}
    if params:
        for pdef in info["params"]:
            raw = params.get(pdef["name"], pdef["default"])
            if raw is None:
                resolved_params[pdef["name"]] = pdef["default"]
            elif pdef["type"] == "int":
                try:
                    resolved_params[pdef["name"]] = int(raw)
                except (TypeError, ValueError):
                    resolved_params[pdef["name"]] = pdef["default"]
            elif pdef["type"] == "float":
                try:
                    resolved_params[pdef["name"]] = float(raw)
                except (TypeError, ValueError):
                    resolved_params[pdef["name"]] = pdef["default"]
            elif pdef["type"] == "bool":
                if isinstance(raw, bool):
                    resolved_params[pdef["name"]] = raw
                else:
                    resolved_params[pdef["name"]] = str(raw).lower() in ("true", "1", "yes")
            else:
                resolved_params[pdef["name"]] = raw
    else:
        resolved_params = {p["name"]: p["default"] for p in info["params"]}

    return info["class"](**resolved_params)
