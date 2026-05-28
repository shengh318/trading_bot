from typing import Any, Optional

from backend.strategies.base import Strategy
from backend.strategies.ml_strategy import MLStrategy
from backend.strategies.sma_crossover import SmaCrossover
from backend.strategies.simple_strat_1 import SimpleStrat1

_REGISTRY: dict[str, dict] = {
    "ML Strategy": {
        "class": MLStrategy,
        "description": "Pre-trained ML model (RandomForest/GradientBoosting) loaded from disk. Predicts next-bar direction from 24 technical indicators. Use --beat-baselines during training to ensure it outperforms SmaCrossover and SimpleStrat 1.",
        "params": [
            {"name": "model_name", "type": "str", "default": "multi_symbol_model"},
            {"name": "confidence_threshold", "type": "float", "default": 0.55},
            {"name": "base_buy_size", "type": "float", "default": 1000.0},
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
            if pdef["type"] == "int":
                resolved_params[pdef["name"]] = int(raw)
            elif pdef["type"] == "float":
                resolved_params[pdef["name"]] = float(raw)
            else:
                resolved_params[pdef["name"]] = raw
    else:
        resolved_params = {p["name"]: p["default"] for p in info["params"]}

    return info["class"](**resolved_params)
