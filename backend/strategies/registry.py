from typing import Any, Optional

from backend.strategies.base import Strategy
from backend.strategies.sma_crossover import SmaCrossover

_REGISTRY: dict[str, dict] = {
    "SmaCrossover": {
        "class": SmaCrossover,
        "description": "Buy when short SMA crosses above long SMA, sell when it crosses below.",
        "params": [
            {"name": "short_window", "type": "int", "default": 20},
            {"name": "long_window", "type": "int", "default": 50},
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
            else:
                resolved_params[pdef["name"]] = raw
    else:
        resolved_params = {p["name"]: p["default"] for p in info["params"]}

    return info["class"](**resolved_params)
