"""Model persistence — save/load trained models with metadata."""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


MODELS_DIR = Path(__file__).parent / "models"


@dataclass
class ModelMetadata:
    feature_columns: list[str]
    model_type: str
    params: dict[str, Any]
    train_date: str = ""
    train_symbols: list[str] = field(default_factory=list)
    train_years: int = 0
    num_train_samples: int = 0
    num_val_samples: int = 0
    validation_metrics: dict[str, float] = field(default_factory=dict)
    baseline_comparison: dict[str, dict[str, float]] = field(default_factory=dict)
    beat_baselines: bool = False


def save_model(
    model: RandomForestClassifier | GradientBoostingClassifier,
    name: str,
    metadata: ModelMetadata,
    models_dir: str | Path = MODELS_DIR,
) -> Path:
    """Save a trained model + metadata to disk.

    Creates two files:
      {name}.joblib         — the scikit-learn model
      {name}_metadata.joblib — ModelMetadata dict
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not metadata.train_date:
        metadata.train_date = datetime.now(timezone.utc).isoformat(timespec="seconds")

    model_path = models_dir / f"{name}.joblib"
    meta_path = models_dir / f"{name}_metadata.joblib"

    joblib.dump(model, model_path)
    joblib.dump(asdict(metadata), meta_path)

    return model_path


def load_model(name: str, models_dir: str | Path = MODELS_DIR) -> tuple[Any, ModelMetadata]:
    """Load a trained model and its metadata from disk.

    Args:
        name: Model name (without extension), e.g. "multi_symbol_model"

    Returns:
        Tuple of (model, ModelMetadata)

    Raises:
        FileNotFoundError: If model files don't exist
    """
    models_dir = Path(models_dir)
    model_path = models_dir / f"{name}.joblib"
    meta_path = models_dir / f"{name}_metadata.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    model = joblib.load(model_path)
    meta_dict = joblib.load(meta_path)
    metadata = ModelMetadata(**meta_dict)

    return model, metadata


def list_models(models_dir: str | Path = MODELS_DIR) -> list[str]:
    """List available trained model names (without extension)."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    names = set()
    for f in models_dir.glob("*.joblib"):
        stem = f.stem
        if stem.endswith("_metadata"):
            names.add(stem.replace("_metadata", ""))
        else:
            names.add(stem)
    return sorted(names)


def delete_model(name: str, models_dir: str | Path = MODELS_DIR) -> None:
    """Delete a model and its metadata from disk."""
    models_dir = Path(models_dir)
    for f in [
        models_dir / f"{name}.joblib",
        models_dir / f"{name}_metadata.joblib",
    ]:
        if f.exists():
            f.unlink()
