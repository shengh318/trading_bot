"""Model persistence — save/load trained models with metadata and versioning."""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


MODELS_DIR = Path(__file__).parent / "models"

ModelT = Any


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
    context_symbols: list[str] = field(default_factory=list)
    version: int = 1


def _next_version(name: str, models_dir: Path) -> int:
    versions = []
    for f in models_dir.glob(f"{name}_v*.joblib"):
        stem = f.stem
        if "_v" in stem and not stem.endswith("_metadata"):
            try:
                v = int(stem.split("_v")[-1])
                versions.append(v)
            except ValueError:
                pass
    return max(versions) + 1 if versions else 1


def save_model(
    model: ModelT,
    name: str,
    metadata: ModelMetadata,
    models_dir: str | Path = MODELS_DIR,
) -> tuple[Path, int]:
    """Save model + metadata with auto-versioning.

    Creates:
      {name}_v{version}.joblib            — versioned model file
      {name}_v{version}_metadata.joblib   — versioned metadata
      {name}.joblib                       — latest (overwrite)
      {name}_metadata.joblib              — latest metadata (overwrite)

    Returns:
        Tuple of (latest_path, version_number)
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not metadata.train_date:
        metadata.train_date = datetime.now(timezone.utc).isoformat(timespec="seconds")

    version = _next_version(name, models_dir)
    metadata.version = version

    ver_model = models_dir / f"{name}_v{version}.joblib"
    ver_meta = models_dir / f"{name}_v{version}_metadata.joblib"

    joblib.dump(model, ver_model)
    joblib.dump(asdict(metadata), ver_meta)

    latest_model = models_dir / f"{name}.joblib"
    latest_meta = models_dir / f"{name}_metadata.joblib"
    tmp = models_dir / f"{name}.joblib.tmp"
    joblib.dump(model, tmp)
    tmp.replace(latest_model)
    tmp2 = models_dir / f"{name}_metadata.joblib.tmp"
    joblib.dump(asdict(metadata), tmp2)
    tmp2.replace(latest_meta)

    return latest_model, version


def load_model(
    name: str,
    models_dir: str | Path = MODELS_DIR,
    version: int | None = None,
) -> tuple[Any, ModelMetadata]:
    """Load a trained model and its metadata.

    Args:
        name: Model name, e.g. "multi_symbol_model".
        version: Specific version.  ``None`` loads the latest.
    """
    models_dir = Path(models_dir)

    if version is not None:
        model_path = models_dir / f"{name}_v{version}.joblib"
        meta_path = models_dir / f"{name}_v{version}_metadata.joblib"
    else:
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


def list_models(
    models_dir: str | Path = MODELS_DIR,
) -> list[dict]:
    """List available models with version info and metadata.

    Returns a list of dicts, each with keys:
      name, version, model_type, train_date, validation_metrics, beat_baselines
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    seen: dict[str, dict] = {}

    for f in sorted(models_dir.glob("*_metadata.joblib")):
        stem = f.stem
        if not stem.endswith("_metadata"):
            continue

        base_raw = stem[:-9]
        version = 1
        name = base_raw
        if "_v" in base_raw:
            parts = base_raw.rsplit("_v", 1)
            try:
                version = int(parts[1])
                name = parts[0]
            except ValueError:
                continue

        try:
            meta_dict = joblib.load(f)
            m = ModelMetadata(**meta_dict)
            version = m.version
        except Exception:
            continue

        seen.setdefault(name, set()).add(version)

    result = []
    for name, versions in seen.items():
        sorted_v = sorted(versions, reverse=True)
        latest_meta = load_model(name, models_dir, version=sorted_v[0])[1]
        entry = {
            "name": name,
            "version": sorted_v[0],
            "model_type": latest_meta.model_type,
            "train_date": latest_meta.train_date,
            "train_symbols": latest_meta.train_symbols,
            "context_symbols": latest_meta.context_symbols,
            "validation_metrics": latest_meta.validation_metrics,
            "beat_baselines": latest_meta.beat_baselines,
            "versions": sorted_v,
        }
        result.append(entry)

    return sorted(result, key=lambda x: x.get("train_date", ""), reverse=True)


def list_versions(name: str, models_dir: str | Path = MODELS_DIR) -> list[int]:
    """Return sorted list of version numbers for a given model name."""
    models_dir = Path(models_dir)
    versions: set[int] = set()
    for f in models_dir.glob(f"{name}_v*.joblib"):
        stem = f.stem
        if "_v" in stem and not stem.endswith("_metadata"):
            try:
                versions.add(int(stem.split("_v")[-1]))
            except ValueError:
                pass
    return sorted(versions)


def delete_model(
    name: str,
    models_dir: str | Path = MODELS_DIR,
    version: int | None = None,
) -> None:
    """Delete a model (optionally a specific version).

    If ``version`` is ``None``, ALL versions are deleted.
    """
    models_dir = Path(models_dir)
    if version is not None:
        files = [
            models_dir / f"{name}_v{version}.joblib",
            models_dir / f"{name}_v{version}_metadata.joblib",
        ]
    else:
        files = list(models_dir.glob(f"{name}*.joblib"))

    for f in files:
        if f.exists():
            f.unlink()
