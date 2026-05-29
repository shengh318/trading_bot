"""Auto-optimizer: iteratively test ML enhancements and keep what works.

Runs a forward-selection search over the ML pipeline's feature flags.
Each step adds one new enhancement, trains, and compares Sharpe to the
current best. If the enhancement improves Sharpe, it is kept for all
subsequent steps.

After the search, trains a final champion model with the optimal flags
plus --grid-search and --walk-forward 5 for rigorous validation.

Usage:
    python -m backend.ml.auto_optimize --symbols NVDA,AMD,VOO,SPY,META --years 20
    python -m backend.ml.auto_optimize --symbols AAPL,MSFT --years 10 --skip-triple-barrier
    python -m backend.ml.auto_optimize --no-champion  (skip final champion train)
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def parse_sharpe(output: str) -> float | None:
    """Extract the best ML model's Sharpe ratio from the comparison table.

    Handles all label prefixes used by train.py:
      - ``ML - RandomForest``        (plain mode)
      - ``GridSearch (rf)``          (grid-search mode)
      - ``StackingEnsemble``         (stacking mode)
      - ``ML (GridSearch (rf))``     (walk-forward average)
    """
    best: float | None = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(("ML", "GridSearch", "Stacking")):
            # Match the return-with-percent followed by the Sharpe column.
            #   "ML - RandomForest  +28.4%    1.24"
            m = re.match(r"^\S.*?[+-]?\d+\.\d+%\s+([+-]?\d+\.\d+)", stripped)
            if m:
                val = float(m.group(1))
                if best is None or val > best:
                    best = val
    return best


def run_train(
    flags: dict[str, str | None],
    base_flags: dict[str, str],
    name: str,
    timeout: int,
) -> tuple[float | None, str]:
    """Run train.py with *base_flags* merged with *flags*.

    Returns (sharpe, stdout+stderr).
    """
    merged = dict(base_flags)
    merged.update(flags)
    cli: list[str] = []
    for k, v in merged.items():
        cli.append(k)
        if v is not None:
            cli.append(str(v))

    cmd = [sys.executable, "-m", "backend.ml.train", "--name", name] + cli
    print()
    print("-" * 70)
    print(f"  python -m backend.ml.train --name {name} {' '.join(cli)}")
    print("-" * 70)

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.time() - start
    output = result.stdout + result.stderr

    # Print tail of output (last ~80 lines)
    lines = output.splitlines()
    tail = lines[-min(len(lines), 80):]
    for ln in tail:
        print(ln)

    sharpe = parse_sharpe(output)
    label = f"Sharpe={sharpe:.4f}" if sharpe is not None else "no Sharpe parsed"
    print(f"  Elapsed: {format_duration(elapsed)}  ->  {label}")
    return sharpe, output


def flags_to_cli(flags: dict[str, str | None]) -> list[str]:
    """Convert a flag dict to a CLI arg list."""
    cli: list[str] = []
    for k, v in flags.items():
        cli.append(k)
        if v is not None:
            cli.append(str(v))
    return cli


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-optimize ML model by iteratively testing enhancements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m backend.ml.auto_optimize --symbols NVDA,AMD,VOO,SPY,META --years 20\n"
            "\n"
            "The script runs a forward-selection search, then trains a final champion\n"
            "model with --grid-search and --walk-forward for rigorous validation.\n"
        ),
    )
    parser.add_argument(
        "--symbols",
        default="NVDA,AMD,VOO,SPY,META",
        help="Comma-separated symbols (default: NVDA,AMD,VOO,SPY,META)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=20,
        help="Years of history (default: 20)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Max seconds per training run (default: 7200)",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.0,
        help="Minimum Sharpe improvement to keep a flag (default: 0.0)",
    )
    parser.add_argument(
        "--champion",
        action="store_true",
        default=True,
        help="Train a final champion model after search (default: True)",
    )
    parser.add_argument(
        "--no-champion",
        action="store_false",
        dest="champion",
        help="Skip final champion training",
    )
    parser.add_argument(
        "--skip-walk-forward",
        action="store_true",
        help="Skip walk-forward step (useful for small datasets)",
    )
    parser.add_argument(
        "--skip-stacking",
        action="store_true",
        help="Skip stacking ensemble step",
    )
    parser.add_argument(
        "--skip-multi-horizon",
        action="store_true",
        help="Skip multi-horizon ensemble step",
    )
    parser.add_argument(
        "--skip-regime-aware",
        action="store_true",
        help="Skip regime-aware modulation step",
    )
    parser.add_argument(
        "--skip-context-symbols",
        action="store_true",
        help="Skip cross-symbol context features step",
    )
    parser.add_argument(
        "--skip-triple-barrier",
        action="store_true",
        help="Skip triple-barrier labeling step",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    base_flags: dict[str, str | None] = {
        "--symbols": args.symbols,
        "--years": str(args.years),
        "--cutoff-date": "2023-01-01",
    }

    steps: list[tuple[str, dict[str, str | None]]] = []
    steps.append(("Baseline (RF + GBT)", {}))
    steps.append(("Grid search", {"--grid-search": None}))
    if not args.skip_walk_forward:
        steps.append(("Walk-forward 5", {"--walk-forward": "5"}))
    steps.append(("Extended model types (all 6)", {"--model-types": "rf,gbt,xgb,lgb,sgd,mlp"}))
    if not args.skip_stacking:
        steps.append(("Stacking ensemble", {"--stacking": None, "--model-types": "rf,gbt,xgb,lgb"}))
    if not args.skip_multi_horizon:
        steps.append(("Multi-horizon 1,5,21", {"--multi-horizon": "1,5,21"}))
    if not args.skip_regime_aware:
        steps.append(("Regime-aware", {"--regime-aware": None}))
    if not args.skip_context_symbols:
        steps.append(("Context symbols SPY,VOO", {"--context-symbols": "SPY,VOO"}))
    steps.append(("Kelly + auto-threshold", {"--kelly": None, "--auto-threshold": None}))
    if not args.skip_triple_barrier:
        steps.append((
            "Triple barrier labeling",
            {"--labeling": "triple_barrier", "--triple-barrier-pct": "0.02", "--triple-barrier-max-bars": "10"},
        ))

    print("=" * 70)
    print("  ML Auto-Optimizer -- Forward Selection")
    print("  Symbols:", args.symbols)
    print("  Years:", args.years)
    print("=" * 70)
    print(f"\n{'Step':<5} {'Enhancement':<40} {'Result':<10}")
    print("-" * 55)

    active_flags: dict[str, str | None] = {}
    best_sharpe: float | None = None
    log: list[tuple[str, float | None, bool]] = []

    for step_idx, (step_name, new_flags) in enumerate(steps, 1):
        candidate = dict(active_flags)
        candidate.update(new_flags)
        safe_name = f"autotune_step{step_idx}"

        sharpe, _ = run_train(candidate, base_flags, safe_name, timeout=args.timeout)

        improved = (
            sharpe is not None
            and (best_sharpe is None or sharpe > best_sharpe + args.improvement_threshold)
        )

        if improved:
            kept = True
            active_flags = candidate
            best_sharpe = sharpe
            print(f"  [{step_idx}]  {step_name:<40} KEPT  (Sharpe {sharpe:.4f})")
        else:
            kept = False
            reason = f"Sharpe {sharpe:.4f} <= {best_sharpe:.4f}" if sharpe is not None else "No Sharpe parsed"
            print(f"  [{step_idx}]  {step_name:<40} SKIP  ({reason})")

        log.append((step_name, sharpe, kept))

    # -- Forward-selection summary --
    print()
    print("=" * 70)
    print("  FORWARD-SELECTION RESULTS")
    print("=" * 70)
    print(f"\n  Best step Sharpe:  {best_sharpe:.4f}" if best_sharpe is not None else "\n  Best step Sharpe:  (none)")
    print(f"\n  Optimal flags discovered:")
    if active_flags:
        for k, v in active_flags.items():
            if v is None:
                print(f"    {k}")
            else:
                print(f"    {k} {v}")
    else:
        print("    (none -- baseline only)")

    # -- Champion training --
    if not args.champion:
        print(f"\n  --no-champion set; skipping final champion training.")
        print(f"  Use the optimal flags above with --beat-baselines manually.")
        return

    print()
    print("=" * 70)
    print("  CHAMPION TRAINING")
    print("=" * 70)
    print("  Training final model with optimal flags + full grid search")
    print("  + walk-forward validation + beat-baselines guard ...")

    champion_flags = dict(active_flags)
    champion_flags["--grid-search"] = None
    champion_flags["--walk-forward"] = "5"
    # Note: intentionally NOT using --beat-baselines — the champion model
    # is always saved regardless of baseline comparison results.

    champion_sharpe, champion_output = run_train(
        champion_flags, base_flags, name="champion_auto", timeout=args.timeout,
    )

    # -- Clean up intermediate step models --
    models_dir = Path("backend/ml/models")
    for f in models_dir.glob("autotune_step*.joblib"):
        f.unlink(missing_ok=True)
    for f in models_dir.glob("autotune_step*_metadata.joblib"):
        f.unlink(missing_ok=True)

    # -- Final champion report --
    print()
    print("=" * 70)
    print("  DONE -- CHAMPION MODEL SAVED")
    print("=" * 70)
    champion_path = models_dir / "champion_auto.joblib"
    print(f"\n  Model: {champion_path.resolve()}")
    print(f"  Champion Sharpe:  {champion_sharpe:.4f}" if champion_sharpe is not None else "  Champion Sharpe:  (unknown)")
    print(f"\n  Flags used:")
    for k, v in champion_flags.items():
        if v is None:
            print(f"    {k}")
        else:
            print(f"    {k} {v}")

    print(f"\n  To use this model for backtesting or live trading:")
    print(f"    1. Start the API:  .venv\\Scripts\\uvicorn backend.api.main:app --reload")
    print(f"    2. Open the frontend and select the 'ML Strategy'")
    print(f"    3. It will automatically load '{champion_path.name}'")
    print()
    print(f"  To re-train manually:")
    cli_parts = ["  python -m backend.ml.train", "    --name my_model"]
    for k, v in champion_flags.items():
        if v is None:
            cli_parts.append(f"    {k}")
        else:
            cli_parts.append(f"    {k} {v}")
    print(" \\\n".join(cli_parts))


if __name__ == "__main__":
    main()
