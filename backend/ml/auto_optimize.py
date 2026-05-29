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

from colorama import init, Fore, Style

init(autoreset=True)


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
    step_label: str = "",
    step_index: int | str = 0,
    total_steps: int | str = 0,
    timeout: int = 7200,
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

    cmd = [sys.executable, "-u", "-m", "backend.ml.train", "--name", name] + cli
    cli_str = " ".join(cli)

    # ── Training banner ──────────────────────────────────────────────
    print()
    print(f"  {Fore.MAGENTA}{'━' * 70}{Style.RESET_ALL}")
    print(f"  {Fore.MAGENTA}Step {step_index}/{total_steps}  :  {Style.BRIGHT}{step_label}{Style.RESET_ALL}")
    print(f"  {Fore.MAGENTA}{'─' * 70}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Command{Style.RESET_ALL}    :  python -m backend.ml.train --name {name} {cli_str}")
    print(f"  {Fore.MAGENTA}{'━' * 70}{Style.RESET_ALL}")
    print()

    # ── Stream subprocess output in real-time ────────────────────────
    start = time.time()
    output_lines: list[str] = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is not None:
        for line in process.stdout:
            print(colorize_output(line), end="", flush=True)
            output_lines.append(line)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        elapsed = time.time() - start
        output = "".join(output_lines)
        print(f"\n  {Fore.RED}Timed out after {format_duration(elapsed)}{Style.RESET_ALL}")
        return None, output
    elapsed = time.time() - start
    output = "".join(output_lines)

    sharpe = parse_sharpe(output)

    # ── Result line ──────────────────────────────────────────────────
    elapsed_str = format_duration(elapsed)
    if sharpe is not None:
        print(f"\n  Result     :  Sharpe {Fore.GREEN}{sharpe:.4f}{Style.RESET_ALL}  |  Elapsed: {Fore.YELLOW}{elapsed_str}{Style.RESET_ALL}")
    else:
        print(f"\n  Result     :  no Sharpe parsed  |  Elapsed: {elapsed_str}")

    return sharpe, output


def print_leaderboard(log: list[tuple[str, float | None, bool]]) -> None:
    """Print a ranked leaderboard of all tested configurations."""
    if not log:
        return

    print(f"\n  {Fore.GREEN}{'═' * 42}{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}{Style.BRIGHT} LEADERBOARD{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}{'═' * 42}{Style.RESET_ALL}")
    print(f"  {'':>4}  {'Enhancement':<38} {'Sharpe':>8}  {'Verdict':<8}")
    print(f"  {Fore.GREEN}{'─' * 64}{Style.RESET_ALL}")

    # Sort by Sharpe descending (None values at bottom)
    ranked = sorted(log, key=lambda x: (x[1] is None, -(x[1] or 0)))

    for rank, (sname, sharpe, kept) in enumerate(ranked, 1):
        sharpe_str = f"{sharpe:.4f}" if sharpe is not None else "  N/A  "
        verdict = f"{Fore.GREEN}KEPT{Style.RESET_ALL}" if kept else f"{Fore.RED}SKIP{Style.RESET_ALL}"
        rank_color = Fore.YELLOW if rank == 1 and sharpe is not None else Fore.WHITE
        prefix = f"{Fore.YELLOW}>{Style.RESET_ALL}" if rank == 1 and sharpe is not None else " "
        sharpe_color = Fore.GREEN if sharpe and sharpe > 0 else Fore.RED if sharpe and sharpe < 0 else Fore.WHITE

        print(f"  {prefix} {rank_color}{rank:>2}{Style.RESET_ALL}.  {sname:<38}  {sharpe_color}{sharpe_str:>8}{Style.RESET_ALL}  {verdict}")
    print(f"  {Fore.GREEN}{'─' * 64}{Style.RESET_ALL}")


def colorize_output(line: str) -> str:
    """Apply ANSI color to common train.py output patterns for readability."""
    s = line.rstrip("\n\r")
    if not s.strip():
        return line

    stripped = s.strip()

    # Errors — bright red
    if stripped.startswith("ERROR:"):
        return f"{Fore.RED}{Style.BRIGHT}{s}{Style.RESET_ALL}\n"

    # Step headers (Step X/5: ...)
    if re.match(r"Step \d/5:", stripped):
        return f"{Fore.CYAN}{Style.BRIGHT}{s}{Style.RESET_ALL}\n"

    # Download progress
    if stripped.startswith("Downloading"):
        return f"{Fore.YELLOW}{Style.BRIGHT}{s}{Style.RESET_ALL}\n"

    # Model training
    if re.match(r"Training \w+", stripped):
        return f"{Fore.MAGENTA}{s}{Style.RESET_ALL}\n"

    # Backtesting
    if stripped.startswith("Backtesting"):
        return f"{Fore.BLUE}{Style.BRIGHT}{s}{Style.RESET_ALL}\n"

    # Walk-forward fold headers
    if re.match(r"--- Fold \d+/\d+ ---", stripped):
        return f"{Fore.YELLOW}{Style.BRIGHT}{s}{Style.RESET_ALL}\n"

    # Grid search iteration
    if re.match(r"\[\d+/\d+\]", stripped) or stripped.startswith("Grid search"):
        return f"{Fore.MAGENTA}{s}{Style.RESET_ALL}\n"

    # Building ensemble / meta-labeler
    if (stripped.startswith("Building StackingEnsemble")
        or stripped.startswith("Building meta-labeler")):
        return f"{Fore.MAGENTA}{Style.BRIGHT}{s}{Style.RESET_ALL}\n"

    # OK / TROPHY messages
    if stripped.startswith(("[OK]", "[TROPHY]")):
        return f"{Fore.GREEN}{Style.BRIGHT}{s}{Style.RESET_ALL}\n"

    # Failure messages
    if stripped.startswith("[X]"):
        return f"{Fore.RED}{Style.BRIGHT}{s}{Style.RESET_ALL}\n"

    # Auto-tune messages
    if stripped.startswith("[>>]"):
        return f"{Fore.YELLOW}{Style.BRIGHT}{s}{Style.RESET_ALL}\n"

    # STRATEGY COMPARISON header
    if "STRATEGY COMPARISON" in s:
        return f"{Fore.WHITE}{Style.BRIGHT}{s}{Style.RESET_ALL}\n"

    # Comparison table strategy rows (label + return% + sharpe)
    m = re.match(r"(\s*\S.*?)([+-]\d+\.\d+%)(\s+\d+\.\d+)", s)
    if m:
        pct = m.group(2)
        pct_color = Fore.GREEN if pct.startswith("+") else Fore.RED
        return f"{m.group(1)}{pct_color}{Style.BRIGHT}{pct}{Style.RESET_ALL}{m.group(3)}\n"

    # Walk-Forward AVERAGE Results
    if "Walk-Forward AVERAGE Results" in s:
        return f"{Fore.WHITE}{Style.BRIGHT}{s}{Style.RESET_ALL}\n"

    # Grid search winner
    if "Grid search winner" in stripped:
        return f"{Fore.GREEN}{Style.BRIGHT}{s}{Style.RESET_ALL}\n"

    # Pruning
    if stripped.startswith("Pruning"):
        return f"{Fore.YELLOW}{s}{Style.RESET_ALL}\n"

    # Retraining on full dataset
    if stripped.startswith("Retraining on full dataset"):
        return f"{Fore.YELLOW}{s}{Style.RESET_ALL}\n"

    # Total samples / feature columns
    if (stripped.startswith("Total samples:")
        or stripped.startswith("Feature columns:")):
        return f"{Fore.CYAN}{s}{Style.RESET_ALL}\n"

    # Return percentage in result lines
    if re.search(r"Return=[+-]?\d+\.?\d*%", stripped):
        return f"{Fore.GREEN}{s}{Style.RESET_ALL}\n"

    # Training accuracy / validation metrics
    if re.match(r"Train acc:", stripped) or re.match(r"Val:", stripped):
        return f"{Fore.CYAN}{s}{Style.RESET_ALL}\n"

    # Feature importance lines
    if re.match(r"\d+\.\s+\w+.*:", stripped):
        return f"{Fore.YELLOW}{s}{Style.RESET_ALL}\n"

    return line


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

    total_steps = len(steps)
    symbol_list = args.symbols.split(",")
    print(f"  {Fore.CYAN}{'━' * 70}{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}{Style.BRIGHT}ML  AUTO-OPTIMIZER  —  Forward Selection{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}{'─' * 70}{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Symbols{Style.RESET_ALL}    :  {'  '.join(Fore.WHITE + s.upper() + Style.RESET_ALL for s in symbol_list)}  ({len(symbol_list)} total)")
    print(f"  {Fore.YELLOW}History{Style.RESET_ALL}    :  {args.years} years  |  Cutoff: 2023-01-01")
    print(f"  {Fore.YELLOW}Steps{Style.RESET_ALL}      :  {total_steps} total")
    print(f"  {Fore.CYAN}{'━' * 70}{Style.RESET_ALL}")

    active_flags: dict[str, str | None] = {}
    best_sharpe: float | None = None
    log: list[tuple[str, float | None, bool]] = []

    for step_idx, (step_name, new_flags) in enumerate(steps, 1):
        candidate = dict(active_flags)
        candidate.update(new_flags)
        safe_name = f"autotune_step{step_idx}"

        sharpe, _ = run_train(
            candidate, base_flags, safe_name,
            step_label=step_name, step_index=step_idx, total_steps=total_steps,
            timeout=args.timeout,
        )

        improved = (
            sharpe is not None
            and (best_sharpe is None or sharpe > best_sharpe + args.improvement_threshold)
        )

        if improved:
            kept = True
            active_flags = candidate
            best_sharpe = sharpe
            print(f"\n  {Fore.GREEN}■ KEPT{Style.RESET_ALL}  {step_name}  →  Sharpe {Fore.GREEN}{sharpe:.4f}{Style.RESET_ALL}  {Fore.MAGENTA}(new best!){Style.RESET_ALL}")
        else:
            kept = False
            if sharpe is not None:
                diff = sharpe - (best_sharpe or 0.0)
                print(f"\n  {Fore.RED}■ SKIP{Style.RESET_ALL}  {step_name}  →  Sharpe {sharpe:.4f}  ({Fore.YELLOW}{diff:+.4f}{Style.RESET_ALL} vs best {best_sharpe:.4f})")
            else:
                print(f"\n  {Fore.RED}■ SKIP{Style.RESET_ALL}  {step_name}  →  no Sharpe parsed")

        log.append((step_name, sharpe, kept))

        # Show leaderboard after each step
        print_leaderboard(log)

    # ── Forward-selection summary ────────────────────────────────────
    print()
    print(f"  {Fore.YELLOW}{'━' * 70}{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}{Style.BRIGHT}FORWARD-SELECTION COMPLETE{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}{'━' * 70}{Style.RESET_ALL}")

    if best_sharpe is not None:
        print(f"\n  {Fore.GREEN}Best Sharpe   :  {Style.BRIGHT}{best_sharpe:.4f}{Style.RESET_ALL}")
    else:
        print(f"\n  Best Sharpe   :  {Fore.RED}(none){Style.RESET_ALL}")

    print(f"\n  {Style.BRIGHT}Optimal flags:{Style.RESET_ALL}")
    if active_flags:
        print(f"    {Fore.YELLOW}{'─' * 50}{Style.RESET_ALL}")
        for k, v in active_flags.items():
            val_str = f"  {v}" if v is not None else ""
            print(f"    {Fore.YELLOW}{k}{Style.RESET_ALL}{val_str}")
        print(f"    {Fore.YELLOW}{'─' * 50}{Style.RESET_ALL}")
    else:
        print(f"    (none — baseline only)")

    # ── Champion training ────────────────────────────────────────────
    if not args.champion:
        print(f"\n  {Fore.YELLOW}--no-champion set; skipping final champion training.{Style.RESET_ALL}")
        print(f"  Use the optimal flags above with --beat-baselines manually.")
        return

    print()
    print(f"  {Fore.MAGENTA}{'━' * 70}{Style.RESET_ALL}")
    print(f"  {Fore.MAGENTA}{Style.BRIGHT}CHAMPION TRAINING{Style.RESET_ALL}")
    print(f"  {Fore.MAGENTA}{'─' * 70}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Building final model{Style.RESET_ALL} with optimal flags + --grid-search")
    print(f"  + --walk-forward 5 for rigorous validation...")
    print(f"  {Fore.MAGENTA}{'━' * 70}{Style.RESET_ALL}")

    champion_flags = dict(active_flags)
    champion_flags["--grid-search"] = None
    champion_flags["--walk-forward"] = "5"

    champion_sharpe, champion_output = run_train(
        champion_flags, base_flags, name="champion_auto",
        step_label="Champion (Final Model)", step_index="*", total_steps="*",
        timeout=args.timeout,
    )

    # ── Clean up intermediate step models ────────────────────────────
    models_dir = Path("backend/ml/models")
    for f in models_dir.glob("autotune_step*.joblib"):
        f.unlink(missing_ok=True)
    for f in models_dir.glob("autotune_step*_metadata.joblib"):
        f.unlink(missing_ok=True)

    # ── Final champion report ────────────────────────────────────────
    print()
    print(f"  {Fore.GREEN}{'━' * 70}{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}{Style.BRIGHT}CHAMPION MODEL SAVED{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}{'━' * 70}{Style.RESET_ALL}")

    champion_path = models_dir / "champion_auto.joblib"
    print(f"\n  Model      :  {Style.BRIGHT}{champion_path.resolve()}{Style.RESET_ALL}")
    if champion_sharpe is not None:
        print(f"  Sharpe     :  {Fore.GREEN}{Style.BRIGHT}{champion_sharpe:.4f}{Style.RESET_ALL}")
    else:
        print(f"  Sharpe     :  {Fore.RED}(unknown){Style.RESET_ALL}")

    print(f"\n  {Style.BRIGHT}Flags used:{Style.RESET_ALL}")
    print(f"    {Fore.GREEN}{'─' * 50}{Style.RESET_ALL}")
    for k, v in champion_flags.items():
        val_str = f"  {v}" if v is not None else ""
        print(f"    {Fore.YELLOW}{k}{Style.RESET_ALL}{val_str}")
    print(f"    {Fore.GREEN}{'─' * 50}{Style.RESET_ALL}")

    print(f"\n  {Style.BRIGHT}Next steps:{Style.RESET_ALL}")
    print(f"    1. Start API:  {Fore.YELLOW}.venv\\Scripts\\uvicorn backend.api.main:app --reload{Style.RESET_ALL}")
    print(f"    2. Open frontend and select 'ML Strategy'")
    print(f"    3. It will auto-load '{champion_path.name}'")
    print()
    print(f"  {Style.BRIGHT}Re-train manually:{Style.RESET_ALL}")
    cli_parts = ["  python -m backend.ml.train", "    --name my_model"]
    for k, v in champion_flags.items():
        if v is None:
            cli_parts.append(f"    {k}")
        else:
            cli_parts.append(f"    {k} {v}")
    print(" \\\n".join(cli_parts))
    print()


if __name__ == "__main__":
    main()
