"""ablation.py — train two ELAJ configs and compare their evaluation metrics.

Trains Model A then Model B sequentially using separate output directories,
then prints a side-by-side comparison table.

Usage:
    python scripts/ablation.py \\
        --configs configs/experiment/full_model.yaml \\
                  configs/experiment/full_model_with_cce.yaml \\
        --split val

The script re-uses train.main() and evaluate.main() under the hood.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.ablation")

_COMPARE_KEYS = ["nb_loss", "pearson_r", "median_kappa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train two ELAJ configs and compare evaluation metrics."
    )
    parser.add_argument(
        "--configs", nargs=2, required=True, metavar="YAML",
        help="Exactly two experiment YAML paths (Model A then Model B).",
    )
    parser.add_argument("--root", default=None, help="Project root (default: cwd).")
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="Dataset split to evaluate on after training (default: val).",
    )
    return parser.parse_args()


def _run_train(config_path: str, root: str) -> Path:
    """Train one model; return the output directory Path."""
    logger.info("=== Training: %s ===", config_path)
    # Re-use train.main() by patching sys.argv
    old_argv = sys.argv[:]
    sys.argv = ["train.py", "--config", config_path, "--root", root]
    try:
        from scripts.train import main as train_main  # noqa: PLC0415
        train_main()
    finally:
        sys.argv = old_argv

    # Resolve output_dir from the experiment YAML
    from endo_model.configs import load_experiment
    exp_cfg, *_ = load_experiment(config_path, root=root)
    return Path(root) / exp_cfg.output_dir


def _run_eval(config_path: str, root: str, out_dir: Path, split: str) -> dict:
    """Evaluate the best checkpoint; return metrics dict."""
    ckpt_dir = out_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    if not ckpts:
        logger.error("No checkpoints found in %s", ckpt_dir)
        return {}

    # Use the last checkpoint (CheckpointManager keeps the best ones)
    best_ckpt = ckpts[-1]
    logger.info("=== Evaluating: %s ===", config_path)
    old_argv = sys.argv[:]
    sys.argv = [
        "evaluate.py",
        "--checkpoint", str(best_ckpt),
        "--config", config_path,
        "--root", root,
        "--split", split,
        "--out", str(out_dir),
    ]
    try:
        from scripts.evaluate import main as eval_main  # noqa: PLC0415
        eval_main()
    finally:
        sys.argv = old_argv

    result_path = out_dir / "eval_results.json"
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)
    return {}


def _print_comparison(cfg_a: str, cfg_b: str, metrics_a: dict, metrics_b: dict) -> None:
    names = [Path(cfg_a).stem, Path(cfg_b).stem]
    keys = _COMPARE_KEYS + [k for k in metrics_a if k not in _COMPARE_KEYS]

    col_w = max(len(names[0]), len(names[1]), 12) + 2
    metric_w = 28

    header = f"{'Metric':<{metric_w}}{'Model A':<{col_w}}{'Model B':<{col_w}}{'Δ (B-A)'}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(f"Ablation comparison")
    print(f"  A: {cfg_a}")
    print(f"  B: {cfg_b}")
    print(sep)
    print(header)
    print(sep)

    for k in keys:
        v_a = metrics_a.get(k)
        v_b = metrics_b.get(k)
        if v_a is None and v_b is None:
            continue
        fa = f"{v_a:.4f}" if isinstance(v_a, float) else str(v_a)
        fb = f"{v_b:.4f}" if isinstance(v_b, float) else str(v_b)
        delta = ""
        if isinstance(v_a, float) and isinstance(v_b, float):
            d = v_b - v_a
            delta = f"{d:+.4f}"
        print(f"{k:<{metric_w}}{fa:<{col_w}}{fb:<{col_w}}{delta}")
    print(sep)


def main() -> None:
    args = parse_args()
    root = args.root or str(Path.cwd())

    cfg_a, cfg_b = args.configs

    out_a = _run_train(cfg_a, root)
    out_b = _run_train(cfg_b, root)

    metrics_a = _run_eval(cfg_a, root, out_a, args.split)
    metrics_b = _run_eval(cfg_b, root, out_b, args.split)

    _print_comparison(cfg_a, cfg_b, metrics_a, metrics_b)


if __name__ == "__main__":
    main()
