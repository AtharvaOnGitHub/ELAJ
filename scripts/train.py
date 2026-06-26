"""Entry point for training. Stub — implemented in Phase 11."""
# Usage: python scripts/train.py --config configs/experiment/full_model_no_cce.yaml

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ELAJ foundation model.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument("--root", default=None, help="Project root (default: cwd).")
    args = parser.parse_args()
    raise NotImplementedError("Trainer not yet implemented (Phase 10).")


if __name__ == "__main__":
    main()
