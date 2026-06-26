"""Entry point for evaluation. Stub — implemented in Phase 12."""
# Usage: python scripts/evaluate.py --checkpoint outputs/run/checkpoint_best.pt

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained ELAJ checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    args = parser.parse_args()
    raise NotImplementedError("Evaluator not yet implemented (Phase 12).")


if __name__ == "__main__":
    main()
