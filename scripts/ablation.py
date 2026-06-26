"""Entry point for running ablation pairs. Stub — implemented in Phase 11."""
# Usage: python scripts/ablation.py \
#     --configs configs/experiment/full_model_no_cce.yaml \
#               configs/experiment/full_model_with_cce.yaml

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ablation study (two configs).")
    parser.add_argument(
        "--configs", nargs=2, required=True, metavar="YAML",
        help="Exactly two experiment YAML paths (Model A then Model B).",
    )
    args = parser.parse_args()
    raise NotImplementedError("Ablation runner not yet implemented (Phase 11).")


if __name__ == "__main__":
    main()
