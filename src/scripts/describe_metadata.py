"""Tabulate and plot what the corpus carries in its metadata columns.

    python -m scripts.describe_metadata --config configs/experiment/kl_check.yaml

Reads each study's obs frame only — never the count matrix — so it runs in
minutes on a login node and needs no GPU.

Writes to <output_dir>/metadata/:
    metadata_summary.csv        one row per (study, column)
    tissue_coverage.png         usable-value fraction per study and level
    disease_status.png          cells per study by disease_status
    tissue_ontology.png         cells per study by UBERON term
    cells_per_study.png

Descriptive only. No model, no training, no conclusions on the axes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("elaj.metadata")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Experiment YAML.")
    p.add_argument("--root", default=".", help="Project root.")
    p.add_argument(
        "--include-excluded", action="store_true",
        help="Describe studies dropped by exclude_studies too. Useful when "
             "deciding whether an exclusion was the right call.",
    )
    return p.parse_args()


def load_obs(path: str, study: str):
    """Read one study's obs frame without touching X.

    anndata's backed mode maps the file rather than reading it, so a 300k-cell
    study costs the same as a 10k-cell one here.
    """
    import anndata

    adata = anndata.read_h5ad(path, backed="r")
    try:
        return adata.obs.copy()
    finally:
        if adata.isbacked:
            adata.file.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()

    from endo_model.configs import load_experiment
    from endo_model.evaluation.metadata_summary import MetadataSummary
    from endo_model.viz import apply_theme, save_figure
    from endo_model.viz.metadata_panels import (
        category_by_study,
        cells_per_study,
        tissue_coverage_heatmap,
    )

    exp_cfg, _, _, data_cfg = load_experiment(args.config, root=args.root)
    studies = (
        data_cfg.resolve_studies(args.root)
        if not args.include_excluded
        else __import__(
            "endo_model.configs.data_config", fromlist=["load_studies_csv"]
        ).load_studies_csv(str(Path(args.root) / data_cfg.studies_csv))
    )

    summary = MetadataSummary()
    for study in studies:
        name = study["name"] if isinstance(study, dict) else study.name
        path = study["path"] if isinstance(study, dict) else study.path
        logger.info("Reading obs for %s", name)
        summary.add_study(name, load_obs(path, name))

    out_dir = Path(args.root) / exp_cfg.output_dir / "metadata"
    summary.to_csv(out_dir / "metadata_summary.csv")

    for problem in summary.describe_bounds():
        logger.warning("OUT OF BOUNDS: %s", problem)

    apply_theme()
    figures = {
        "tissue_coverage": lambda: tissue_coverage_heatmap(
            summary.tissue_level_coverage()
        ),
        "cells_per_study": lambda: cells_per_study(
            summary.value_counts("disease_status")
        ),
        "disease_status": lambda: category_by_study(
            summary.value_counts("disease_status"), "disease_status"
        ),
        "tissue_ontology": lambda: category_by_study(
            summary.value_counts("tissue_ontology_term_id"),
            "tissue_ontology_term_id",
        ),
        "disease_ontology": lambda: category_by_study(
            summary.value_counts("disease_ontology_term_id"),
            "disease_ontology_term_id",
        ),
    }
    import matplotlib.pyplot as plt

    for name, build in figures.items():
        try:
            fig = build()
        except Exception as exc:  # noqa: BLE001 — one bad panel must not lose the rest
            logger.warning("figure %r failed (%s: %s)", name, type(exc).__name__, exc)
            continue
        save_figure(fig, out_dir / name)
        plt.close(fig)

    logger.info("Done. Everything written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
