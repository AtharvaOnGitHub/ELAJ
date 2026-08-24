"""What the validation label columns actually contain, per study.

    python -m scripts.describe_labels --config configs/experiment/r8_direction.yaml

Reads each study's obs frame only -- never the count matrix -- so it runs in
about a minute on a login node with no GPU.

Answers the question that decides which validation figures can be built: not
whether a label column exists, but what SHAPE its values take. An ordinal stage
(I, II, III, IV) supports a distance figure; a binary yes/no does not; a column
holding one value everywhere supports nothing at all.

Writes label_survey.csv beside the run's other metadata, and prints the same
thing. Descriptive only -- no model, no conclusions.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("elaj.labels")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Experiment YAML.")
    p.add_argument("--root", default=".", help="Project root.")
    p.add_argument("--max-values", type=int, default=8,
                   help="Distinct values listed per study before truncating. "
                        "AnnotationRefined has 61, and printing all of them "
                        "buries the columns that have four.")
    p.add_argument("--out", default=None,
                   help="Where label_survey.csv goes. Defaults to the run's "
                        "output_dir/metadata/.")
    return p.parse_args()


def load_obs(path: str):
    """One study's obs frame, without touching X.

    Backed mode maps the file rather than reading it, so a 300k-cell study
    costs about what a 10k-cell one does. Same helper as describe_metadata.
    """
    import anndata

    adata = anndata.read_h5ad(path, backed="r")
    try:
        return adata.obs.copy()
    finally:
        if adata.isbacked:
            adata.file.close()


def survey_studies(studies, columns) -> list:
    """Every (study, column) pair that holds at least one labelled value."""
    from endo_model.evaluation.label_survey import survey_column

    surveys = []
    for study in studies:
        logger.info("Reading obs for %s", study.name)
        obs = load_obs(study.path)
        for column in columns:
            found = survey_column(
                study.name, column,
                obs[column] if column in obs.columns else None,
            )
            if found is not None:
                surveys.append(found)
    return surveys


def write_csv(surveys, path: Path) -> None:
    """One row per (study, column, value). Long form, so it pivots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["study", "column", "kind", "n_labelled",
                           "n_distinct", "value", "n_cells"],
        )
        writer.writeheader()
        for s in surveys:
            for value, count in s.values:
                writer.writerow({
                    "study": s.study, "column": s.column, "kind": s.kind,
                    "n_labelled": s.n_labelled, "n_distinct": s.n_distinct,
                    "value": value, "n_cells": count,
                })


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()

    from endo_model.configs import load_experiment
    from endo_model.evaluation.label_survey import SURVEY_COLUMNS, format_survey

    exp_cfg, _, _, data_cfg = load_experiment(args.config, root=args.root)
    studies = data_cfg.resolve_studies(args.root)
    logger.info("Surveying %d columns across %d studies",
                len(SURVEY_COLUMNS), len(studies))

    surveys = survey_studies(studies, SURVEY_COLUMNS)
    logger.info("%s", format_survey(surveys, max_values=args.max_values))

    out = (Path(args.out) if args.out
           else Path(args.root) / exp_cfg.output_dir / "metadata")
    write_csv(surveys, out / "label_survey.csv")
    logger.info("Written to %s", out / "label_survey.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
