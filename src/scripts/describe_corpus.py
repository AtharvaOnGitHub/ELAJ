"""Corpus statistics for the thesis: what actually goes into the atlas.

    python -m scripts.describe_corpus --config configs/experiment/s1_circ_low.yaml

Reads each study's obs and var — never the count matrix — so it runs in minutes.

Writes to <output_dir>/corpus/:
    corpus_summary.csv        one row per study
    cells_and_patients.png    compulsory
    disease_status.png        compulsory
    tissue_site.png           compulsory
    gene_panel_overlap.png    informative
    disease_by_site.png       informative
    cells_per_donor.png       informative

Descriptive only. Counts, no fitted lines, no captions asserting what a
distribution means.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("elaj.corpus")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Experiment YAML.")
    p.add_argument("--root", default=".", help="Project root.")
    p.add_argument(
        "--include-excluded", action="store_true",
        help="Describe studies dropped by exclude_studies too.",
    )
    p.add_argument(
        "--apply-disease-filter", action="store_true",
        help="Count only the cells the model would train on. Off by default: "
             "the thesis chapter describes the corpus, and the filter is a "
             "modelling choice made downstream of it.",
    )
    return p.parse_args()


def read_study(path: str, keep: set[str] | None):
    """One study's obs frame and gene set, without touching X."""
    import anndata

    adata = anndata.read_h5ad(path, backed="r")
    try:
        obs = adata.obs.copy()
        genes = set(map(str, adata.var_names))
    finally:
        if adata.isbacked:
            adata.file.close()

    if keep is not None and "disease_status" in obs.columns:
        obs = obs[obs["disease_status"].astype(str).isin(keep)]
    return obs, genes


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()

    from endo_model.configs import load_experiment
    from endo_model.data.tissue_hierarchy import ONTOLOGY_COLUMN, TissueHierarchy
    from endo_model.viz import apply_theme, save_figure
    from endo_model.viz.corpus_panels import (
        cells_and_patients,
        cells_per_donor,
        composition_by_study,
        disease_by_site,
        gene_panel_overlap,
    )
    from scripts.describe_metadata import resolve_studies

    exp_cfg, _model_cfg, _train_cfg, data_cfg = load_experiment(args.config, root=args.root)
    studies = resolve_studies(data_cfg, args.root, args.include_excluded)
    keep = (
        set(data_cfg.include_disease_statuses)
        if args.apply_disease_filter and data_cfg.include_disease_statuses
        else None
    )
    hierarchy = (
        TissueHierarchy(Path(args.root) / data_cfg.tissue_hierarchy_csv)
        if data_cfg.tissue_hierarchy_csv else None
    )
    labels = (
        {t: e.label for t, e in hierarchy._by_term.items()} if hierarchy else {}
    )

    cells: dict[str, int] = {}
    donors: dict[str, int] = {}
    panels: dict[str, set] = {}
    per_donor: dict[str, list[int]] = {}
    disease: dict[str, dict] = {}
    site: dict[str, dict] = {}
    cross: dict[tuple[str, str], int] = defaultdict(int)
    rows: list[dict] = []

    for study in studies:
        logger.info("Reading %s", study.name)
        obs, genes = read_study(study.path, keep)

        cells[study.name] = len(obs)
        panels[study.name] = genes

        if "patient_id" in obs.columns:
            counts = obs["patient_id"].astype(str).value_counts()
            donors[study.name] = int(counts.size)
            per_donor[study.name] = counts.tolist()
        else:
            donors[study.name] = 0
            per_donor[study.name] = []

        status = (
            obs["disease_status"].astype(str)
            if "disease_status" in obs.columns else None
        )
        disease[study.name] = (
            dict(status.value_counts().items()) if status is not None else {}
        )

        if ONTOLOGY_COLUMN in obs.columns:
            terms = obs[ONTOLOGY_COLUMN].astype(str)
            named = terms.map(lambda t: labels.get(t, t))
            site[study.name] = dict(named.value_counts().items())
            if status is not None:
                for (tissue, stat), n in (
                    named.groupby([named, status], observed=True).size().items()
                ):
                    cross[(str(tissue), str(stat))] += int(n)
        else:
            site[study.name] = {}

        rows.append({
            "study": study.name,
            "cells": cells[study.name],
            "donors": donors[study.name],
            "genes": len(genes),
            "disease_status": "; ".join(
                f"{k}={v}" for k, v in sorted(disease[study.name].items())
            ),
            "sites": "; ".join(
                f"{k}={v}" for k, v in sorted(site[study.name].items())
            ),
        })

    out_dir = Path(args.root) / exp_cfg.output_dir / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "corpus_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "Corpus: %d studies, %s cells, %s donors",
        len(studies), f"{sum(cells.values()):,}", f"{sum(donors.values()):,}",
    )

    apply_theme()
    figures = {
        "cells_and_patients": lambda: cells_and_patients(cells, donors),
        "disease_status": lambda: composition_by_study(disease, "disease_status"),
        "tissue_site": lambda: composition_by_study(site, "anatomical site"),
        "gene_panel_overlap": lambda: gene_panel_overlap(panels),
        "disease_by_site": lambda: disease_by_site(dict(cross)),
        "cells_per_donor": lambda: cells_per_donor(per_donor),
    }
    for name, build in figures.items():
        try:
            fig = build()
        except Exception as exc:  # noqa: BLE001 — one bad panel must not lose the rest
            logger.warning("figure %r failed (%s: %s)", name, type(exc).__name__, exc)
            continue
        save_figure(fig, out_dir / name)
        plt_close(fig)

    logger.info("Written to %s", out_dir)
    return 0


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
