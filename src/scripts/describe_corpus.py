"""Corpus statistics for the thesis: what actually goes into the atlas.

    python -m scripts.describe_corpus --config configs/experiment/s1_circ_low.yaml

Sixteen panels plus a per-study CSV, in <output_dir>/corpus/.

    Compulsory              cells_and_patients, disease_status, tissue_site
    Structure and confound  gene_panel_overlap, disease_by_site,
                            study_site_matrix, within_site_contrasts,
                            corpus_concentration
    Sample size             cells_per_donor, donors_by_status,
                            sampling_hierarchy
    Coverage                vocabulary_coverage, panel_size_vs_coverage
    Sequencing QC           library_size, genes_per_cell, detection_rate

Two panels are drawn only when they would say something. samples_per_donor is
skipped when every donor gave one sample, which is the case here.
age_distribution and metadata_completeness are gone: no study in this corpus
records donor age, so the first was twelve rows of "no age recorded" and the
second was one 0% cell beside thirty 100% ones. Both facts are in the log and
the CSV.

Everything but the last three reads obs and var only and takes minutes. The QC
three read X in row chunks — the slow part, roughly 20 GB of sparse data across
the corpus — and `--skip-expression` turns them off.

Descriptive only. Counts, no fitted lines, no captions asserting what a
distribution means. The reader is being shown what went into the atlas so they
can judge it; a caption telling them what to conclude would be doing that
judging for them.
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
        "--skip-expression", action="store_true",
        help="Skip the three panels computed from the count matrix (depth, "
             "genes per cell, detection rate). Those read X in chunks and are "
             "the only slow part; everything else is obs-only and takes minutes.",
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


QC_DEPTH_COLUMNS = ("total_counts", "n_counts", "nCount_RNA")
QC_GENE_COLUMNS = ("n_genes_by_counts", "n_genes", "nFeature_RNA")


def _first_present(obs, names):
    """The first of `names` that exists in obs, or None."""
    for name in names:
        if name in obs.columns:
            return name
    return None


def expression_stats(path: str, obs_index, chunk: int = 20_000):
    """Per-cell depth, per-cell gene count, and per-gene detection rate.

    Uses whatever the source pipeline already recorded (total_counts, n_counts,
    nCount_RNA and their gene-count equivalents) and only reads X for what is
    missing. Several of these studies came through Seurat and carry them; those
    cost nothing.

    Reads X in row chunks. The corpus is roughly 20 GB of sparse data and
    holding a study dense is what gets a login-node job killed, so this never
    materialises more than `chunk` rows at once.

    Returns:
        (depth, genes, detection) — the first two per cell in obs order, the
        third per gene as a fraction of cells.
    """
    import anndata
    import numpy as np
    import scipy.sparse as sp

    adata = anndata.read_h5ad(path, backed="r")
    try:
        n_cells, n_genes = adata.shape
        detected = np.zeros(n_genes, dtype=np.int64)
        depth = np.zeros(n_cells, dtype=np.float64)
        genes = np.zeros(n_cells, dtype=np.float64)

        for start in range(0, n_cells, chunk):
            stop = min(start + chunk, n_cells)
            block = adata.X[start:stop]
            if not sp.issparse(block):
                block = sp.csr_matrix(block)
            block = block.tocsr()
            block.data = np.abs(block.data)          # DNS sentinel is -1
            depth[start:stop] = np.asarray(block.sum(axis=1)).ravel()
            counted = block.copy()
            counted.data = np.ones_like(counted.data)
            genes[start:stop] = np.asarray(counted.sum(axis=1)).ravel()
            detected += np.asarray(counted.sum(axis=0)).ravel().astype(np.int64)
    finally:
        if adata.isbacked:
            adata.file.close()

    keep = obs_index if obs_index is not None else slice(None)
    return depth[keep], genes[keep], detected / max(n_cells, 1)


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
        corpus_concentration,
        detection_rate,
        disease_by_site,
        donors_by_status,
        gene_panel_overlap,
        genes_per_cell,
        library_size,
        panel_size_vs_coverage,
        samples_per_donor,
        sampling_hierarchy,
        study_site_matrix,
        within_site_contrasts,
        vocabulary_coverage,
    )
    from endo_model.data.vocabulary import GeneVocabulary
    import numpy as np
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
    samples: dict[str, int] = {}
    ages: dict[str, list[float]] = {}
    donor_status: dict[str, dict] = {}
    completeness: dict[str, dict] = {}
    measured: dict[str, int] = {}
    depth_by_study: dict[str, list] = {}
    genes_by_study: dict[str, list] = {}
    detection_by_study: dict[str, list] = {}
    samples_per_donor_counts: dict[str, list[int]] = {}
    triple: dict[tuple[str, str, str], int] = defaultdict(int)

    vocab = GeneVocabulary(str(Path(args.root) / data_cfg.vocabulary_path))
    # GeneVocabulary exposes membership, not a name list — build the set once
    # rather than calling __contains__ per gene per study.
    vocab_genes = set(vocab._gene_to_entry)

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
                    triple[(study.name, str(tissue), str(stat))] += int(n)
        else:
            site[study.name] = {}

        # Samples sit between donor and cell: one donor can give several.
        samples[study.name] = (
            int(obs["source_sample_id"].astype(str).nunique())
            if "source_sample_id" in obs.columns else 0
        )

        # Samples per donor: the middle level of the nesting. sampling_hierarchy
        # shows the three totals; this shows whether a study's samples come from
        # many donors once each or a few donors repeatedly.
        if "source_sample_id" in obs.columns and "patient_id" in obs.columns:
            samples_per_donor_counts[study.name] = (
                obs.groupby(obs["patient_id"].astype(str), observed=True)[
                    "source_sample_id"
                ].nunique().tolist()
            )
        else:
            samples_per_donor_counts[study.name] = []

        # Age per DONOR, not per cell — a 200k-cell study from six donors would
        # otherwise contribute six values 200k times to an age histogram.
        if "age" in obs.columns and "patient_id" in obs.columns:
            import pandas as pd

            per = (
                pd.to_numeric(obs["age"], errors="coerce")
                .groupby(obs["patient_id"].astype(str)).first()
            )
            ages[study.name] = [v for v in per.tolist() if v == v]
        else:
            ages[study.name] = []

        # Donors per status. Cells from one donor are not independent
        # observations of a disease; this is the denominator that is.
        if status is not None and "patient_id" in obs.columns:
            donor_status[study.name] = {
                str(k): int(v)
                for k, v in obs.groupby(status, observed=True)["patient_id"]
                .nunique().items()
            }
        else:
            donor_status[study.name] = {}

        completeness[study.name] = {
            field: float(
                (obs[field].notna() & (obs[field].astype(str) != "nan")).mean()
            ) if field in obs.columns and len(obs) else 0.0
            for field in (
                "patient_id", "source_sample_id", "disease_status",
                "tissue_ontology_term_id", "disease_ontology_term_id", "age",
            )
        }

        measured[study.name] = len(genes & vocab_genes) if vocab_genes else len(genes)

        if not args.skip_expression:
            logger.info("  reading counts for %s ...", study.name)
            try:
                d, g, det = expression_stats(study.path, None)
                depth_by_study[study.name] = d.tolist()
                genes_by_study[study.name] = g.tolist()
                detection_by_study[study.name] = det.tolist()
            except Exception as exc:  # noqa: BLE001 — one study must not lose the set
                logger.warning(
                    "  count matrix unreadable for %s (%s: %s) — QC panels will "
                    "omit it", study.name, type(exc).__name__, exc,
                )

        rows.append({
            "study": study.name,
            "cells": cells[study.name],
            "donors": donors[study.name],
            "genes": len(genes),
            "genes_in_vocabulary": measured[study.name],
            "samples": samples[study.name],
            "donors_with_age": len(ages[study.name]),
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

    recorded_age = {k: len(v) for k, v in ages.items() if v}
    if not recorded_age:
        logger.warning(
            "No study records donor age — %d studies, 0 ages. age_distribution "
            "and metadata_completeness are not drawn: one would be an empty "
            "panel and the other a single 0%% cell beside five 100%% ones.",
            len(ages),
        )

    apply_theme()

    # Fix the study->colour mapping for the whole run, largest study first, so
    # a study is the same colour in every panel and the deepest berry tones go
    # to the biggest contributors. study_key then names them; the panels carry
    # the colour.
    from endo_model.viz.corpus_panels import set_study_order, study_key
    set_study_order(sorted(cells, key=lambda s: -cells[s]))

    figures = {
        "study_key": lambda: study_key(),
        # Compulsory
        "cells_and_patients": lambda: cells_and_patients(cells, donors),
        "disease_status": lambda: composition_by_study(disease, "disease_status"),
        "tissue_site": lambda: composition_by_study(site, "anatomical site"),
        # Structure and confounds
        "gene_panel_overlap": lambda: gene_panel_overlap(panels),
        "disease_by_site": lambda: disease_by_site(dict(cross)),
        "study_site_matrix": lambda: study_site_matrix(site),
        "corpus_concentration": lambda: corpus_concentration(cells, donors),
        # Sample size
        "cells_per_donor": lambda: cells_per_donor(per_donor),
        "donors_by_status": lambda: donors_by_status(donor_status),
        "sampling_hierarchy": lambda: sampling_hierarchy(cells, samples, donors),
        # Coverage and completeness
        "vocabulary_coverage": lambda: vocabulary_coverage(
            measured, len(vocab_genes) or max(measured.values(), default=1)
        ),
        "panel_size_vs_coverage": lambda: panel_size_vs_coverage(
            {k: len(v) for k, v in panels.items()}, measured,
            len(vocab_genes) or max(measured.values(), default=1),
        ),

        "within_site_contrasts": lambda: within_site_contrasts(dict(triple)),
    }
    # source_sample_id is 1:1 with patient_id in every study here, so
    # samples_per_donor would be a solid bar of "1 sample" across the board and
    # sampling_hierarchy's middle bar is already redundant with its top one.
    # Drawn only if some study actually has a donor giving more than one sample.
    if any(c > 1 for counts in samples_per_donor_counts.values() for c in counts):
        figures["samples_per_donor"] = lambda: samples_per_donor(
            samples_per_donor_counts
        )
    else:
        logger.info(
            "Every donor gave exactly one sample in every study — skipping "
            "samples_per_donor, which would be one flat bar."
        )

    if depth_by_study:
        figures["library_size"] = lambda: library_size(depth_by_study)
        figures["genes_per_cell"] = lambda: genes_per_cell(genes_by_study)
        figures["detection_rate"] = lambda: detection_rate(detection_by_study)
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
