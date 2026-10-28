"""One tidy row per model, every benchmark axis as a column, for a spreadsheet.

    python -m scripts.export_frontier_table --benchmarks ../Outputs/Benchmarks \
        --out ../Outputs/figures/frontier_table.csv

CPU, seconds. Reads benchmark.csv files and nothing else.

WHY THIS EXISTS. plot_frontier.py draws twenty points with a twenty-entry
legend, which is a figure for a reader who already knows which point they are
looking for. Choosing WHICH points to show is a judgement that changes with
the argument being made, and that judgement belongs to whoever is making it
rather than to a SOURCES list in a script. This writes the numbers out flat so
the choosing can happen somewhere interactive.

WIDE, NOT LONG. One row per model, one column per axis. A scatter chart needs
x and y side by side, and Excel cannot build a scatter from a PivotTable --
PivotCharts support bars, lines and pies and refuse XY. Long format would
force a pivot that then cannot be plotted.

DIRECTION IS IN THE HEADER, not in a convention anyone has to remember: every
column that is better when smaller is suffixed `_lower_better`.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger("export_frontier_table")

#: column name -> (test, label) in benchmark.csv. Order is the column order.
AXES: dict[str, tuple[str, str]] = {
    "xdonor_family":            ("cross-donor", "cell family"),
    "xdonor_phase":             ("cross-donor", "menstrual phase"),
    "xdonor_stage":             ("cross-donor", "Endometriosis_stage"),
    "xdonor_grade":             ("cross-donor", "EndometriosisGrade"),
    "xdonor_family_p":          ("cross-donor-p", "cell family"),
    "xdonor_phase_p":           ("cross-donor-p", "menstrual phase"),
    "xdonor_stage_p":           ("cross-donor-p", "Endometriosis_stage"),
    "disease_xdonor":           ("disease-xdonor", "all studies (pooled)"),
    "disease_xdonor_p":         ("disease-xdonor-p", "all studies (pooled)"),
    "disease_marecko":          ("disease-xdonor", "marecko_2024"),
    "disease_duempelmann":      ("disease-xdonor", "duempelmann_2026"),
    "knn_family":               ("knn-agreement", "cell family"),
    "knn_phase":                ("knn-agreement", "menstrual phase"),
    "knn_finetype":             ("knn-agreement", "AnnotationRefined"),
    "knn_cellcycle":            ("knn-agreement", "cell cycle phase"),
    "donor_knn_lower_better":   ("knn-agreement", "patient_id"),
    "study_mixing_lower_better": ("study-mixing", "all studies"),
    "study_purity_lower_better": ("study-purity", "all studies"),
    "nmi_family":               ("scib-nmi", "cell family"),
    "ari_family":               ("scib-ari", "cell family"),
    "graph_conn_family":        ("scib-graph_connectivity", "cell family"),
    "silhouette_bio_family":    ("scib-silhouette_bio", "cell family"),
}

#: (directory, method prefix, display name, family, val_nb, latent width).
#: val_nb is each arm's own best epoch, read from its metrics.csv -- the epoch
#: the benchmark scored. None where the model has no comparable value.
SOURCES = [
    ("p1_dab168_kl1",          "elaj",   "ELAJ (reported)",  "ELAJ p1",    0.4024, 16),
    ("p1_dab168_kl30",         "elaj",   "p1 dab168/kl30",   "ELAJ p1",    0.4046, 16),
    ("p1_dab7_kl1",            "elaj",   "p1 dab7/kl1",      "ELAJ p1",    0.4016, 16),
    ("p1_dab7_kl30",           "elaj",   "p1 dab7/kl30",     "ELAJ p1",    0.4025, 16),
    ("p1_dab336_kl1",          "elaj",   "p1 dab336/kl1",    "ELAJ p1",    0.3991, 16),
    ("p1_dab336_kl30",         "elaj",   "p1 dab336/kl30",   "ELAJ p1",    0.4023, 16),
    ("a_series/a1_nophase",    "elaj",   "-phase head",      "Ablation",   0.4083, 16),
    ("a_series/a1_noecs",      "elaj",   "-ECS",             "Ablation",   0.4004, 16),
    ("a_series/a1_clsonly",    "elaj",   "-statistics",      "Ablation",   0.3979, 16),
    ("a_series/a1_nodns",      "elaj",   "-sentinel",        "Ablation",   0.4005, 16),
    ("a_series/a1_randgroups", "elaj",   "random groups",    "Ablation",   0.3911, 16),
    ("m5_study_v2",            "elaj",   "m5",               "ELAJ prior", 0.4043, 16),
    ("ind_m6_drain_ep13",      "elaj",   "m6 ep13",          "ELAJ prior", 0.4130, 16),
    ("ind_m6_drain",           "elaj",   "m6 ep16",          "ELAJ prior", 0.4130, 16),
    ("ind_m6_drain_hard",      "elaj",   "m6 hard",          "ELAJ prior", 0.4091, 16),
    ("axis_m6_drain",          "pca",    "PCA",              "Baseline",   None,   16),
    ("axis_m6_drain",          "scvi",   "scVI",             "Baseline",   None,   16),
    ("axis_m6_drain",          "scanvi", "scANVI",           "Baseline",   None,   16),
]

FOUNDATION = [
    (["fm_geneformer", "fm_all", "fm_baseline+Geneformer"], "geneformer", "Geneformer", 256),
    (["fm_scgpt", "fm_all"],  "scgpt",        "scGPT",        512),
    (["fm_scimilarity"],      "scimilarity",  "SCimilarity",  128),
    (["fm_scfoundation"],     "scfoundation", "scFoundation", None),
]


def read_axes(path: Path, prefix: str) -> dict[str, float]:
    by_row = {}
    for row in csv.DictReader(path.open()):
        if row["method"].startswith(prefix):
            by_row[(row["test"], row["label"])] = row["value"]
    out = {}
    for column, key in AXES.items():
        raw = by_row.get(key)
        if raw not in (None, ""):
            try:
                out[column] = float(raw)
            except ValueError:
                pass
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    benchmarks, out = Path(args.benchmarks), Path(args.out)
    rows = []

    def add(folder, prefix, name, family, val_nb, width):
        path = benchmarks / folder / "benchmark.csv"
        if not path.is_file():
            logger.warning("skipping %-18s (no %s)", name, path); return
        axes = read_axes(path, prefix)
        if not axes:
            logger.warning("skipping %-18s (no %r rows)", name, prefix); return
        rows.append({"model": name, "family": family, "width": width,
                     "val_nb_lower_better": val_nb, **axes})
        logger.info("%-18s %-26s %d axes", name, folder, len(axes))

    for folder, prefix, name, family, val_nb, width in SOURCES:
        add(folder, prefix, name, family, val_nb, width)
    for folders, prefix, name, width in FOUNDATION:
        for folder in folders:
            if (benchmarks / folder / "benchmark.csv").is_file():
                add(folder, prefix, name, "Foundation", None, width)
                break

    columns = ["model", "family", "width", "val_nb_lower_better"] + list(AXES)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("\n%d models x %d columns -> %s", len(rows), len(columns), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
