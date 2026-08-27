"""Every validation figure, written from one LatentBundle.

One method per figure, each doing its own measurement and calling its own panel
function. They are independent on purpose: a figure that fails takes only
itself down, which matters when a run of two dozen figures is the deliverable
and one label turns out to be missing from the corpus.

Reading order for the figures themselves is in docs/walkthrough/12-validation.md.
The numeric prefixes are the presentation order, not the order they compute in.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from endo_model.evaluation.label_harmonisation import (
    CELL_CYCLE, CELL_CYCLE_HOURS, CELL_CYCLE_PERIOD_HOURS,
    CELL_CYCLE_SOURCE_COLUMNS, LINEAGE, LINEAGE_SOURCE_COLUMNS,
    MENSTRUAL_PHASE, PHASE_SOURCE_COLUMNS, cell_cycle_angle,
)
from endo_model.evaluation.latent_topology import (
    betti_numbers, circle_occupancy, explained_variance, intrinsic_dimension,
    noise_threshold, persistence_diagram,
)
from endo_model.evaluation.latent_validation import (
    circular_linear_corr, label_counts_by_group, label_knn_agreement,
    neighbour_confusion, ordinal_centroid_distance, per_label_agreement,
)
from endo_model.viz import save_figure
from endo_model.viz.latent_panels import safe_umap
from endo_model.viz.torus_panels import (
    circle_occupancy_panel, dimension_panel, kappa_panel, persistence_panel,
    phase_on_circles, torus_surface, torus_tiles,
)
from endo_model.viz.validation_panels import (
    agreement_bars, confusion_heatmap, control_panel, correlation_heatmap,
    cross_donor_bars, donor_label_matrix, label_coverage, ordinal_distance,
    per_label_bars, umap_continuous, umap_cyclic, umap_grid,
)

logger = logging.getLogger(__name__)

# Where each covariate enters the pipeline. This grouping is a fact about the
# model's wiring, not a claim about any result -- which is what lets it go on a
# figure under the no-interpretation rule.
#
# `age` appears in NEITHER list. It reaches MetadataEncoder, so it belongs in
# the first, except that no study in the corpus carries an age column:
# `_age_vector` returns all-NaN, `AgeEncoder` zeroes NaN rows, and the metadata
# token is an exactly zero vector for every cell. There is nothing to measure.
#
# Both of these are DECODER inputs, not encoder inputs, so the block is headed
# "given to the decoder". Conditioning the decoder on them is what permits the
# encoder not to store them -- which makes a high bar here informative rather
# than expected.
GIVEN_TO_MODEL = ("tissue", "platform")
SHOULD_BE_ABSENT = ("study", "patient_id", "phase")   # marecko.phase = G1/S/G2M

QC_CONTROLS = ("pct_counts_mt", "total_counts", "n_genes_by_counts",
               "doublet_score", "scDblFinder.score", "scrublet_score")
BIOLOGY_CONTINUOUS = ("ProgesteroneSerum", "epithelial.pseudotime", "CycleDay")


class ValidationFigureWriter:
    """Draws and saves the validation figures for one embedded run.

    Args:
        bundle: the embedding pass and its metadata.
        out_dir: where the PNGs go. Figures are written HERE and not beside the
            checkpoint, so two runs over different splits do not overwrite
            each other's pictures.
        k: neighbours per cell.
        seed: fixed so every figure and every rerun sees the same sample.
        umap_cells: how many points the UMAP figures draw.
    """

    def __init__(self, bundle, out_dir, k: int = 30, seed: int = 24,
                 umap_cells: int = 150000, cycle_period: float = 28.0,
                 topology_cells: int = 2000) -> None:
        self.bundle = bundle
        self.out_dir = Path(out_dir)
        self.k = k
        self.seed = seed
        self.umap_cells = umap_cells
        self.cycle_period = cycle_period
        self.topology_cells = topology_cells
        self._phase = bundle.harmonised(MENSTRUAL_PHASE, PHASE_SOURCE_COLUMNS)
        self._family = bundle.harmonised(LINEAGE, LINEAGE_SOURCE_COLUMNS)
        self._umap: dict[str, np.ndarray | None] | None = None
        self._umap_rows: np.ndarray | None = None

    # ── orchestration ────────────────────────────────────────────────────────

    def write_all(self) -> list[str]:
        """Every figure, each wrapped so one failure does not stop the rest."""
        figures = {
            "01_umap_grid": self.umap_grid,
            "02_umap_phase": self.umap_phase,
            "03_umap_untested": self.umap_untested,
            "06_umap_continuous": self.umap_continuous,
            "04_agreement_bars": self.agreement_bars,
            "07_kappa": self.kappa,
            "09_correlation": self.correlation,
            "10_label_coverage": self.label_coverage,
            "11_celltype_confusion": self.celltype_confusion,
            "12_celltype_per_label": self.celltype_per_label,
            "16_donor_by_phase": self.donor_by_phase,
            "17_phase_ordering": self.phase_ordering,
            "18_control_panel": self.control_panel,
            "25_cellcycle_on_circles": self.cell_cycle_on_circles,
            "26_cellcycle_umap": self.cell_cycle_umap,
            "27_cellcycle_torus": self.cell_cycle_torus,
            "20_torus_tiles": self.torus_tiles,
            "21_torus_surface": self.torus_surface,
            "22_circle_occupancy": self.circle_occupancy,
            "23_persistence": self.persistence,
            "24_dimension": self.dimension,
        }
        written = []
        for name, method in figures.items():
            try:
                fig = method()
            except Exception as exc:                          # noqa: BLE001
                logger.warning("  %s skipped (%s: %s)", name,
                               type(exc).__name__, exc)
                continue
            if fig is None:
                logger.info("  %s skipped (nothing to draw)", name)
                continue
            self._save(name, fig)
            written.append(name)
        logger.info("%d figures written to %s", len(written), self.out_dir)
        return written

    def _save(self, name: str, fig) -> None:
        from matplotlib import pyplot as plt

        save_figure(fig, self.out_dir / name)
        plt.close(fig)
        logger.info("  %s", name)

    def _n(self) -> str:
        """The cell count, for a figure title. Never a footnote in this corpus."""
        return f"{len(self.bundle):,} cells"

    # ── A: what the latent looks like ────────────────────────────────────────

    def _coords(self) -> tuple[dict, np.ndarray]:
        """UMAP of both halves, computed once and reused by every colouring."""
        if self._umap is None:
            rows = self.bundle.sample(self.umap_cells, self.seed)
            logger.info("  embedding %d cells with UMAP (both halves)",
                        len(rows))
            self._umap = {
                "Gaussian half": safe_umap(self.bundle.mu_gauss[rows],
                                           seed=self.seed),
                "circular half": safe_umap(self.bundle.circular[rows],
                                           seed=self.seed),
            }
            self._umap_rows = rows
        return self._umap, self._umap_rows

    def umap_grid(self):
        coords, rows = self._coords()
        return umap_grid(coords, {
            "study": self.bundle.study_labels()[rows],
            "cell family": self._family[rows] if self._family.size else
                           np.full(len(rows), None, dtype=object),
            "donor": self.bundle.meta["patient_id"][rows],
            "split": self.bundle.split[rows],
        }, f"The latent, coloured four ways — {len(rows):,} of {self._n()}, "
           f"random sample, seed {self.seed}")

    def umap_phase(self):
        if not self._phase.size:
            return None
        coords, rows = self._coords()
        return umap_grid(coords, {"menstrual phase": self._phase[rows]},
                         f"Menstrual phase over the latent — "
                         f"{len(rows):,} of {self._n()}, seed {self.seed}")

    def umap_untested(self):
        """The labels the statistics could NOT settle, drawn anyway.

        Disease grade came back non-significant, disease stage came back
        negative, and cell-cycle phase cannot be tested across donors at all
        because every donor's majority label is G1. A picture cannot rescue
        any of those, and is not meant to -- but "no structure is visible
        either" is worth being able to show rather than assert, and a reader
        who is handed only a p-value will reasonably ask what it looked like.

        Tissue and platform ride along because they are the decoder's inputs:
        they are what visible structure looks like in this latent, which sets
        the scale for reading the others.
        """
        coords, rows = self._coords()
        columns = {
            "EndometriosisGrade": self.bundle.meta.get("EndometriosisGrade"),
            "Endometriosis_stage": self.bundle.meta.get("Endometriosis_stage"),
            "phase": self.bundle.meta.get("phase"),
            "tissue": self.bundle.meta.get("tissue"),
            "platform": self.bundle.meta.get("platform"),
        }
        present = {k: v[rows] for k, v in columns.items()
                   if v is not None and len(v)}
        if not present:
            return None
        return umap_grid(
            coords, present,
            f"Labels the statistics could not settle — {len(rows):,} of "
            f"{self._n()}, random sample, seed {self.seed}")

    def umap_continuous(self):
        """Cycle day and the other continuous covariates, as colour ramps.

        Cycle day is the headline negative: the correlation against every
        circular dimension is 0.01-0.06. What that number cannot show is
        whether the days are laid out in ANY arrangement -- a gradient across
        the map, a ring, or nothing. The ramp shows it.

        The two strongest technical covariates are included on the same axes
        because the circular half tracks gene count (r = 0.32) five times more
        strongly than it tracks cycle day (r = 0.06), and the two pictures
        side by side is that sentence made visible.
        """
        coords, rows = self._coords()
        columns = ["CycleDay", "ProgesteroneSerum", "epithelial.pseudotime",
                   "n_genes_by_counts"]
        present = {}
        for name in columns:
            values = self.bundle.numeric(name)
            if values.size and np.isfinite(values).sum() >= 100:
                present[name] = values[rows]
        if not present:
            return None
        return umap_continuous(
            coords, present,
            f"Continuous covariates over the latent — {len(rows):,} of "
            f"{self._n()}, random sample, seed {self.seed}")

    # ── B: does the latent know the label ────────────────────────────────────

    def _labels_under_test(self) -> dict[str, np.ndarray]:
        return {
            "cell family": self._family,
            "cell type (fine)": self.bundle.meta.get("AnnotationRefined"),
            "menstrual phase": self._phase,
            "disease grade": self.bundle.meta.get("EndometriosisGrade"),
            "disease stage": self.bundle.meta.get("Endometriosis_stage"),
            "cell cycle phase (control)": self.bundle.meta.get("phase"),
            "patient identity": self.bundle.meta.get("patient_id"),
        }

    def agreement_bars(self):
        results = {}
        for name, values in self._labels_under_test().items():
            if values is None or not len(values):
                continue
            per = {b: label_knn_agreement(lat, values, k=self.k, seed=self.seed)
                   for b, lat in self.bundle.branches.items()}
            # The baseline covers a random subset of rows, so its labels are
            # gathered by index rather than sliced.
            if self.bundle.baseline is not None:
                per["raw counts"] = label_knn_agreement(
                    self.bundle.baseline,
                    np.asarray(values)[self.bundle.baseline_rows],
                    k=self.k, seed=self.seed)
            results[name] = per
        return agreement_bars(
            results,
            f"Neighbour agreement by label and latent half — k={self.k}")

    def cross_donor(self, results):
        """Drawn from measurements the caller already made and logged."""
        return cross_donor_bars(
            results, f"Agreement with same-donor neighbours discarded — "
                     f"k={max(self.k, 100)}")

    # ── E: cell type ─────────────────────────────────────────────────────────

    def celltype_confusion(self):
        if not self._family.size:
            return None
        matrix, names, counts = neighbour_confusion(
            self.bundle.mu_gauss, self._family, k=self.k, seed=self.seed)
        return confusion_heatmap(
            matrix, names, counts,
            f"Cell family of a cell against its {self.k} nearest neighbours "
            f"(Gaussian half)")

    def celltype_per_label(self):
        column = (self._family if self._family.size
                  else self.bundle.meta.get("AnnotationRefined"))
        if column is None or not len(column):
            return None
        return per_label_bars(
            per_label_agreement(self.bundle.mu_gauss, column, k=self.k,
                                seed=self.seed),
            "Neighbour agreement per cell family (Gaussian half)")

    # ── F: disease and ordering ──────────────────────────────────────────────

    def donor_by_phase(self):
        if not self._phase.size:
            return None
        donors, names, grid = label_counts_by_group(
            self._phase, self.bundle.meta["patient_id"],
            order=MENSTRUAL_PHASE.order)
        return donor_label_matrix(donors, names, grid,
                                  "Cells per donor and menstrual phase")

    def phase_ordering(self):
        """Phase is ordered AND wraps, so distance should follow the cycle."""
        if not self._phase.size:
            return None
        matrix, names = ordinal_centroid_distance(
            self.bundle.mu_gauss, self._phase, MENSTRUAL_PHASE.order)
        return ordinal_distance(
            matrix, names,
            "Latent distance between menstrual phases, in cycle order "
            "(Gaussian half)")

    # ── G: controls ──────────────────────────────────────────────────────────

    def control_panel(self):
        blocks = {"GIVEN TO THE DECODER": [], "SHOULD BE ABSENT": [],
                  "THE TESTS": []}
        for name in GIVEN_TO_MODEL:
            self._add_control(blocks["GIVEN TO THE DECODER"], name,
                              self.bundle.meta.get(name), "gaussian")
        for name in SHOULD_BE_ABSENT:
            values = (self.bundle.study_labels() if name == "study"
                      else self.bundle.meta.get(name))
            self._add_control(blocks["SHOULD BE ABSENT"], name, values,
                              "gaussian")
        for name, values in (("cell family", self._family),
                             ("menstrual phase", self._phase)):
            for branch in ("gaussian", "circular"):
                self._add_control(blocks["THE TESTS"], name, values, branch)
        return control_panel(
            blocks, f"Every measurement on one axis — k={self.k}, {self._n()}")

    def _add_control(self, block: list, name: str, values, branch: str) -> None:
        if values is None or not len(values):
            return
        got = label_knn_agreement(self.bundle.branches[branch], values,
                                  k=self.k, seed=self.seed)
        if np.isfinite(got.get("agreement", np.nan)):
            block.append({**got, "label": name, "branch": branch})

    def correlation(self):
        """Biology and technical covariates against every latent dimension."""
        rows, names, group_at = [], [], {}
        for heading, columns in (("BIOLOGY", BIOLOGY_CONTINUOUS),
                                 ("TECHNICAL", QC_CONTROLS)):
            marked = False
            for column in columns:
                x = self.bundle.numeric(column)
                ok = np.isfinite(x) if x.size else np.array([], dtype=bool)
                if ok.sum() < 30:
                    continue
                if not marked:
                    group_at[len(rows)] = heading
                    marked = True
                rows.append(self._corr_row(x, ok))
                names.append(f"{column}  (n={int(ok.sum()):,})")
        if not rows:
            return None
        dims = ([f"g{i}" for i in range(self.bundle.mu_gauss.shape[1])]
                + [f"c{i}" for i in range(self.bundle.mu_angle.shape[1])])
        return correlation_heatmap(
            np.array(rows), names, dims, group_at,
            "Covariates against each latent dimension  (|correlation|)")

    def _corr_row(self, x: np.ndarray, ok: np.ndarray) -> list[float]:
        """Pearson against each Gaussian dim, circular-linear against each angle.

        Two different coefficients because the second set of columns are
        angles: a plain Pearson against an angle depends on where the branch
        cut falls, and this project has already shipped that bug once.
        """
        gauss = [abs(np.corrcoef(self.bundle.mu_gauss[ok, d], x[ok])[0, 1])
                 for d in range(self.bundle.mu_gauss.shape[1])]
        circ = [circular_linear_corr(self.bundle.mu_angle[ok, d], x[ok])
                for d in range(self.bundle.mu_angle.shape[1])]
        return [float(v) if np.isfinite(v) else 0.0 for v in gauss + circ]

    def label_coverage(self):
        columns = ["AnnotationMain", "AnnotationRefined", "lineage", "celltype",
                   "MenstrualCyclePhase", "Stage", "CycleDay",
                   "EndometriosisGrade", "Endometriosis_stage", "patient_id"]
        return label_coverage(self.bundle.label_coverage(columns),
                              "Cells carrying each label, by study")

    # ── H: the shape of the circular latent ──────────────────────────────────

    # ── Check A: the cycle this corpus can actually resolve ──────────────────

    def _cell_cycle(self) -> tuple[np.ndarray, np.ndarray]:
        """Harmonised cell-cycle phase, and its angle around the 24 h cycle.

        Separate from menstrual phase for a reason that is easy to get wrong,
        and was got wrong here for two waves: `phase` in marecko holds
        G1/S/G2M, which is CELL cycle. Its menstrual phase is in `Stage`.
        """
        names = self.bundle.harmonised(CELL_CYCLE, CELL_CYCLE_SOURCE_COLUMNS)
        if not names.size:
            return names, np.array([])
        return names, cell_cycle_angle(names)

    def cell_cycle_on_circles(self):
        """Where G1, S and G2M sit on each circular dimension.

        The most direct test available of whether the circular latent has
        learned a cycle. Unlike the menstrual cycle -- which is constant within
        a donor here, so a per-cell encoder has nothing to represent -- the cell
        cycle varies WITHIN a sample and returns to its start.
        """
        names, _ = self._cell_cycle()
        if not names.size:
            return None
        return phase_on_circles(
            self.bundle.mu_angle, names, CELL_CYCLE.order,
            "Cell-cycle phase around each circular dimension")

    def cell_cycle_umap(self):
        names, angle = self._cell_cycle()
        if not angle.size or not np.isfinite(angle).any():
            return None
        coords, rows = self._coords()
        ticks = {2 * np.pi * h / CELL_CYCLE_PERIOD_HOURS: n
                 for n, h in CELL_CYCLE_HOURS.items()}
        return umap_cyclic(
            coords, angle[rows],
            f"Cell-cycle phase over the latent — {len(rows):,} of "
            f"{self._n()}, cyclic colour so G2M adjoins G1",
            tick_labels=ticks)

    def cell_cycle_torus(self):
        """The cell cycle painted onto the doughnut itself.

        If the model has laid this cycle around one of its circles, the colour
        travels round the ring and closes. If it has not, the colours are
        scattered over the surface with no rotation. That is a claim a reader
        can check by looking, which no correlation coefficient allows.
        """
        names, angle = self._cell_cycle()
        if not angle.size or not np.isfinite(angle).any():
            return None
        rows = self._occupancy()
        if len(rows) < 2:
            return None
        best = sorted(range(len(rows)), key=lambda d: -rows[d]["entropy"])[:2]
        from endo_model.viz.theme import cyclic_cmap
        return torus_surface(
            self.bundle.mu_angle, best[0], best[1], seed=self.seed,
            colour_by=angle, colour_name="cell-cycle phase (radians)",
            cmap=cyclic_cmap(), vmin=0.0, vmax=2 * np.pi,
            subtitle="coloured by cell-cycle phase on a cyclic map — a learned "
                     "cycle appears as colour travelling round the ring")

    def torus_tiles(self):
        return torus_tiles(self.bundle.mu_angle)

    def torus_surface(self):
        """The two most-used circles, chosen by occupancy rather than by index.

        Rendering dims 0 and 1 regardless would show two collapsed dimensions
        as a doughnut with four dots on it if the used pair happened to be 2
        and 3.
        """
        rows = self._occupancy()
        if len(rows) < 2:
            return None
        best = sorted(range(len(rows)), key=lambda d: -rows[d]["entropy"])[:2]
        return torus_surface(self.bundle.mu_angle, best[0], best[1],
                             seed=self.seed)

    def _occupancy(self) -> list[dict]:
        return [circle_occupancy(self.bundle.mu_angle[:, d])
                for d in range(self.bundle.mu_angle.shape[1])]

    def circle_occupancy(self):
        return circle_occupancy_panel(self.bundle.mu_angle, self._occupancy())

    def kappa(self):
        floor = float(np.min(self.bundle.kappa)) if self.bundle.kappa.size else None
        return kappa_panel(self.bundle.kappa, kappa_floor=floor)

    def persistence(self):
        if self.topology_cells <= 0:
            return None
        logger.info("  persistent homology on %d cells (roughly cubic — this "
                    "is the slow figure)", self.topology_cells)
        result = persistence_diagram(self.bundle.circular, max_dim=1,
                                     n_points=self.topology_cells,
                                     seed=self.seed)
        if not result.get("available"):
            logger.warning("  topology: %s", result.get("reason"))
            return persistence_panel(result, float("nan"), {})
        threshold = noise_threshold(result["diagrams"])
        betti = betti_numbers(result["diagrams"], threshold)
        logger.info("  Betti numbers of the circular latent: %s "
                    "(cut-off %.3f, %d cells sampled)",
                    betti, threshold, result["n_points"])
        return persistence_panel(result, threshold, betti)

    def dimension(self):
        branches = self.bundle.branches
        return dimension_panel(
            explained_variance(self.bundle.mu_gauss, seed=self.seed),
            {name: intrinsic_dimension(lat, seed=self.seed)
             for name, lat in branches.items()},
            {name: lat.shape[1] for name, lat in branches.items()},
        )
