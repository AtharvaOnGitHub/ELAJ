"""EmbeddingScorer: the same questions ELAJ is asked, put to any embedding.

The measurements are not new. ``label_knn_agreement``, ``cross_group_agreement``
and ``compute_scib_metrics`` are exactly what ``scripts/validate_latent.py``
runs on ELAJ, called with the same k, the same permutation count and the same
seed. This module's whole job is to hold those settings in one place and apply
them to an embedding without knowing where it came from, so that when two rows
of the final table differ the difference is the method and nothing else.

FOUR KINDS OF SCORE, READ DIFFERENTLY.

  kNN agreement      does the latent group the label. High is good for
                     biology, BAD for study and patient identity.
  cross-donor        the same question with same-donor neighbours discarded,
                     plus a permutation p-value. This is the test that
                     separates "learned biology" from "learned who the patient
                     is", and it is the one ELAJ is judged on. Read p, not
                     excess, at low donor counts.
  scIB               the field's standard battery. Batch correction and bio
                     conservation as the scIB benchmark reports them, so the
                     table can be read next to any published comparison.
  disease            control against eutopic endometrium, the one disease
                     contrast in this corpus that is not circular. It is the
                     stopping list's criterion 5, a label the model has
                     never received in any form. See
                     endo_model.evaluation.disease_contrast for the three
                     routes to circularity and what closes each.

The harmonised labels (``cell family``, ``menstrual phase``, ``cell cycle
phase``) are added to the corpus metadata here, once, before any method runs,
so scANVI can be pointed at the same column the scorer will use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix
from endo_model.evaluation.baselines.methods import Embedding

logger = logging.getLogger(__name__)

SEED = 24
HARMONISED_LINEAGE = "cell family"
HARMONISED_PHASE = "menstrual phase"
HARMONISED_CELL_CYCLE = "cell cycle phase"
# The two severity labels, added 5 November 2026. Distinct from
# validate_latent's `severity (study vocabularies)`, which is a UNION whose
# levels are study-prefixed (`stage_0`, `grade_mild`) and therefore readable
# off study identity alone. These two are MERGES onto one clinical scale, and
# both cohorts contribute every level, so study does not determine the label.
# The merge is not an assumption: duempelmann_2026 defines its mild and severe
# groups as rASRM I-II and III-IV, and marecko_2024 records rASRM I-IV, so the
# two are the same scale written two ways.
HARMONISED_SEVERITY = "severity"
HARMONISED_SEVERITY_CASES = "severity, patients only"
# Every harmonised name, for the places that must not request these from the
# Dataset as obs columns: they are built by add_harmonised_labels, not loaded.
HARMONISED_NAMES = (HARMONISED_LINEAGE, HARMONISED_PHASE,
                    HARMONISED_CELL_CYCLE, HARMONISED_SEVERITY,
                    HARMONISED_SEVERITY_CASES)

# Labels scored by plain kNN agreement. Patient identity is included as the
# thing that should NOT score: a latent that groups patients has learned the
# wrong thing, however good its cell-type number looks.
KNN_LABELS = (HARMONISED_LINEAGE, HARMONISED_PHASE, HARMONISED_CELL_CYCLE,
              "AnnotationRefined", "patient_id")
# Labels that are constant within a donor and so need the cross-donor test.
CROSS_DONOR_LABELS = (HARMONISED_LINEAGE, HARMONISED_PHASE,
                      "EndometriosisGrade", "Endometriosis_stage",
                      HARMONISED_SEVERITY, HARMONISED_SEVERITY_CASES)
# Labels the scIB battery is run against, with study as the batch.
SCIB_LABELS = (HARMONISED_LINEAGE, "AnnotationRefined")
MIN_SCIB_CELLS = 100
# Appended to a disease row measured over too few donors to be read as a
# number. cross_group_agreement at six donors spans -0.67 to +0.48 on a
# latent whose true answer is zero, so the p-value is all there is.
FEW_GROUPS = " FEW GROUPS, excess is unreadable, read p"


def required_obs_columns() -> list[str]:
    """Every obs column the dataset must load for the scorer to work.

    The harmonised labels are merged from per-study source columns, so those
    sources have to be requested from the Dataset even though the scorer
    never reads them directly. Computed from the harmonisers rather than
    listed by hand, so a new source column added there is picked up here.
    """
    from endo_model.evaluation.label_harmonisation import (
        CELL_CYCLE_SOURCE_COLUMNS, LINEAGE_SOURCE_COLUMNS, PHASE_SOURCE_COLUMNS,
        SEVERITY_SOURCE_COLUMNS,
    )

    from endo_model.evaluation.disease_contrast import (
        DISEASE_COLUMN, TISSUE_COLUMN,
    )

    direct = [c for c in KNN_LABELS + CROSS_DONOR_LABELS + SCIB_LABELS
              if c not in HARMONISED_NAMES + ("patient_id",)]
    seen: dict[str, None] = {}
    for c in (*LINEAGE_SOURCE_COLUMNS, *PHASE_SOURCE_COLUMNS,
              *CELL_CYCLE_SOURCE_COLUMNS, *SEVERITY_SOURCE_COLUMNS, *direct,
              DISEASE_COLUMN, TISSUE_COLUMN):
        seen.setdefault(c, None)
    return list(seen)


@dataclass
class ScoreRow:
    """One measurement of one embedding. The unit the report is built from."""

    method: str
    test: str
    label: str
    n_cells: int
    value: float
    note: str = ""
    gene_policy: str = ""
    width: int = 0
    fit_seconds: float = 0.0
    supervised_on: str = ""
    fit_on: str = ""

    @property
    def key(self) -> str:
        """The row's identity in the wide table: ``method/gene_policy/fit_on``
        with empty parts dropped, so ``scvi/all`` is transductive and
        ``scvi/all/train`` was fitted on the training donors."""
        return "/".join(p for p in (self.method, self.gene_policy, self.fit_on) if p)

    def as_dict(self) -> dict:
        d = {"method": self.method, "gene_policy": self.gene_policy,
             "test": self.test, "label": self.label, "n_cells": self.n_cells,
             "value": (round(float(self.value), 4)
                       if np.isfinite(self.value) else float("nan")),
             "note": self.note, "width": self.width,
             "fit_seconds": round(self.fit_seconds, 1),
             "supervised_on": self.supervised_on, "fit_on": self.fit_on}
        return d


@dataclass
class ScorerSettings:
    k: int = 30
    k_cross_donor: int = 100
    permutations: int = 200
    seed: int = SEED
    scib_neighbours: int = 15
    # Every method is projected to this many components before the study
    # neighbour search. kNN purity falls as the ambient dimension rises even
    # when the structure is unchanged -- 39.7% excess in 8 dimensions against
    # 28.2% in 60 on synthetic data with one fixed amount of separation -- so
    # an ELAJ latent of 24 would score as better mixed than a 16-dimensional
    # baseline for free. 16 is every baseline's width, and `project` is a
    # no-op for anything already at or below it. None disables the projection
    # and makes the column incomparable across widths.
    #
    # IT REDUCES THE BIAS, IT DOES NOT ERASE IT. Projecting a 32-dimensional
    # latent whose extra 29 dimensions are pure noise down to 3 moves the
    # measured separation from 0.536 to 0.794 against the 3-dimensional
    # original's 0.887: the gap shrinks 0.35 to 0.09, not to 0. PCA cannot
    # recover the informative subspace when the noise carries equal variance.
    # A trained latent is the easier case, since its dimensions are ordered
    # by variance that means something, but the residual is real and a
    # difference of a few hundredths between methods of different width
    # should not be read as a result.
    study_mixing_components: int | None = 16
    # Every other battery is disabled by emptying its label tuple. This one
    # reads `corpus.study_id`, which always exists, so it needs its own
    # switch to be controllable the same way.
    study_mixing: bool = True
    # Cells for the study neighbour search, drawn evenly across studies.
    #
    # WAS 20,000, WHICH REPORTED THE HELD-OUT SET ON 4.8% OF ITSELF. Changed
    # 4 November 2026. The old value gave 20000 // 11 = 1,818 cells per study,
    # 19,998 in total, out of 416,256 held-out cells. Two reasons were given
    # for it and neither holds.
    #
    # The first was balance: a proportional draw has a chance floor of
    # sum(p^2) = 0.1824 on this split, dominated by duempelmann_2026's 31.8%,
    # against 1/11 = 0.091 for an even one. But `excess` is
    # (purity - floor) / (1 - floor), so the floor is already divided out.
    # Balancing protects a quantity that was never exposed.
    #
    # The second was cost, quoted as a quarter-hour at 380,000 cells against a
    # second at 20,000. That is a real cost and it buys a number measured on
    # the whole test split instead of a twentieth of it, which is the only
    # version anyone can be asked to believe.
    #
    # A balanced draw also cannot exceed the smallest study: garcia_alonso_2021_b
    # contributes 3,072 held-out cells, so the ceiling on an even draw is
    # 3,072 x 11 = 33,792. The old cap sat 41% below even that.
    #
    # 0 means every cell, which is now the default.
    study_mixing_max_cells: int = 0
    knn_labels: tuple[str, ...] = KNN_LABELS
    cross_donor_labels: tuple[str, ...] = CROSS_DONOR_LABELS
    scib_labels: tuple[str, ...] = SCIB_LABELS
    # Control against eutopic endometrium, the one disease contrast that is
    # not circular here. Its own switch rather than a label tuple, because
    # the battery derives its column instead of reading one. See
    # endo_model.evaluation.disease_contrast.
    disease_contrast: bool = True
    # Minimum donors in a study before its row is logged without a warning.
    # cross_group_agreement at six donors spans -0.67 to +0.48 on a latent
    # whose true answer is zero.
    disease_min_donors: int = 20


def add_harmonised_labels(corpus: CorpusMatrix) -> None:
    """Merge the per-study label columns into shared ones, in place.

    Idempotent. What ``validate_latent.py`` does inline before section 3, so
    the baseline table scores the same merged labels the model's table does.
    """
    from endo_model.evaluation.label_harmonisation import (
        CELL_CYCLE, CELL_CYCLE_SOURCE_COLUMNS, LINEAGE, LINEAGE_SOURCE_COLUMNS,
        MENSTRUAL_PHASE, PHASE_SOURCE_COLUMNS, SEVERITY, SEVERITY_CASES,
        SEVERITY_SOURCE_COLUMNS, merge_columns,
    )

    for name, harmoniser, columns in (
        (HARMONISED_LINEAGE, LINEAGE, LINEAGE_SOURCE_COLUMNS),
        (HARMONISED_PHASE, MENSTRUAL_PHASE, PHASE_SOURCE_COLUMNS),
        (HARMONISED_CELL_CYCLE, CELL_CYCLE, CELL_CYCLE_SOURCE_COLUMNS),
        (HARMONISED_SEVERITY, SEVERITY, SEVERITY_SOURCE_COLUMNS),
        (HARMONISED_SEVERITY_CASES, SEVERITY_CASES, SEVERITY_SOURCE_COLUMNS),
    ):
        if name in corpus.meta:
            continue
        present = [c for c in columns if c in corpus.meta]
        if not present:
            corpus.meta[name] = np.full(corpus.n_cells, None, dtype=object)
            continue
        corpus.meta[name] = merge_columns(corpus.meta, columns, harmoniser)
        labelled = sum(1 for v in corpus.meta[name] if v is not None)
        logger.info("Harmonised %-18s %7d of %d cells labelled",
                    name, labelled, corpus.n_cells)


class EmbeddingScorer:
    """Applies the validation battery to an embedding over a CorpusMatrix."""

    def __init__(self, settings: ScorerSettings | None = None) -> None:
        self.settings = settings or ScorerSettings()

    def score(self, embedding: Embedding, corpus: CorpusMatrix,
              gene_policy: str = "", supervised_on: str = "") -> list[ScoreRow]:
        """Every score for one embedding. Never raises on a missing label."""
        if embedding.latent.shape[0] != corpus.n_cells:
            raise ValueError(
                f"{embedding.method}: {embedding.latent.shape[0]} embedded rows "
                f"for {corpus.n_cells} cells"
            )
        add_harmonised_labels(corpus)
        stamp = dict(method=embedding.method, gene_policy=gene_policy,
                     width=embedding.width, fit_seconds=embedding.fit_seconds,
                     supervised_on=supervised_on, fit_on=embedding.fit_on)
        rows: list[ScoreRow] = []
        rows += self._knn(embedding.latent, corpus, stamp)
        rows += self._cross_donor(embedding.latent, corpus, stamp)
        rows += self._scib(embedding.latent, corpus, stamp)
        rows += self._study_mixing(embedding.latent, corpus, stamp)
        rows += self._disease(embedding.latent, corpus, stamp)
        return rows

    # ── the four batteries ────────────────────────────────────────────────

    @staticmethod
    def _column(corpus: CorpusMatrix, name: str):
        """A label column by name, or None. ``patient_id`` is a field of the
        corpus rather than an entry in ``meta``, and it is the one label every
        study carries, so it is resolved here rather than special-cased at
        every call site."""
        if name == "patient_id":
            return corpus.patient_id
        return corpus.meta.get(name)

    def _study_mixing(self, latent, corpus, stamp) -> list[ScoreRow]:
        """How separated the studies are in this embedding, overall and each.

        WHY THIS BATTERY EXISTS. Batch correction is the claim and until
        October 2026 no baseline had a study-level number at all. The scored
        table carried `scib-ari`, `scib-nmi` and `scib-silhouette_bio`, which
        are the biological-conservation half of scIB, plus three batch
        metrics that are degenerate on this corpus: `ilisi_knn` reads exactly
        0.0000 and `pcr_batch` exactly 1.0000 for PCA, scVI, scANVI and ELAJ
        alike, and `kbet` sits at 0.000 to 0.025 for all of them. Eleven
        studies of this size and composition saturate them.

        kNN study purity does not saturate: on the same corpus it spreads
        0.629 to 0.977 per study against a 0.091 floor, and 0.714 to 0.834
        between ELAJ arms. It was computed in the training loop and nowhere
        else, so it could never be read against a baseline.

        `study-mixing` is the comparable number: purity rescaled so 0 is
        perfectly mixed and 1 is fully separated, which removes the corpus's
        own share term. LOW IS GOOD. `study-purity` rows carry the raw
        fractions, one per study plus the corpus, because a single figure
        hides whether eleven studies are half mixed or ten are mixed and one
        sits in an island.
        """
        from endo_model.evaluation import study_mixing

        if not self.settings.study_mixing:
            return []
        n_components = self.settings.study_mixing_components
        rows_kept = study_mixing.balanced_rows(
            corpus.study_id, self.settings.study_mixing_max_cells,
            self.settings.seed)
        got = study_mixing.measure(
            latent[rows_kept], corpus.study_id[rows_kept], k=self.settings.k,
            seed=self.settings.seed, n_components=n_components,
        )
        if not got:
            logger.info("   study    too few cells to measure")
            return []

        projected = min(latent.shape[1], n_components or latent.shape[1])
        shared = (f"k={got['k']}; {projected}d; "
                  f"mixed={got['floor']:.3f}; {len(got['per_study'])} studies")
        rows = [
            ScoreRow(test="study-mixing", label="all studies",
                     n_cells=got["n_cells"], value=got["excess"],
                     note=f"high is BAD: 0=perfectly mixed, 1=fully "
                          f"separated; purity={got['purity']:.3f}; {shared}",
                     **stamp),
            ScoreRow(test="study-purity", label="all studies",
                     n_cells=got["n_cells"], value=got["purity"],
                     note=f"high is BAD: own-study share of {got['k']} "
                          f"neighbours; {shared}", **stamp),
        ]
        for row in got["per_study"]:
            name = corpus.study_names.get(row["study_id"], str(row["study_id"]))
            rows.append(ScoreRow(
                test="study-purity", label=name, n_cells=row["n_cells"],
                value=row["purity"],
                note=f"high is BAD: mixed={row['floor']:.3f}; "
                     f"excess={row['excess']:.1%}; {shared}", **stamp))
        logger.info("   study    %-20s n=%7d separated=%5.1f%% (purity %.3f "
                    "of floor %.3f, %dd)", "all studies", got["n_cells"],
                    100.0 * got["excess"], got["purity"], got["floor"],
                    projected)
        return rows

    def _knn(self, latent, corpus, stamp) -> list[ScoreRow]:
        from endo_model.evaluation.latent_validation import label_knn_agreement

        rows = []
        for col in self.settings.knn_labels:
            values = self._column(corpus, col)
            if values is None:
                continue
            res = label_knn_agreement(latent, values,
                                      k=self.settings.k, seed=self.settings.seed)
            if res["n_cells"] < self.settings.k + 1:
                continue
            note = (f"high is BAD: individual identity; {res['n_labels']} donors; "
                    f"chance={res['chance']:.3f}" if col == "patient_id"
                    else f"{res['n_labels']} labels; chance={res['chance']:.3f}")
            rows.append(ScoreRow(test="knn-agreement", label=col,
                                 n_cells=res["n_cells"], value=res["excess"],
                                 note=note, **stamp))
            logger.info("   knn      %-20s n=%7d excess=%6.3f", col,
                        res["n_cells"], res["excess"])
        return rows

    def _cross_donor(self, latent, corpus, stamp) -> list[ScoreRow]:
        from endo_model.evaluation.latent_validation import cross_group_agreement

        rows = []
        for col in self.settings.cross_donor_labels:
            values = self._column(corpus, col)
            if values is None:
                continue
            res = cross_group_agreement(
                latent, values, corpus.patient_id,
                k=self.settings.k_cross_donor, seed=self.settings.seed,
                n_permutations=self.settings.permutations,
            )
            if not np.isfinite(res.get("agreement", float("nan"))):
                continue
            p = res.get("p_value", float("nan"))
            floor = res.get("p_floor", float("nan"))
            note = (f"p={p:.3f}; floor={floor:.3f}; "
                    f"null95={res.get('null_p95', float('nan')):.2f}; "
                    f"{res['neighbours_kept']:.0f} kept; "
                    f"{res.get('n_groups', 0)} donors")
            rows.append(ScoreRow(test="cross-donor", label=col,
                                 n_cells=res["n_cells"], value=res["excess"],
                                 note=note, **stamp))
            rows.append(ScoreRow(test="cross-donor-p", label=col,
                                 n_cells=res["n_cells"], value=p,
                                 note=f"floor={floor:.3f}", **stamp))
            logger.info("   x-donor  %-20s n=%7d excess=%6.3f p=%.3f (%d donors)",
                        col, res["n_cells"], res["excess"], p,
                        res.get("n_groups", 0))
        return rows

    def _disease(self, latent, corpus, stamp) -> list[ScoreRow]:
        """Control against eutopic endometrium: the non-circular disease test.

        WHY IT IS NOT CIRCULAR. `disease_status` reaches neither encoder nor
        decoder in any arm (`d_disease` 0 throughout), so this is a label the
        model has never seen. The three routes by which it could still leak
        are closed by restriction rather than by hope, and
        `disease_contrast`'s docstring records which closes which.

        THREE ROWS, IN DESCENDING STRICTNESS AND ASCENDING NOISE.

          per study      same-donor neighbours discarded, inside ONE study.
                         Platform is fixed and DSBN's study parameters are
                         fixed, so neither can carry the label. This is the
                         primary result. Read the donor count: a study with
                         six donors cannot support a claim.
          pooled, donor  all qualifying studies at once, so 168 donors
                         instead of 6 to 60. Better powered, but the two
                         sides sit at different rates in each study, so a
                         latent that merely recognised the study would score
                         a little above chance here.
          pooled, study  neighbours from the SAME STUDY discarded as well.
                         The strictest question in the battery: is a control
                         cell placed near control cells FROM OTHER STUDIES?
                         Its p is DEGENERATE and pinned near 1.000: the
                         permutation null shuffles labels between groups, and
                         both labels occur inside every study, so there is
                         only one distinct assignment to draw. Read the
                         excess on this row and read p on the others. The
                         excess is still well behaved here for the same
                         reason the p is not -- the label varies within each
                         group, which is the case the few-groups noise table
                         in `cross_group_agreement` does NOT describe.
        """
        from endo_model.evaluation import disease_contrast as dc

        if not self.settings.disease_contrast:
            return []
        label = dc.build(self._column(corpus, dc.DISEASE_COLUMN),
                         self._column(corpus, dc.TISSUE_COLUMN),
                         corpus.study_id)
        studies = dc.qualifying_studies(label, corpus.study_id)
        if not studies:
            return []

        rows: list[ScoreRow] = []
        for sid in studies:
            inside = corpus.study_id == sid
            name = corpus.study_names.get(sid, str(sid))
            rows += self._one_disease_test(
                latent[inside], label[inside], corpus.patient_id[inside],
                test="disease-xdonor", label_name=name, stamp=stamp,
                note_prefix="within one study; ")

        pooled = np.isin(corpus.study_id, studies)
        rows += self._one_disease_test(
            latent[pooled], label[pooled], corpus.patient_id[pooled],
            test="disease-xdonor", label_name="all studies (pooled)",
            stamp=stamp,
            note_prefix="pooled, so study base rates differ; ")
        rows += self._one_disease_test(
            latent[pooled], label[pooled], corpus.study_id[pooled],
            test="disease-xstudy", label_name="all studies (pooled)",
            stamp=stamp,
            note_prefix="other-STUDY neighbours only; READ THE EXCESS, p is "
                        "degenerate because both labels occur in every "
                        "study; ",
            few_groups_note="")
        return rows

    def _one_disease_test(self, latent, label, groups, test: str,
                          label_name: str, stamp: dict,
                          note_prefix: str = "",
                          few_groups_note: str = FEW_GROUPS) -> list[ScoreRow]:
        """One cross-group run, as an excess row and a p-value row.

        Separated from `_disease` because the same measurement is made three
        times over different cell subsets and different grouping keys, and
        the only thing that varies is what is held constant.

        Args:
            groups:          The key whose neighbours are discarded, donor or
                             study.
            few_groups_note: Appended when there are fewer groups than
                             `disease_min_donors`. Empty for the xstudy row,
                             where few groups is by construction and the p
                             rather than the excess is the unreadable half.
        """
        from endo_model.evaluation.latent_validation import cross_group_agreement

        res = cross_group_agreement(
            latent, label, groups, k=self.settings.k_cross_donor,
            seed=self.settings.seed,
            n_permutations=self.settings.permutations,
        )
        if not np.isfinite(res.get("agreement", float("nan"))):
            return []
        p = res.get("p_value", float("nan"))
        n_groups = int(res.get("n_groups", 0))
        thin = (few_groups_note
                if n_groups < self.settings.disease_min_donors else "")
        note = (f"{note_prefix}p={p:.3f}; "
                f"floor={res.get('p_floor', float('nan')):.3f}; "
                f"{res['neighbours_kept']:.0f} kept; {n_groups} groups{thin}")
        logger.info("   disease  %-24s n=%7d excess=%6.3f p=%.3f (%d groups)%s",
                    f"{test.split('-')[1]} {label_name}", res["n_cells"],
                    res["excess"], p, n_groups, thin)
        return [
            ScoreRow(test=test, label=label_name, n_cells=res["n_cells"],
                     value=res["excess"], note=note, **stamp),
            ScoreRow(test=f"{test}-p", label=label_name,
                     n_cells=res["n_cells"], value=p,
                     note=f"{note_prefix}floor="
                          f"{res.get('p_floor', float('nan')):.3f}", **stamp),
        ]

    def _scib(self, latent, corpus, stamp) -> list[ScoreRow]:
        from endo_model.evaluation.latent_validation import _is_labelled
        from endo_model.evaluation.scib_metrics import compute_scib_metrics

        rows = []
        for col in self.settings.scib_labels:
            vals = self._column(corpus, col)
            if vals is None:
                continue
            ok = np.array([_is_labelled(v) for v in vals])
            if ok.sum() < MIN_SCIB_CELLS:
                continue
            _, bio = np.unique(np.asarray(vals)[ok].astype(str),
                               return_inverse=True)
            try:
                metrics = compute_scib_metrics(
                    latent[ok], corpus.study_id[ok], bio,
                    n_neighbors=self.settings.scib_neighbours,
                )
            except Exception as exc:                          # noqa: BLE001
                logger.warning("   scib %s failed: %s", col, exc)
                continue
            for name, value in metrics.items():
                if not np.isfinite(value):
                    continue
                rows.append(ScoreRow(test=f"scib-{name}", label=col,
                                     n_cells=int(ok.sum()), value=value,
                                     **stamp))
            shown = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                              if np.isfinite(v))
            logger.info("   scib     %-20s n=%7d %s", col, int(ok.sum()), shown)
        return rows
