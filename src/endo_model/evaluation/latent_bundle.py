"""One embedding pass, held in one object, read by every test and figure.

The validation script embeds the corpus once and then asks two dozen questions
of it. Passing seven parallel arrays into every function invites the failure
this project has already had once: a metadata array joined to embeddings in a
different order, which produces numbers that are wrong in a way that looks
entirely plausible.

So the arrays travel together, and the joins that matter -- harmonising labels
across studies, splitting the latent into its two halves, sampling for the
expensive figures -- happen here once rather than at each call site.

No I/O, no model, no matplotlib. `scripts/validate_latent.py` fills it;
`viz/validation_figures.py` reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LatentBundle:
    """Everything one embedding pass produced, aligned row for row.

    Attributes:
        mu_gauss:  (n, d_gauss) the Gaussian branch's means.
        mu_angle:  (n, n_vMF) the circular branch's mean angles, in radians.
        kappa:     (n, n_vMF) per-cell concentration.
        study_id:  (n,) integer study index.
        split:     (n,) which split each cell came from.
        meta:      column name -> (n,) object array of per-cell metadata.
        study_names: study index -> name, for figure legends.
    """

    mu_gauss: np.ndarray
    mu_angle: np.ndarray
    kappa: np.ndarray
    study_id: np.ndarray
    split: np.ndarray
    meta: dict[str, np.ndarray]
    study_names: dict[int, str] = field(default_factory=dict)
    # Raw-count principal components and the rows they came from. Absent when
    # the baseline was not collected; every test that uses it checks first.
    baseline: np.ndarray | None = None
    baseline_rows: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.mu_gauss)
        widths = {
            "mu_angle": len(self.mu_angle), "kappa": len(self.kappa),
            "study_id": len(self.study_id), "split": len(self.split),
        }
        bad = {k: v for k, v in widths.items() if v != n}
        # Loud rather than silent. A length mismatch means every subsequent
        # join pairs each cell with another cell's labels.
        if bad:
            raise ValueError(
                f"LatentBundle rows disagree: mu_gauss has {n}, but {bad} — "
                "the metadata join would be misaligned"
            )
        for name, values in self.meta.items():
            if len(values) != n:
                raise ValueError(
                    f"metadata column {name!r} has {len(values)} rows, "
                    f"expected {n}"
                )

    def __len__(self) -> int:
        return len(self.mu_gauss)

    @property
    def circular(self) -> np.ndarray:
        """The circular means as (cos, sin) pairs.

        An angle cannot go into a Euclidean neighbour search directly -- 0.01
        and 6.27 radians are adjacent on the circle and 6.26 apart as numbers.
        On the unit circle, straight-line distance is monotone in the angle
        between two points, so neighbours here ARE angular neighbours.
        """
        return np.concatenate([np.cos(self.mu_angle), np.sin(self.mu_angle)],
                              axis=-1)

    @property
    def branches(self) -> dict[str, np.ndarray]:
        """The three representations every categorical test runs against.

        Both halves separately and the concatenation the decoder actually
        receives. Which half does the work is the open question about the
        circular branch, and the first validation run tested only the Gaussian
        half before reporting the result as "the latent groups cell type".

        The raw-counts baseline is NOT here: it covers a sample of rows rather
        than all of them, so it cannot be indexed alongside these. See
        `baseline_meta`.
        """
        return {
            "gaussian": self.mu_gauss,
            "circular": self.circular,
            "combined": np.concatenate([self.mu_gauss, self.circular], axis=-1),
        }

    def baseline_meta(self, column: str) -> np.ndarray | None:
        """The metadata for the baseline's sampled rows, aligned to it.

        The reservoir keeps a random subset of cells, so its rows are not the
        first N and its labels have to be gathered by index rather than sliced.
        """
        if self.baseline is None or self.baseline_rows is None:
            return None
        if column not in self.meta:
            return None
        return self.meta[column][self.baseline_rows]

    def harmonised(self, harmoniser, columns) -> np.ndarray:
        """One label column merged from whichever studies supply it."""
        from endo_model.evaluation.label_harmonisation import merge_columns

        return merge_columns(self.meta, columns, harmoniser)

    def numeric(self, column: str) -> np.ndarray:
        """A metadata column as floats, NaN wherever it will not convert."""
        values = self.meta.get(column)
        if values is None:
            return np.array([])
        out = np.full(len(values), np.nan, dtype=float)
        for i, v in enumerate(values):
            try:
                if v is not None:
                    out[i] = float(v)
            except (TypeError, ValueError):
                pass
        return out

    def study_labels(self) -> np.ndarray:
        """Study index as readable names, for colouring a figure by study."""
        return np.array([self.study_names.get(int(s), str(s))
                         for s in self.study_id], dtype=object)

    def sample(self, n: int, seed: int = 24) -> np.ndarray:
        """Row indices for a fixed random subset, or all rows if there are few.

        UMAP on 800,000 points costs most of an hour and renders as a solid
        block of colour; the figures that use it say the sample size and the
        seed on their face so the picture is reproducible.
        """
        if len(self) <= n:
            return np.arange(len(self))
        return np.random.default_rng(seed).choice(len(self), n, replace=False)

    def label_coverage(self, columns) -> dict[str, dict[str, int]]:
        """Cells carrying each label, per study.

        The corpus has no shared metadata schema, so a headline "n = 48,000" is
        usually one study. This is what stops a single-study result being read
        as a corpus-wide one.
        """
        from endo_model.evaluation.latent_validation import _is_labelled

        out: dict[str, dict[str, int]] = {
            name: {} for name in self.study_names.values()
        }
        for column in columns:
            values = self.meta.get(column)
            if values is None:
                continue
            ok = np.array([_is_labelled(v) for v in values])
            if not ok.any():
                continue
            ids, counts = np.unique(self.study_id[ok], return_counts=True)
            for i, c in zip(ids, counts):
                name = self.study_names.get(int(i), str(i))
                out.setdefault(name, {})[column] = int(c)
        return out
