"""The competitors: one class per method, one interface, one output shape.

Every method here turns a CorpusMatrix into an ``(n_cells, d)`` embedding and
nothing else. Scoring is somebody else's job (scorer.py), which is what makes
the comparison fair: the same cells, the same labels, the same neighbour
counts and the same permutation null are applied to every embedding,
including ELAJ's, by code that does not know which method produced it.

WHAT IS AND IS NOT CONTROLLED. Each method gets the seed and runs on the
device it finds. Hyperparameters are each method's documented defaults, set
once here and recorded in the result, so that "we ran scVI" means the same
thing in every table. A method that needs tuning to be competitive on this
corpus can be tuned in a subclass; the default is what a reader would get
following the method's own tutorial.

Widths are not forced to match. ELAJ's combined latent is 16 numbers (eight
Gaussian, four circles as cos/sin pairs). scVI and PCA are set to 16 so that
the neighbour searches happen in spaces of equal size; Harmony and Scanorama
work in their own PCA spaces by design, and cutting them to 16 would be
scoring a crippled version. The width goes in the result so a reader can see
what was compared.

Dependencies are checked, not assumed. A method whose package is absent
reports that and is skipped; the run does not die on the fourth method after
three hours of the first three.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix

logger = logging.getLogger(__name__)

SEED = 24
# ELAJ's combined latent: 8 Gaussian + 4 circular dimensions as (cos, sin).
ELAJ_LATENT_WIDTH = 16
UNLABELLED = "Unknown"


class MissingDependency(RuntimeError):
    """Raised by ``embed`` when a method's package is not installed."""


@dataclass
class Embedding:
    """What a method produces, with enough provenance to be reported."""

    method: str
    latent: np.ndarray
    fit_seconds: float
    settings: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.latent.shape[1])

    def __post_init__(self) -> None:
        self.latent = np.asarray(self.latent, dtype=np.float32)
        if self.latent.ndim != 2:
            raise ValueError(f"{self.method}: latent must be 2-D, got "
                             f"shape {self.latent.shape}")
        if not np.isfinite(self.latent).all():
            raise ValueError(f"{self.method}: latent contains non-finite values")


def _installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


class BaselineMethod(ABC):
    """Turns a CorpusMatrix into an embedding. Subclasses fill in ``_embed``."""

    def __init__(self, seed: int = SEED) -> None:
        self.seed = int(seed)

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, table-safe identifier: ``scvi``, ``harmony``."""

    @property
    @abstractmethod
    def requires(self) -> tuple[str, ...]:
        """Importable package names this method needs."""

    @abstractmethod
    def _embed(self, corpus: CorpusMatrix) -> np.ndarray:
        """``(n_cells, d)`` float array. Rows in corpus order."""

    def settings(self) -> dict:
        """Hyperparameters worth recording. Override to add the method's own."""
        return {"seed": self.seed}

    def missing(self) -> list[str]:
        return [p for p in self.requires if not _installed(p)]

    def available(self) -> bool:
        return not self.missing()

    def embed(self, corpus: CorpusMatrix) -> Embedding:
        """The template: check, time, validate. Subclasses never override this."""
        absent = self.missing()
        if absent:
            raise MissingDependency(
                f"{self.name} needs {', '.join(absent)}; "
                f"pip install {' '.join(absent)}"
            )
        logger.info("── %s on %d cells x %d genes", self.name,
                    corpus.n_cells, corpus.n_genes)
        started = time.perf_counter()
        latent = self._embed(corpus)
        elapsed = time.perf_counter() - started
        if latent.shape[0] != corpus.n_cells:
            raise RuntimeError(
                f"{self.name} returned {latent.shape[0]} rows for "
                f"{corpus.n_cells} cells — rows would not align with labels"
            )
        logger.info("   %s: width %d, %.0f s", self.name, latent.shape[1], elapsed)
        return Embedding(method=self.name, latent=latent,
                         fit_seconds=elapsed, settings=self.settings())


# ---------------------------------------------------------------------------
# Shared preprocessing. scanpy is the common substrate: every method below
# either is a scanpy step or expects scanpy-style input.
# ---------------------------------------------------------------------------


def to_anndata(corpus: CorpusMatrix, extra_obs: dict[str, np.ndarray] | None = None):
    """A CorpusMatrix as an AnnData with ``study`` in obs, counts in X."""
    import anndata
    import pandas as pd

    obs = {"study": pd.Categorical(corpus.study_id.astype(str))}
    for name, values in (extra_obs or {}).items():
        obs[name] = pd.Categorical(np.asarray(values, dtype=object).astype(str))
    adata = anndata.AnnData(
        X=corpus.counts.copy(),
        obs=pd.DataFrame(obs, index=[str(i) for i in range(corpus.n_cells)]),
    )
    adata.var_names = corpus.gene_names
    return adata


def log_normalise(adata, target_sum: float = 1e4) -> None:
    """In place: counts -> log1p(counts per ``target_sum``). Raw kept in a layer."""
    import scanpy as sc

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def pca(adata, n_comps: int, seed: int) -> np.ndarray:
    """PCA on ``adata.X`` (assumed log-normalised). Returns ``(n, n_comps)``."""
    import scanpy as sc

    n = min(n_comps, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n, random_state=seed)
    return np.asarray(adata.obsm["X_pca"], dtype=np.float32)


# ---------------------------------------------------------------------------
# The methods
# ---------------------------------------------------------------------------


class PCABaseline(BaselineMethod):
    """Log-normalised counts to principal components. No batch correction.

    The floor. Any method that scores worse than this on biology has removed
    signal; any method that scores no better on study mixing has not corrected
    anything. Matched to ELAJ's width so the comparison is between
    representations of equal size.
    """

    def __init__(self, n_components: int = ELAJ_LATENT_WIDTH,
                 seed: int = SEED) -> None:
        super().__init__(seed)
        self.n_components = int(n_components)

    @property
    def name(self) -> str:
        return "pca"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("scanpy", "anndata")

    def settings(self) -> dict:
        return {**super().settings(), "n_components": self.n_components}

    def _embed(self, corpus: CorpusMatrix) -> np.ndarray:
        adata = to_anndata(corpus)
        log_normalise(adata)
        return pca(adata, self.n_components, self.seed)


class HarmonyBaseline(BaselineMethod):
    """PCA, then Harmony corrects the components for study.

    The most widely used non-deep integration and a strong performer on
    tasks of moderate complexity in the scIB benchmark, which makes it the
    honest non-neural comparison rather than a straw one.
    """

    def __init__(self, n_pcs: int = 50, seed: int = SEED) -> None:
        super().__init__(seed)
        self.n_pcs = int(n_pcs)

    @property
    def name(self) -> str:
        return "harmony"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("scanpy", "anndata", "harmonypy")

    def settings(self) -> dict:
        return {**super().settings(), "n_pcs": self.n_pcs}

    def _embed(self, corpus: CorpusMatrix) -> np.ndarray:
        import harmonypy

        adata = to_anndata(corpus)
        log_normalise(adata)
        components = pca(adata, self.n_pcs, self.seed)
        out = harmonypy.run_harmony(components, adata.obs, "study",
                                    random_state=self.seed, verbose=False)
        corrected = np.asarray(out.Z_corr, dtype=np.float32)
        # harmonypy returns (d, n) in some versions and (n, d) in others.
        if corrected.shape[0] != corpus.n_cells:
            corrected = corrected.T
        return corrected


class ScanoramaBaseline(BaselineMethod):
    """Scanorama: mutual-nearest-neighbour panoramas across studies.

    Runs on a per-study list, so the corpus is split by study, integrated,
    and reassembled in the original row order. Getting that order back is
    the part that has to be right; the rest is the library.
    """

    def __init__(self, dimred: int = 100, seed: int = SEED) -> None:
        super().__init__(seed)
        self.dimred = int(dimred)

    @property
    def name(self) -> str:
        return "scanorama"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("scanpy", "anndata", "scanorama")

    def settings(self) -> dict:
        return {**super().settings(), "dimred": self.dimred}

    def _embed(self, corpus: CorpusMatrix) -> np.ndarray:
        import scanorama

        adata = to_anndata(corpus)
        log_normalise(adata)
        studies = np.unique(corpus.study_id)
        rows_by_study = [np.flatnonzero(corpus.study_id == s) for s in studies]
        parts = [adata[rows].copy() for rows in rows_by_study]
        scanorama.integrate_scanpy(parts, dimred=self.dimred, seed=self.seed)
        latent = np.empty((corpus.n_cells, self.dimred), dtype=np.float32)
        for rows, part in zip(rows_by_study, parts):
            latent[rows] = part.obsm["X_scanorama"]
        return latent


class ScVIBaseline(BaselineMethod):
    """scVI: the standard deep generative baseline, unsupervised.

    The apples-to-apples comparison for ELAJ: both are variational
    autoencoders on counts with the study as a batch covariate, both produce
    a latent that is scored without ever being shown a label. Everything ELAJ
    adds on top of this -- pathway grouping, the DNS sentinel, the circular
    branch, per-study normalisation -- is what the difference between the two
    rows measures.

    ``max_epochs=None`` lets scvi-tools pick from the cell count, which is
    what its tutorial does and what a reader reproducing this would get.
    """

    def __init__(self, n_latent: int = ELAJ_LATENT_WIDTH,
                 max_epochs: int | None = None, seed: int = SEED) -> None:
        super().__init__(seed)
        self.n_latent = int(n_latent)
        self.max_epochs = max_epochs

    @property
    def name(self) -> str:
        return "scvi"

    @property
    def requires(self) -> tuple[str, ...]:
        return ("scanpy", "anndata", "scvi")

    def settings(self) -> dict:
        return {**super().settings(), "n_latent": self.n_latent,
                "max_epochs": self.max_epochs}

    def _train(self, corpus: CorpusMatrix):
        """Fit and return the scvi model; ScANVI builds on this."""
        import scvi

        scvi.settings.seed = self.seed
        adata = to_anndata(corpus)
        scvi.model.SCVI.setup_anndata(adata, batch_key="study")
        model = scvi.model.SCVI(adata, n_latent=self.n_latent)
        model.train(max_epochs=self.max_epochs)
        return model, adata

    def _embed(self, corpus: CorpusMatrix) -> np.ndarray:
        model, _ = self._train(corpus)
        return np.asarray(model.get_latent_representation(), dtype=np.float32)


class ScANVIBaseline(ScVIBaseline):
    """scANVI: scVI with a label head, trained semi-supervised.

    Given ``label_column`` from the corpus metadata; cells without a value
    are marked unlabelled, which scANVI handles natively. Reads whichever
    column it is pointed at, so the choice of label is made by the caller and
    recorded in the result, not buried here.

    READ THE SCORE WITH CARE. A score on the label scANVI was trained with is
    not comparable to ELAJ's on the same label: scANVI was told the answer.
    Its scores on OTHER labels are the fair comparison, and the report marks
    the supervised label so the two are not confused.
    """

    def __init__(self, label_column: str = "cell family",
                 n_latent: int = ELAJ_LATENT_WIDTH,
                 max_epochs: int | None = None, seed: int = SEED) -> None:
        super().__init__(n_latent=n_latent, max_epochs=max_epochs, seed=seed)
        self.label_column = label_column

    @property
    def name(self) -> str:
        return "scanvi"

    def settings(self) -> dict:
        return {**super().settings(), "label_column": self.label_column}

    @staticmethod
    def _labels_for(corpus: CorpusMatrix, column: str) -> np.ndarray:
        if column not in corpus.meta:
            raise KeyError(
                f"scANVI label column {column!r} is not in the corpus "
                f"metadata; available: {sorted(corpus.meta)}"
            )
        raw = corpus.meta[column]
        return np.array([UNLABELLED if v is None else str(v) for v in raw],
                        dtype=object)

    def _embed(self, corpus: CorpusMatrix) -> np.ndarray:
        import scvi

        labels = self._labels_for(corpus, self.label_column)
        n_labelled = int((labels != UNLABELLED).sum())
        logger.info("   scANVI labels from %r: %d of %d cells labelled, "
                    "%d classes", self.label_column, n_labelled,
                    corpus.n_cells, len(set(labels)) - 1)
        if n_labelled == 0:
            raise ValueError(f"no cell carries a {self.label_column!r} label")

        scvi.settings.seed = self.seed
        adata = to_anndata(corpus, extra_obs={"label": labels})
        scvi.model.SCVI.setup_anndata(adata, batch_key="study")
        base = scvi.model.SCVI(adata, n_latent=self.n_latent)
        base.train(max_epochs=self.max_epochs)
        model = scvi.model.SCANVI.from_scvi_model(
            base, labels_key="label", unlabeled_category=UNLABELLED,
        )
        # scANVI's own default is a short fine-tune on top of the trained
        # scVI; it is not a second full training.
        model.train(max_epochs=20, n_samples_per_label=100)
        return np.asarray(model.get_latent_representation(), dtype=np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHODS: dict[str, type[BaselineMethod]] = {
    "pca": PCABaseline,
    "harmony": HarmonyBaseline,
    "scanorama": ScanoramaBaseline,
    "scvi": ScVIBaseline,
    "scanvi": ScANVIBaseline,
}


def build_method(name: str, seed: int = SEED, **kwargs) -> BaselineMethod:
    """``"scvi"`` -> ``ScVIBaseline(seed=..., **kwargs)``, and so on."""
    key = name.strip().lower()
    if key not in METHODS:
        raise ValueError(f"unknown method {name!r}; known: {sorted(METHODS)}")
    return METHODS[key](seed=seed, **kwargs)
