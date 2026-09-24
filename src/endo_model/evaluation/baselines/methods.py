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

TRANSDUCTIVE AND INDUCTIVE. ``embed(corpus)`` fits a method on the cells it
is then scored on. That is how integration benchmarks are normally run, and
it is not what ELAJ is asked to do: ELAJ is trained on one set of donors and
scored on donors it never saw. ``embed(corpus, fit_corpus=train)`` asks the
same of a baseline: fit on ``fit_corpus``, embed ``corpus``. PCA, scVI and
scANVI can do that; Harmony and Scanorama have no out-of-sample map and
raise ``TransductiveOnly`` so the report can say why the row is absent. The
``fit_on`` field of the result records which was done, and the two are never
mixed in one table row.
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


class TransductiveOnly(RuntimeError):
    """Raised by ``embed`` when a method is given a fit corpus it cannot use.

    Harmony corrects the cells it is given and Scanorama stitches them; neither
    defines an embedding for a cell it did not see. Asking them to is not an
    error in the caller, it is a fact about the method, and the report should
    say so rather than silently scoring the transductive result in an
    inductive table.
    """


# What ``Embedding.fit_on`` holds when a method was fitted on the cells it
# embedded, which is the default and what every integration benchmark does.
FIT_ON_SCORED = ""


@dataclass
class Embedding:
    """What a method produces, with enough provenance to be reported.

    ``fit_on`` names the split the method was fitted on when that is not the
    scored cells themselves: ``"train"`` for a baseline fitted on the training
    donors and applied to the held-out ones. Empty means transductive.
    """

    method: str
    latent: np.ndarray
    fit_seconds: float
    settings: dict = field(default_factory=dict)
    fit_on: str = FIT_ON_SCORED

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
        """``(n_cells, d)`` float array. Rows in corpus order. Transductive."""

    def _fit_embed(self, fit: CorpusMatrix, corpus: CorpusMatrix) -> np.ndarray:
        """Fit on ``fit``, embed ``corpus``. Rows in ``corpus`` order.

        The default says the method cannot; a method that can overrides this
        and sets ``inductive = True``.
        """
        raise TransductiveOnly(
            f"{self.name} has no out-of-sample map: it can only embed the "
            f"cells it was fitted on"
        )

    # True for methods that override _fit_embed. Read by the driver to decide
    # whether to build a fit corpus for them at all.
    inductive: bool = False

    def settings(self) -> dict:
        """Hyperparameters worth recording. Override to add the method's own."""
        return {"seed": self.seed}

    def missing(self) -> list[str]:
        return [p for p in self.requires if not _installed(p)]

    def available(self) -> bool:
        return not self.missing()

    def embed(self, corpus: CorpusMatrix,
              fit_corpus: CorpusMatrix | None = None,
              fit_on: str = "train") -> Embedding:
        """The template: check, time, validate. Subclasses never override this.

        With ``fit_corpus`` the method is fitted there and applied to
        ``corpus``; ``fit_on`` is the name recorded for that split.
        """
        absent = self.missing()
        if absent:
            raise MissingDependency(
                f"{self.name} needs {', '.join(absent)}; "
                f"pip install {' '.join(absent)}"
            )
        if fit_corpus is not None and not self.inductive:
            raise TransductiveOnly(
                f"{self.name} has no out-of-sample map: it can only embed the "
                f"cells it was fitted on"
            )
        if fit_corpus is not None and fit_corpus.n_genes != corpus.n_genes:
            raise ValueError(
                f"{self.name}: fit corpus has {fit_corpus.n_genes} genes, "
                f"scored corpus {corpus.n_genes}; apply one gene policy to both"
            )
        if fit_corpus is None:
            logger.info("── %s on %d cells x %d genes", self.name,
                        corpus.n_cells, corpus.n_genes)
        else:
            logger.info("── %s fitted on %d cells (%s), applied to %d cells x "
                        "%d genes", self.name, fit_corpus.n_cells, fit_on,
                        corpus.n_cells, corpus.n_genes)
        started = time.perf_counter()
        latent = (self._embed(corpus) if fit_corpus is None
                  else self._fit_embed(fit_corpus, corpus))
        elapsed = time.perf_counter() - started
        if latent.shape[0] != corpus.n_cells:
            raise RuntimeError(
                f"{self.name} returned {latent.shape[0]} rows for "
                f"{corpus.n_cells} cells — rows would not align with labels"
            )
        logger.info("   %s: width %d, %.0f s", self.name, latent.shape[1], elapsed)
        return Embedding(method=self.name, latent=latent,
                         fit_seconds=elapsed, settings=self.settings(),
                         fit_on=(FIT_ON_SCORED if fit_corpus is None else fit_on))


# ---------------------------------------------------------------------------
# Shared preprocessing. scanpy is the common substrate: every method below
# either is a scanpy step or expects scanpy-style input.
# ---------------------------------------------------------------------------


def to_anndata(corpus: CorpusMatrix, extra_obs: dict[str, np.ndarray] | None = None,
               study_categories: list[str] | None = None):
    """A CorpusMatrix as an AnnData with ``study`` in obs, counts in X.

    ``study_categories`` fixes the category set of ``study``. scvi-tools
    registers batch as a categorical and refuses a query whose categories are
    not the training set's, so a fit corpus and a scored corpus have to be
    built with the union of their studies, in the same order.
    """
    import anndata
    import pandas as pd

    obs = {"study": pd.Categorical(corpus.study_id.astype(str),
                                   categories=study_categories)}
    for name, values in (extra_obs or {}).items():
        obs[name] = pd.Categorical(np.asarray(values, dtype=object).astype(str))
    adata = anndata.AnnData(
        X=corpus.counts.copy(),
        obs=pd.DataFrame(obs, index=[str(i) for i in range(corpus.n_cells)]),
    )
    adata.var_names = corpus.gene_names
    return adata


def shared_study_categories(*corpora: CorpusMatrix) -> list[str]:
    """The union of study ids across corpora, as sorted strings."""
    ids = set()
    for c in corpora:
        ids.update(str(s) for s in np.unique(c.study_id))
    return sorted(ids, key=lambda s: (len(s), s))


def require_seen_studies(fit: CorpusMatrix, corpus: CorpusMatrix, name: str) -> None:
    """Refuse a query study the fit corpus never contained.

    A conditional model learns one batch row per study. With the categories
    shared (which scvi needs), a study absent from the fit corpus still gets
    a row, an untrained one at its random initialisation, and scvi embeds
    the query through it without complaint. The result would be a number in
    the table that means nothing. Every study in the query must have
    training cells; the split guarantees that for this corpus, and this
    guarantees it for any other.
    """
    unseen = sorted(set(np.unique(corpus.study_id)) - set(np.unique(fit.study_id)))
    if unseen:
        raise ValueError(
            f"{name}: query studies {unseen} have no cells in the fit corpus, "
            f"so their batch rows would be untrained"
        )


def log_normalise(adata, target_sum: float = 1e4,
                  keep_counts: bool = False) -> None:
    """In place: counts -> log1p(counts per ``target_sum``).

    ``keep_counts`` copies the raw matrix into ``layers["counts"]`` first.
    It defaults to OFF because nothing reads that layer: PCA, Harmony and
    Scanorama all work from the log-normalised X, and scVI never calls this
    function at all, building its own AnnData of raw counts and registering X
    directly. The copy was 18 GB on the 1.2 M-cell training corpus, paid on
    every baseline, for no reader. Pass True if a future method wants both
    matrices at once.
    """
    import scanpy as sc

    if keep_counts:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


# `sc.pp.pca` densifies the whole matrix to centre it. On the 1.2 M-cell
# training corpus that is 1,195,264 x 15,496 x 4 bytes = 74 GB, on top of two
# sparse copies at 18 GB each, which is what killed job 55184824 inside a
# 180 GB allocation. r8_inductive did the same thing at 784,896 cells and
# peaked at 101.88 GB; the corpus grew 52% when the controls came in and the
# allocation did not.
#
# The exact decomposition is worth keeping when it fits, so the choice is made
# against the job's OWN allocation rather than a number typed here.
DENSE_FRACTION = 0.35
"""How much of the job's memory the densified matrix may take. The rest holds
two sparse copies of the corpus, the query corpus, and the SVD workspace."""

FALLBACK_DENSE_GB = 8.0
"""Used when SLURM_MEM_PER_NODE is unset, i.e. off the cluster."""


def dense_budget_gb() -> float:
    """The largest dense matrix this job should build, from its allocation."""
    import os

    mem_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if not mem_mb:
        return FALLBACK_DENSE_GB
    try:
        return max(FALLBACK_DENSE_GB, DENSE_FRACTION * int(mem_mb) / 1024)
    except ValueError:
        return FALLBACK_DENSE_GB


def pca(adata, n_comps: int, seed: int,
        max_dense_gb: float | None = None) -> np.ndarray:
    """PCA on ``adata.X`` (assumed log-normalised). Returns ``(n, n_comps)``.

    Chunks when the densified matrix would exceed ``max_dense_gb``, which
    swaps the exact decomposition for sklearn's IncrementalPCA: it centres
    and decomposes a chunk at a time and never holds the dense matrix. The
    leading components agree closely with the exact ones, which is what a
    baseline needs, and the alternative on a corpus this size is not an exact
    answer but no answer.

    The path taken is logged, because two runs that took different paths are
    not exactly comparable and the log is the only record of which was which.
    """
    import scanpy as sc

    if max_dense_gb is None:
        max_dense_gb = dense_budget_gb()
    n = min(n_comps, adata.n_obs - 1, adata.n_vars - 1)
    dense_gb = adata.n_obs * adata.n_vars * 4 / 1e9

    if dense_gb <= max_dense_gb:
        logger.info("PCA: exact, %d x %d densifies to %.1f GB (budget %.1f)",
                    adata.n_obs, adata.n_vars, dense_gb, max_dense_gb)
        sc.pp.pca(adata, n_comps=n, random_state=seed)
        return np.asarray(adata.obsm["X_pca"], dtype=np.float32)

    # A chunk holds chunk_size x n_vars floats. Sized to ~1 GB, which keeps
    # the peak far below the allocation while staying well above n_comps
    # (IncrementalPCA needs at least n_comps rows per chunk).
    chunk_size = max(n + 1, int(1e9 / (adata.n_vars * 4)))
    logger.info(
        "PCA: chunked at %d cells, because %d x %d would densify to %.1f GB "
        "(limit %.1f). IncrementalPCA, not the exact decomposition.",
        chunk_size, adata.n_obs, adata.n_vars, dense_gb, max_dense_gb,
    )
    sc.pp.pca(adata, n_comps=n, random_state=seed,
              chunked=True, chunk_size=chunk_size)
    return np.asarray(adata.obsm["X_pca"], dtype=np.float32)


def pca_projection(fitted, query) -> np.ndarray:
    """Project ``query.X`` onto the components ``pca`` fitted on ``fitted``.

    scanpy centres the fit matrix before decomposing it (implicitly, when it
    is sparse), so a query cell has to be centred by the FIT mean, not its
    own: ``(x - m) @ W = x @ W - m @ W``. The second form never densifies
    the query. Both matrices must be log-normalised the same way and share
    the gene axis.
    """
    components = np.asarray(fitted.varm["PCs"], dtype=np.float32)   # (genes, k)
    mean = np.asarray(fitted.X.mean(axis=0), dtype=np.float32).ravel()
    projected = query.X @ components
    projected = np.asarray(projected, dtype=np.float32)
    return projected - (mean @ components)[None, :]


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

    inductive = True

    def _embed(self, corpus: CorpusMatrix) -> np.ndarray:
        adata = to_anndata(corpus)
        log_normalise(adata)
        return pca(adata, self.n_components, self.seed)

    def _fit_embed(self, fit: CorpusMatrix, corpus: CorpusMatrix) -> np.ndarray:
        fitted = to_anndata(fit)
        log_normalise(fitted)
        pca(fitted, self.n_components, self.seed)
        query = to_anndata(corpus)
        log_normalise(query)
        return pca_projection(fitted, query)


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

    inductive = True

    def _train(self, corpus: CorpusMatrix, study_categories=None):
        """Fit and return the scvi model; ScANVI builds on this."""
        import scvi

        scvi.settings.seed = self.seed
        adata = to_anndata(corpus, study_categories=study_categories)
        scvi.model.SCVI.setup_anndata(adata, batch_key="study")
        model = scvi.model.SCVI(adata, n_latent=self.n_latent)
        model.train(max_epochs=self.max_epochs)
        return model, adata

    def _embed(self, corpus: CorpusMatrix) -> np.ndarray:
        model, _ = self._train(corpus)
        return np.asarray(model.get_latent_representation(), dtype=np.float32)

    def _fit_embed(self, fit: CorpusMatrix, corpus: CorpusMatrix) -> np.ndarray:
        """Train on ``fit``; encode ``corpus`` with the trained encoder.

        scvi-tools transfers the training registry to a new AnnData when it
        is handed one, provided the genes match and the batch categories are
        the training set's; ``shared_study_categories`` gives both AnnDatas
        the same category list. That alone would let an unseen study through
        on an untrained row, so ``require_seen_studies`` refuses it first.
        The query cells are never trained on.
        """
        require_seen_studies(fit, corpus, self.name)
        categories = shared_study_categories(fit, corpus)
        model, _ = self._train(fit, study_categories=categories)
        query = to_anndata(corpus, study_categories=categories)
        return np.asarray(model.get_latent_representation(query), dtype=np.float32)


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

    def _train_scanvi(self, corpus: CorpusMatrix, study_categories=None):
        """scVI on ``corpus``, then the scANVI fine-tune on its labels."""
        import scvi

        labels = self._labels_for(corpus, self.label_column)
        n_labelled = int((labels != UNLABELLED).sum())
        logger.info("   scANVI labels from %r: %d of %d cells labelled, "
                    "%d classes", self.label_column, n_labelled,
                    corpus.n_cells, len(set(labels)) - 1)
        if n_labelled == 0:
            raise ValueError(f"no cell carries a {self.label_column!r} label")

        scvi.settings.seed = self.seed
        adata = to_anndata(corpus, extra_obs={"label": labels},
                           study_categories=study_categories)
        scvi.model.SCVI.setup_anndata(adata, batch_key="study")
        base = scvi.model.SCVI(adata, n_latent=self.n_latent)
        base.train(max_epochs=self.max_epochs)
        model = scvi.model.SCANVI.from_scvi_model(
            base, labels_key="label", unlabeled_category=UNLABELLED,
        )
        # scANVI's own default is a short fine-tune on top of the trained
        # scVI; it is not a second full training.
        model.train(max_epochs=20, n_samples_per_label=100)
        return model

    def _embed(self, corpus: CorpusMatrix) -> np.ndarray:
        model = self._train_scanvi(corpus)
        return np.asarray(model.get_latent_representation(), dtype=np.float32)

    def _fit_embed(self, fit: CorpusMatrix, corpus: CorpusMatrix) -> np.ndarray:
        """Train on ``fit`` with its labels; encode ``corpus`` UNLABELLED.

        The query's labels are withheld on purpose. scANVI's encoder does not
        read the label at inference, but the registry transfer needs the
        column to exist, and handing it the true labels would leave a reader
        unable to tell whether they were used. Every query cell is marked
        unlabelled, which is a category the training set carries.
        """
        require_seen_studies(fit, corpus, self.name)
        categories = shared_study_categories(fit, corpus)
        model = self._train_scanvi(fit, study_categories=categories)
        withheld = np.full(corpus.n_cells, UNLABELLED, dtype=object)
        query = to_anndata(corpus, extra_obs={"label": withheld},
                           study_categories=categories)
        return np.asarray(model.get_latent_representation(query), dtype=np.float32)


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
