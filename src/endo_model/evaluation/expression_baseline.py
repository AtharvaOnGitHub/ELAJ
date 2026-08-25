"""The same measurement, run on the raw counts instead of the latent.

WHY EVERY BAR NEEDS THIS. A label scoring at chance in the latent has two
completely different explanations and the latent alone cannot separate them:

    raw counts   latent    reading
    high         high      the model kept the signal
    high         LOW       the model destroyed it
    low          low       the signal is not in the data -- not a model failure
    low          high      the model concentrated a weak signal

The menstrual-phase negative currently sits in row 2 or row 3 with no way to
tell which, and those are opposite findings. One says the architecture failed;
the other says the corpus cannot answer the question and no model could.

This is also what replaces `age` as a positive control. Age was going to serve
as proof the measurement apparatus can detect a signal known to be present --
until the survey showed no study carries an age column, `_age_vector` returns
all-NaN, and `AgeEncoder` zeroes NaN rows, so the metadata token is an exactly
zero vector for every cell in the corpus. There is no known-present signal to
use. A per-label ceiling from the data itself is a better control anyway,
because it is measured rather than assumed.

WHAT THIS IS NOT. Not a trained baseline and not a competitor model -- it is a
ceiling on what a linear reading of the input can see, which is the reference
class the latent should be judged against. scVI and PCA baselines are separate
and live in evaluation/baseline.py.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DNS_SENTINEL = -1.0


def normalise_counts(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Raw UMI counts -> log1p of counts per `target_sum`, DNS treated as zero.

    The standard single-cell normalisation, and it is needed rather than
    optional: without it the first principal component is library size, so the
    baseline would measure how much RNA each cell had rather than which genes
    it expressed.

    Genes not measured in a cell's study carry DNS_SENTINEL (-1), which must be
    read as "not observed" and not as a negative count. They become 0, matching
    the measured zeros -- an imperfect reading, but the alternative of dropping
    genes per study leaves no shared feature space to run PCA in.
    """
    x = np.asarray(counts, dtype=np.float32)
    x = np.where(x == DNS_SENTINEL, 0.0, x)
    library = x.sum(axis=1, keepdims=True)
    library[library <= 0] = 1.0
    return np.log1p(x * (target_sum / library))


def counts_pca(counts: np.ndarray, n_components: int = 12,
               seed: int = 24) -> np.ndarray:
    """Log-normalised counts projected to `n_components` principal directions.

    Matched to the latent's own width by default, so the comparison is between
    two representations of the same size rather than between 12 numbers and
    15,000. A 15,000-dimensional neighbour search would lose to the latent on
    dimensionality alone and the comparison would be meaningless.
    """
    from sklearn.decomposition import PCA

    x = normalise_counts(counts)
    keep = min(n_components, min(x.shape) - 1)
    if keep < 1:
        return np.empty((len(x), 0), dtype=np.float32)
    return PCA(n_components=keep, random_state=seed).fit_transform(x)


class ReservoirSampler:
    """A fixed-size uniform sample from a stream of unknown length.

    The full corpus is ~800,000 cells by ~15,000 genes, which is 48 GB in
    float32 and cannot be held to run a PCA over. Every cell has an equal
    chance of being in the sample regardless of when it arrives, so the sample
    is not biased toward the studies the loader happens to visit first --
    which taking the first N cells would be, and severely: the sampler groups
    batches BY STUDY, so the first N cells are one or two studies entirely.

    Standard reservoir sampling (Vitter, 1985), extended to arrive in blocks
    because the loader yields batches rather than single cells.
    """

    def __init__(self, capacity: int, seed: int = 24) -> None:
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._rows: list[np.ndarray] = []
        self._meta: list[int] = []
        self.n_seen = 0

    def add(self, block: np.ndarray, tags: np.ndarray | None = None) -> None:
        """Offer a batch of rows to the sample."""
        block = np.asarray(block)
        tags = (np.zeros(len(block), dtype=int) if tags is None
                else np.asarray(tags))
        for row, tag in zip(block, tags):
            self.n_seen += 1
            if len(self._rows) < self.capacity:
                self._rows.append(np.asarray(row, dtype=np.float32))
                self._meta.append(int(tag))
                continue
            # After the reservoir fills, the n-th arrival replaces a random
            # existing entry with probability capacity/n. That is what keeps
            # every cell equally likely to survive.
            j = int(self._rng.integers(0, self.n_seen))
            if j < self.capacity:
                self._rows[j] = np.asarray(row, dtype=np.float32)
                self._meta[j] = int(tag)

    @property
    def rows(self) -> np.ndarray:
        if not self._rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(self._rows)

    @property
    def tags(self) -> np.ndarray:
        return np.asarray(self._meta, dtype=int)

    def __len__(self) -> int:
        return len(self._rows)
