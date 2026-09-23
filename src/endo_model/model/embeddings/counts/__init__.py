"""How a raw UMI count is scaled before it is turned into a vector.

ELAJ has always fed `ValueEncoder` the raw count: no library-size
normalisation, no log. Every other model in this space transforms first, and
so does the PCA reference the stage probe scores the encoder against, which
makes "PCA beats the group encoders" a comparison of two things at once.

Two scalers, so the two can be separated:

    RawCounts              the count as it arrives. What every run to date did.
    LogNormalisedCounts    log1p(count / library_size * target_sum). What scVI,
                           Seurat and the PCA reference see.

Raw counts run from 0 to several thousand within one cell. Two consequences
inside the group encoder. The value part of a token is `count * w`, so a
housekeeping gene's token is a thousand times longer than a marker's and DSBN
spends its capacity on that. And a gene's attention score is shifted by
`count * s` for one shared scalar `s`, so at counts in the thousands the
softmax saturates and attention becomes a near-hard pick of the highest-count
gene, the same gene for every cell of a type.
"""

from endo_model.model.embeddings.counts.base import CountScaler
from endo_model.model.embeddings.counts.factory import (
    COUNT_SCALERS, build_count_scaler,
)
from endo_model.model.embeddings.counts.log_normalised import LogNormalisedCounts
from endo_model.model.embeddings.counts.raw import RawCounts

__all__ = [
    "COUNT_SCALERS",
    "CountScaler",
    "LogNormalisedCounts",
    "RawCounts",
    "build_count_scaler",
]
