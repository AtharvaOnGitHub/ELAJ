"""build_value_part: name from a config to the ValuePart it means."""

from __future__ import annotations

from endo_model.model.embeddings.values.base import ValuePart
from endo_model.model.embeddings.values.gene_direction import (
    GeneDirectionValue, GeneTableValue,
)
from endo_model.model.embeddings.values.scalar_maps import (
    DEFAULT_HIDDEN, DEFAULT_N_FREQUENCIES, FourierMap, LinearMap, MLPMap,
)
from endo_model.model.embeddings.values.shared_direction import SharedDirectionValue

VALUE_PARTS = ("linear", "fourier", "mlp", "gene_direction", "gene_table")

# Maps whose input range they assume cannot be met by raw counts. A sine of a
# count in the thousands wraps many times between neighbouring genes, which
# adds noise rather than resolution, so the pairing is refused rather than run
# and reported as a failed idea.
_REQUIRES_LOG = ("fourier",)


def build_value_part(
    kind: str,
    d_token: int,
    group_sizes: dict[str, int],
    count_scale: str = "raw",
    n_frequencies: int = DEFAULT_N_FREQUENCIES,
    mlp_hidden: int = DEFAULT_HIDDEN,
) -> ValuePart:
    """Args:
        kind:          One of VALUE_PARTS.
        d_token:       Token width.
        group_sizes:   group name -> gene count. Read only by "gene_table".
        count_scale:   The CountScaler's name, checked against _REQUIRES_LOG.
        n_frequencies: Read only by "fourier".
        mlp_hidden:    Read only by "mlp".

    Raises:
        ValueError: on an unknown name, or on a map paired with a count scale
        it cannot work under.
    """
    if kind not in VALUE_PARTS:
        raise ValueError(f"value part must be one of {VALUE_PARTS}, got {kind!r}")
    if kind in _REQUIRES_LOG and count_scale != "log":
        raise ValueError(
            f"value part {kind!r} needs count_scale='log'; got {count_scale!r}. "
            f"Its input range assumes log-scaled counts."
        )

    if kind == "linear":
        return SharedDirectionValue(LinearMap(d_token))
    if kind == "fourier":
        return SharedDirectionValue(
            FourierMap(d_token, n_frequencies=n_frequencies)
        )
    if kind == "mlp":
        return SharedDirectionValue(MLPMap(d_token, hidden=mlp_hidden))
    if kind == "gene_direction":
        return GeneDirectionValue(d_token)
    return GeneTableValue(group_sizes, d_token)
