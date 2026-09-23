"""Value parts: the rank of the count-to-token map, which is the whole point.

The diagnosis these exist to fix is a rank statement, so the tests are rank
statements. With the original linear map, the set of value vectors a group can
produce as its counts vary spans ONE direction, so an attention head's
count-dependent output is one scalar times a fixed vector. Every replacement
here is tested for spanning more than one, and the gene-specific ones are
tested for the stronger property that two genes at the SAME count contribute
different directions, which is what makes a head able to act like a principal
component.

Rank is measured with `torch.linalg.matrix_rank` on the matrix of value
vectors, which is exactly the quantity the argument is about.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.embeddings.values import (  # noqa: E402
    VALUE_PARTS, FourierMap, GeneDirectionValue, GeneTableValue, LinearMap,
    MLPMap, SharedDirectionValue, TokenInputs, build_value_part,
)

D_TOKEN = 32
N_GENES = 16
GROUP_SIZES = {"alpha": N_GENES, "beta": 8}


def inputs(counts, seed: int = 0, group: str = "alpha") -> TokenInputs:
    """One group's TokenInputs with random identities for its genes."""
    torch.manual_seed(seed)
    n_genes = counts.shape[1]
    identity = torch.randn(counts.shape[0], n_genes, D_TOKEN) * 0.02
    within = torch.arange(n_genes).unsqueeze(0).expand(counts.shape[0], -1)
    return TokenInputs(group_name=group, counts=counts, identity=identity,
                       within_idxs=within)


def value_rank(part, counts) -> int:
    """Rank of the value vectors produced across a spread of counts.

    One row per gene, so the matrix is (n_genes, d_token) and its rank is how
    many independent directions this group's counts can reach.
    """
    with torch.no_grad():
        values = part(inputs(counts))[0]              # (n_genes, d_token)
    return int(torch.linalg.matrix_rank(values.double(), tol=1e-6))


SPREAD = torch.linspace(0.0, 9.0, N_GENES).unsqueeze(0)     # log-count range


class TestTheLinearMapIsRankOne:
    def test_value_vectors_span_two_directions_at_most(self):
        """`c * w + b` traces a straight LINE, so the values span the
        direction w plus the offset b: rank two, and the count contributes
        only one of those. This is the collapse the sweep is trying to break.
        """
        part = SharedDirectionValue(LinearMap(D_TOKEN))
        assert value_rank(part, SPREAD) <= 2

    def test_removing_the_offset_leaves_rank_one(self):
        """With the bias subtracted, what is left is c times one fixed vector
        for every gene, which is rank one exactly."""
        part = SharedDirectionValue(LinearMap(D_TOKEN))
        with torch.no_grad():
            values = part(inputs(SPREAD))[0] - part.scalar_map.linear.bias
        assert int(torch.linalg.matrix_rank(values.double(), tol=1e-6)) == 1

    def test_two_genes_at_the_same_count_are_identical(self):
        """No gene identity reaches the value, which is the second half of the
        problem: the map cannot weight genes differently."""
        part = SharedDirectionValue(LinearMap(D_TOKEN))
        counts = torch.full((1, 4), 3.0)
        with torch.no_grad():
            values = part(inputs(counts))[0]
        assert torch.allclose(values[0], values[3], atol=1e-6)


class TestNonlinearScalarMaps:
    @pytest.mark.parametrize("build", [
        lambda: FourierMap(D_TOKEN, n_frequencies=8),
        lambda: MLPMap(D_TOKEN, hidden=64),
    ])
    def test_value_vectors_span_many_directions(self, build):
        """A curve, not a line. Rank well above the linear map's two is the
        property that stops the head's count term collapsing to a scalar."""
        part = SharedDirectionValue(build())
        assert value_rank(part, SPREAD) > 4

    @pytest.mark.parametrize("build", [
        lambda: FourierMap(D_TOKEN, n_frequencies=8),
        lambda: MLPMap(D_TOKEN, hidden=64),
    ])
    def test_two_genes_at_the_same_count_are_still_identical(self, build):
        """These fix the rank but not the gene-specificity. Stated as a test
        because it is the reason the gene_direction arms exist as well."""
        part = SharedDirectionValue(build())
        counts = torch.full((1, 4), 3.0)
        with torch.no_grad():
            values = part(inputs(counts))[0]
        assert torch.allclose(values[0], values[3], atol=1e-6)

    def test_the_fourier_map_is_continuous(self):
        """Nearby counts give nearby vectors. The reason a binned map is not
        offered: the decoder is asked to reconstruct the count, and a bin edge
        throws away the resolution needed to do it."""
        part = FourierMap(D_TOKEN, n_frequencies=8)
        counts = torch.tensor([[3.0, 3.001]])
        with torch.no_grad():
            values = part(counts)[0]
        assert torch.allclose(values[0], values[1], atol=1e-2)


class TestGeneSpecificDirections:
    @pytest.mark.parametrize("build", [
        lambda: GeneDirectionValue(D_TOKEN),
        lambda: GeneTableValue(GROUP_SIZES, D_TOKEN),
    ])
    def test_two_genes_at_the_same_count_differ(self, build):
        """The property the shared maps cannot have, and the one that makes a
        weighted sum over genes a per-gene linear projection of the counts,
        which is the form a principal component takes."""
        part = build()
        counts = torch.full((1, N_GENES), 3.0)
        with torch.no_grad():
            values = part(inputs(counts))[0]
        assert not torch.allclose(values[0], values[3], atol=1e-6)

    @pytest.mark.parametrize("build", [
        lambda: GeneDirectionValue(D_TOKEN),
        lambda: GeneTableValue(GROUP_SIZES, D_TOKEN),
    ])
    def test_a_constant_count_still_spans_many_directions(self, build):
        """The sharp version of the claim. Hold the count FIXED so nothing the
        scalar maps do could help, and the values still span many directions,
        because the direction is the gene's own."""
        part = build()
        counts = torch.full((1, N_GENES), 3.0)
        assert value_rank(part, counts) > 4

    def test_the_factorised_direction_is_one_small_matrix(self):
        """1,024 parameters at d_token 32, which is the version worth
        training. The free table is 26% of the model and is a ceiling."""
        part = GeneDirectionValue(D_TOKEN)
        assert sum(p.numel() for p in part.parameters()) == D_TOKEN * D_TOKEN

    def test_zero_count_gives_zero_value(self):
        """No bias on these, so an undetected gene contributes nothing beyond
        its identity."""
        part = GeneDirectionValue(D_TOKEN)
        counts = torch.zeros(1, N_GENES)
        with torch.no_grad():
            values = part(inputs(counts))
        assert torch.allclose(values, torch.zeros_like(values))


class TestFactory:
    @pytest.mark.parametrize("kind", VALUE_PARTS)
    def test_every_advertised_name_builds(self, kind):
        scale = "log" if kind == "fourier" else "raw"
        part = build_value_part(kind, D_TOKEN, GROUP_SIZES, count_scale=scale)
        out = part(inputs(torch.rand(2, N_GENES) * 5))
        assert out.shape == (2, N_GENES, D_TOKEN)

    def test_an_unknown_name_is_refused(self):
        with pytest.raises(ValueError, match="value part"):
            build_value_part("binned", D_TOKEN, GROUP_SIZES)

    def test_fourier_on_raw_counts_is_refused(self):
        """A sine of a count in the thousands wraps many times between
        neighbouring genes. Refusing the pairing keeps a meaningless arm from
        being run and reported as a failed idea."""
        with pytest.raises(ValueError, match="count_scale='log'"):
            build_value_part("fourier", D_TOKEN, GROUP_SIZES, count_scale="raw")
