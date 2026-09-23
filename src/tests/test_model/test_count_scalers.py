"""Count scaling: what the value encoder reads before it reads anything.

The two claims that matter downstream. Zero stays zero, because a DNS gene is
set to 0.0 before scaling and would otherwise need a special case. And the log
scaler actually removes sequencing depth: two cells with the same composition
but ten times the depth must come out the same, which is the whole reason the
PCA reference sees a cleaner signal than the model does.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.embeddings.counts import (  # noqa: E402
    COUNT_SCALERS, LogNormalisedCounts, RawCounts, build_count_scaler,
)


class TestRawCounts:
    def test_returns_the_counts_unchanged(self):
        counts = torch.tensor([[0.0, 3.0, 1200.0]])
        library = counts.sum(dim=-1)
        assert torch.equal(RawCounts()(counts, library), counts)


class TestLogNormalisedCounts:
    def test_zero_maps_to_zero(self):
        """A DNS gene arrives at 0.0 and must leave at 0.0, so that the DNS
        substitution downstream is the only thing that marks it."""
        counts = torch.tensor([[0.0, 5.0, 0.0, 20.0]])
        out = LogNormalisedCounts()(counts, counts.sum(dim=-1))
        assert out[0, 0].item() == 0.0
        assert out[0, 2].item() == 0.0

    def test_depth_is_removed(self):
        """Two cells, same composition, ten times the depth. Raw counts put
        them far apart; the log-normalised counts put them on top of each
        other. This is the difference the sweep's pca32_raw row measures."""
        shallow = torch.tensor([[1.0, 2.0, 7.0]])
        deep = shallow * 10.0
        scaler = LogNormalisedCounts()
        a = scaler(shallow, shallow.sum(dim=-1))
        b = scaler(deep, deep.sum(dim=-1))
        assert torch.allclose(a, b, atol=1e-6)

    def test_dynamic_range_is_compressed(self):
        """Raw counts inside one cell span three orders of magnitude, which is
        what saturates the attention softmax. After this they span one."""
        counts = torch.tensor([[1.0, 30.0, 3000.0]])
        out = LogNormalisedCounts()(counts, counts.sum(dim=-1))
        raw_ratio = counts.max() / counts.min()
        log_ratio = out.max() / out.min()
        assert raw_ratio > 1000
        assert log_ratio < 10

    def test_a_zero_library_does_not_divide_by_zero(self):
        counts = torch.zeros(1, 4)
        out = LogNormalisedCounts()(counts, counts.sum(dim=-1))
        assert torch.isfinite(out).all()


class TestFactory:
    @pytest.mark.parametrize("kind", COUNT_SCALERS)
    def test_every_advertised_name_builds(self, kind):
        assert build_count_scaler(kind).name == kind

    def test_an_unknown_name_is_refused(self):
        """A typo in a config must stop the run, not quietly pick a default."""
        with pytest.raises(ValueError, match="count scaler"):
            build_count_scaler("lognorm")
