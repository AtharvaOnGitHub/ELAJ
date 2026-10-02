"""Equalising the measured-gene count, and the three ways it could go wrong.

The m1 arms fragmented into one island per study because the DNS mask is a
study fingerprint. The masker exists to take the easy half of that signal
away: after it, every cell shows the encoder the same number of genes.

Three things have to hold or it does harm rather than good. The counts must
actually come out equal, whatever the study. The study's own DNS genes must
never be "unmasked". And the decoder's view must not change at all, because
the whole design rests on hiding genes from the encoder while still being
required to predict them.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.training.gene_masking import (  # noqa: E402
    INPUT_MASK_KEY, GeneMasker,
)

VOCAB, TARGET = 2_000, 1_200


def batch(panel_sizes: list[int], n_groups: int = 4) -> dict:
    """One cell per entry, each measuring `panel_sizes[i]` of VOCAB genes.

    Panels are nested the way real study panels overlap: the smallest is a
    subset of the next. That is what makes the identity signal survive
    masking, and the tests should run against it rather than against
    independent random panels.
    """
    B = len(panel_sizes)
    is_dns = torch.ones(B, VOCAB, dtype=torch.bool)
    for i, size in enumerate(panel_sizes):
        is_dns[i, :size] = False

    per_group = VOCAB // n_groups
    groups = {}
    for g in range(n_groups):
        lo, hi = g * per_group, (g + 1) * per_group
        groups[f"g{g}"] = is_dns[:, lo:hi].clone()
    return {"is_dns": is_dns, "group_dns_mask": groups}


def measured_after(b: dict) -> torch.Tensor:
    """Measured genes per cell, from the INPUT mask."""
    masks = b[INPUT_MASK_KEY]
    return sum((~m).sum(dim=-1) for m in masks.values())


class TestTheCountIsEqualised:
    def test_two_very_different_panels_end_up_the_same_size(self):
        """The point of the class. 13,144 against 15,056 in the real corpus,
        scaled down here."""
        masker = GeneMasker(target_measured=TARGET, seed=1)
        b = masker.apply(batch([1_300, 1_900]))
        kept = measured_after(b).float()
        assert abs(float(kept[0]) - TARGET) < 60
        assert abs(float(kept[1]) - TARGET) < 60
        # And equal to each other, which is what removes the signal.
        assert abs(float(kept[0] - kept[1])) < 90

    def test_without_masking_the_counts_give_the_study_away(self):
        """The test is only meaningful if the thing it guards is real."""
        masker = GeneMasker(target_measured=0)
        b = masker.apply(batch([1_300, 1_900]))
        assert INPUT_MASK_KEY not in b, "a disabled masker must add nothing"
        counts = [int((~m).sum()) for m in
                  (b["is_dns"][0], b["is_dns"][1])]
        assert counts == [1_300, 1_900]

    def test_the_keep_probability_scales_with_the_panel(self):
        masker = GeneMasker(target_measured=TARGET)
        keep = masker.keep_probability(batch([1_300, 1_900])["is_dns"])
        assert float(keep[0]) == pytest.approx(TARGET / 1_300, rel=1e-5)
        assert float(keep[1]) == pytest.approx(TARGET / 1_900, rel=1e-5)


class TestItOnlyEverRemoves:
    def test_a_gene_the_study_lacks_is_never_unmasked(self):
        """Masking can only hide. If the input mask ever cleared a true DNS
        bit, the encoder would be shown a count for a gene that was never
        measured, which is fabricated data."""
        masker = GeneMasker(target_measured=TARGET, seed=3)
        b = masker.apply(batch([1_300, 1_900]))
        for name, original in b["group_dns_mask"].items():
            masked = b[INPUT_MASK_KEY][name]
            assert torch.all(masked[original]), (
                f"group {name}: a study-DNS gene came back as measured")

    def test_the_mask_is_a_superset(self):
        masker = GeneMasker(target_measured=TARGET, seed=4)
        b = masker.apply(batch([1_300, 1_900]))
        for name, original in b["group_dns_mask"].items():
            assert int((~b[INPUT_MASK_KEY][name]).sum()) <= int((~original).sum())


class TestTheDecoderIsUntouched:
    def test_is_dns_and_group_dns_mask_are_unchanged(self):
        """`endo_model._build_gene_queries` caches the decoder layout per
        study from is_dns[0]. If masking reached those tensors, the decoder
        would predict a different gene set per batch and the reconstruction
        target would move under the loss."""
        masker = GeneMasker(target_measured=TARGET, seed=5)
        b = batch([1_300, 1_900])
        before_is_dns = b["is_dns"].clone()
        before_groups = {k: v.clone() for k, v in b["group_dns_mask"].items()}

        masker.apply(b)

        assert torch.equal(b["is_dns"], before_is_dns)
        for k, v in before_groups.items():
            assert torch.equal(b["group_dns_mask"][k], v)


class TestItVariesWhereItShould:
    def test_two_cells_in_one_batch_get_different_masks(self):
        """A mask shared across the batch is still constant within a study,
        which is the thing being removed."""
        masker = GeneMasker(target_measured=TARGET, seed=6)
        b = masker.apply(batch([1_900, 1_900]))
        g0 = b[INPUT_MASK_KEY]["g0"]
        assert not torch.equal(g0[0], g0[1])

    def test_two_steps_draw_different_masks(self):
        """The generator advances, so an epoch does not show the encoder the
        same hidden genes every time it sees a cell."""
        masker = GeneMasker(target_measured=TARGET, seed=7)
        first = masker.apply(batch([1_900]))[INPUT_MASK_KEY]["g0"].clone()
        second = masker.apply(batch([1_900]))[INPUT_MASK_KEY]["g0"]
        assert not torch.equal(first, second)


class TestShortPanels:
    def test_a_panel_below_the_target_keeps_everything_and_warns(self, caplog):
        """Such a study cannot reach the target, so it stays identifiable by
        count. That has to be said out loud, not discovered in a UMAP."""
        masker = GeneMasker(target_measured=TARGET, seed=8)
        with caplog.at_level("WARNING"):
            b = masker.apply(batch([900]))
        assert int(measured_after(b)[0]) == 900
        assert any("cannot reach the target" in r.message for r in caplog.records)

    def test_a_negative_target_is_refused(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            GeneMasker(target_measured=-1)
