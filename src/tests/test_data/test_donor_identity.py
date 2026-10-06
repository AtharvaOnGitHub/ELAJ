"""One person, one donor id — and the chain that a flat mapping would miss.

garcia_alonso_2021_b is a strict subset of _a with all fifteen sample ids
shared and zero donor ids shared, because CELLxGENE mints a donor UUID per
dataset. Seven of those fifteen people landed on opposite sides of the
train/test wall, putting 5,026 cells into training and into scoring.

Pure bookkeeping over strings, so this runs anywhere.
"""

import pytest

from endo_model.data.donor_identity import DonorAliases


class TestTheGarciaCase:
    def test_two_ids_sharing_a_sample_become_one_person(self):
        a = DonorAliases()
        a.add("uuid-from-dataset-a", "SAMN15049042")
        a.add("uuid-from-dataset-b", "SAMN15049042")
        assert a.canonical("uuid-from-dataset-a") == a.canonical("uuid-from-dataset-b")

    def test_a_person_split_once_cannot_straddle_the_wall(self):
        """The defect itself: one canonical donor goes to one split, so there
        is no precedence rule to apply afterwards."""
        a = DonorAliases()
        a.add_study(["a1", "a1", "a2"], ["S1", "S1", "S2"])
        a.add_study(["b1", "b2"], ["S1", "S2"])
        split = {a.canonical("a1"): "train", a.canonical("a2"): "test"}
        assert split[a.canonical("b1")] == "train"
        assert split[a.canonical("b2")] == "test"

    def test_the_report_names_the_merge(self):
        a = DonorAliases()
        a.add_study(["a1"], ["S1"])
        a.add_study(["b1"], ["S1"])
        groups = a.aliased()
        assert len(groups) == 1
        assert sorted(next(iter(groups.values()))) == ["a1", "b1"]
        assert "2 donor ids resolve to 1 people" in a.describe()


class TestTransitivity:
    def test_a_chain_through_two_samples_is_one_person(self):
        """A and B share S1; B and C share S2. All three are one donor, and a
        flat pair mapping would resolve only what it saw directly."""
        a = DonorAliases()
        a.add("A", "S1")
        a.add("B", "S1")
        a.add("B", "S2")
        a.add("C", "S2")
        assert a.canonical("A") == a.canonical("B") == a.canonical("C")

    def test_the_canonical_name_is_stable(self):
        """Two orderings of the same facts must give the same canonical id,
        or the split stops being reproducible."""
        one, two = DonorAliases(), DonorAliases()
        one.add("zeta", "S1"); one.add("alpha", "S1")
        two.add("alpha", "S1"); two.add("zeta", "S1")
        assert one.canonical("zeta") == two.canonical("zeta") == "alpha"


class TestItLinksNothingItShouldNot:
    def test_unrelated_donors_stay_apart(self):
        a = DonorAliases()
        a.add_study(["p1", "p2"], ["S1", "S2"])
        assert a.canonical("p1") != a.canonical("p2")
        assert a.aliased() == {}
        assert "no two donor ids share a sample" in a.describe()

    def test_a_study_without_sample_ids_links_nothing(self):
        """Seven of eleven studies may lack the column. Absent evidence must
        keep donors distinct, not merge them."""
        a = DonorAliases()
        a.add_study(["p1", "p2"], None)
        assert a.canonical("p1") != a.canonical("p2")

    @pytest.mark.parametrize("blank", ["", "  ", "nan", "NA", "None"])
    def test_placeholder_sample_ids_link_nothing(self, blank):
        a = DonorAliases()
        a.add("p1", blank)
        a.add("p2", blank)
        assert a.canonical("p1") != a.canonical("p2")


class TestResolveAll:
    def test_it_preserves_order_and_length(self):
        a = DonorAliases()
        a.add_study(["x", "y"], ["S1", "S2"])
        a.add_study(["z"], ["S1"])
        out = a.resolve_all(["x", "z", "y"])
        assert len(out) == 3
        assert out[0] == out[1] != out[2]


class TestAliasingCanBeTurnedOff:
    """`donor_aliasing: false` must reproduce the pre-October-2026 wall.

    Aliasing moved the split, so an arm trained before it cannot be scored
    against the current one: seven of the fifteen aliased people were
    assigned to opposite sides, which puts cells the arm TRAINED on into val
    or test, and `benchmark_baselines.py` has no SplitGuard to refuse them.
    The switch exists so an old checkpoint is scored on its own wall.
    """

    def test_an_empty_aliases_resolves_every_id_to_itself(self):
        """The off path builds a DonorAliases and never calls add_study, so
        nothing downstream needs a branch. That only holds if an untouched
        instance is the identity."""
        aliases = DonorAliases()
        for pid in ("donor_a", "donor_b", "P0001"):
            assert aliases.canonical(pid) == pid
        assert aliases.resolve_all(["x", "y"]) == ["x", "y"]
        assert aliases.aliased() == {}

    def test_off_keeps_two_ids_for_one_person_apart(self):
        """The defect, on purpose: with aliasing off the two ids that share a
        sample stay separate donors, which is what the old wall did."""
        on = DonorAliases()
        on.add_study(["id_a", "id_b"], ["sample_1", "sample_1"])
        assert on.canonical("id_a") == on.canonical("id_b")

        off = DonorAliases()   # add_study never called, as load_corpus does
        assert off.canonical("id_a") != off.canonical("id_b")

    def test_the_config_default_is_on(self):
        from endo_model.configs.data_config import SplitConfig
        assert SplitConfig().donor_aliasing is True
        assert SplitConfig.from_dict({}).donor_aliasing is True
        assert SplitConfig.from_dict({"donor_aliasing": False}).donor_aliasing is False
