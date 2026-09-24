"""`max_size` is a hard ceiling, not a preference.

The group encoder pads every group in a bucket to its longest member and
attends over the padded length, so attention memory is quadratic in the
largest group and one overflow is paid by every group sharing its bucket.
`max_size` exists for that reason and its docstring has always called it a
hard constraint.

It was not one. `_pick_partner` ranked overflow as the FIRST SORT KEY, so when
every candidate would overflow, `min` still returned one and the merge went
ahead. The real build produced a 503-gene group against `max_size=500`: a
249-gene leftover merged into a 254-gene group because nothing smaller was
left.

The floor and the ceiling can genuinely conflict, and these tests pin which
one gives way. `min_size` bounds the group count and so the parameter count; a
group slightly under it costs one extra small transformer. `max_size` bounds
memory. The ceiling wins.
"""

import pytest

from endo_model.data.grouping.union_cut import merge_undersized  # noqa: E402


def sized(**groups) -> dict[str, set[str]]:
    """Groups of the requested sizes, with globally distinct gene names."""
    out, n = {}, 0
    for name, size in groups.items():
        out[name] = {f"g{i}" for i in range(n, n + size)}
        n += size
    return out


class TestTheCeilingHolds:
    def test_the_real_case_that_produced_503(self):
        """249 + 254 = 503 against a ceiling of 500. The leftover stays
        undersized instead."""
        groups = sized(leftover=249, small=254, mid=300, large=400)
        merged = merge_undersized(groups, min_size=250, max_size=500)
        sizes = sorted(len(g) for g in merged.values())
        assert max(sizes) <= 500, sizes
        assert sizes == [249, 254, 300, 400], (
            "nothing should have merged: every partner overflows"
        )

    def test_a_merge_that_fits_still_happens(self):
        """The ceiling must not block merges that were always fine."""
        groups = sized(tiny=10, small=260, mid=300)
        merged = merge_undersized(groups, min_size=250, max_size=500)
        sizes = sorted(len(g) for g in merged.values())
        assert sizes == [270, 300], sizes

    def test_the_smallest_fitting_partner_is_chosen(self):
        """Ranking among candidates that fit is unchanged: prefer a small one,
        keeping the size distribution tight."""
        groups = sized(tiny=20, small=260, large=470)
        merged = merge_undersized(groups, min_size=250, max_size=500)
        sizes = sorted(len(g) for g in merged.values())
        assert sizes == [280, 470], sizes

    def test_no_ceiling_merges_everything_as_before(self):
        groups = sized(leftover=249, small=254, mid=300)
        merged = merge_undersized(groups, min_size=250, max_size=None)
        sizes = sorted(len(g) for g in merged.values())
        assert sizes == [300, 503], sizes

    def test_several_stuck_groups_do_not_loop(self):
        """A group that cannot merge is parked for good. Merges only grow a
        group, so it can never become mergeable later, and re-picking it would
        spin."""
        groups = sized(a=240, b=245, c=249, d=490)
        merged = merge_undersized(groups, min_size=250, max_size=500)
        sizes = sorted(len(g) for g in merged.values())
        assert max(sizes) <= 500, sizes
        assert sum(sizes) == 240 + 245 + 249 + 490


class TestNothingElseChanged:
    def test_groups_already_over_the_floor_are_left_alone(self):
        groups = sized(a=300, b=400)
        merged = merge_undersized(groups, min_size=250, max_size=500)
        assert sorted(len(g) for g in merged.values()) == [300, 400]

    def test_a_floor_above_the_universe_collapses_to_one_group(self):
        """Documented behaviour: honest outcome rather than an error."""
        groups = sized(a=100, b=120)
        merged = merge_undersized(groups, min_size=10_000, max_size=None)
        assert len(merged) == 1
        assert len(next(iter(merged.values()))) == 220

    def test_genes_are_never_lost_or_duplicated(self):
        groups = sized(a=40, b=260, c=300, d=249)
        every = set().union(*groups.values())
        merged = merge_undersized(groups, min_size=250, max_size=500)
        kept = [g for genes in merged.values() for g in genes]
        assert len(kept) == len(every), "a gene was duplicated across groups"
        assert set(kept) == every

    def test_merged_names_record_every_part(self):
        groups = sized(tiny=10, small=260)
        merged = merge_undersized(groups, min_size=250, max_size=500)
        name = next(iter(merged))
        assert "tiny" in name and "small" in name


class TestSingleGroup:
    def test_one_group_below_the_floor_is_returned_unchanged(self):
        groups = sized(only=100)
        merged = merge_undersized(groups, min_size=250, max_size=500)
        assert sorted(len(g) for g in merged.values()) == [100]

    @pytest.mark.parametrize("max_size", [None, 500])
    def test_empty_input_is_empty_output(self, max_size):
        assert merge_undersized({}, min_size=250, max_size=max_size) == {}
