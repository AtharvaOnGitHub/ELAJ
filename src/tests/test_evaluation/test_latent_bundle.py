"""The bundle exists to stop a misaligned join, so that is what is tested.

A metadata array paired with embeddings in a different order produces numbers
that are wrong in a way that looks entirely plausible -- every figure renders,
every value is in range, and the conclusions are noise. This project has hit
that once already, which is why `sampler_order` exists in the driver.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.label_harmonisation import (
    LINEAGE, LINEAGE_SOURCE_COLUMNS,
)
from endo_model.evaluation.latent_bundle import LatentBundle

SEED = 24
N = 200


def _bundle(n: int = N, **over) -> LatentBundle:
    rng = np.random.default_rng(SEED)
    kwargs = dict(
        mu_gauss=rng.normal(0, 1, (n, 8)),
        mu_angle=rng.uniform(-np.pi, np.pi, (n, 4)),
        kappa=rng.lognormal(2, 0.5, (n, 4)),
        study_id=rng.integers(0, 3, n),
        split=np.array(["val"] * n, dtype=object),
        meta={"patient_id": np.array([f"P{i % 7}" for i in range(n)],
                                     dtype=object)},
        study_names={0: "duempelmann_2026", 1: "marecko_2024", 2: "tan_2022"},
    )
    kwargs.update(over)
    return LatentBundle(**kwargs)


class TestAlignment:
    def test_a_short_array_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="misaligned"):
            _bundle(kappa=np.zeros((N - 1, 4)))

    def test_a_short_metadata_column_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="patient_id"):
            _bundle(meta={"patient_id": np.array(["P0"] * (N - 3),
                                                 dtype=object)})

    def test_a_consistent_bundle_constructs(self):
        assert len(_bundle()) == N


class TestBranches:
    def test_the_circular_half_is_twice_as_wide_as_the_angles(self):
        b = _bundle()
        assert b.circular.shape == (N, 8)

    def test_angles_become_points_on_the_unit_circle(self):
        b = _bundle()
        c, s = b.circular[:, :4], b.circular[:, 4:]
        assert np.allclose(c ** 2 + s ** 2, 1.0)

    def test_wrapping_does_not_move_a_cell(self):
        """0.01 and 0.01 + 2pi are the same angle and must be the same point.

        A Euclidean neighbour search on raw angles would put them 6.28 apart.
        """
        b = _bundle()
        wrapped = _bundle(mu_angle=b.mu_angle + 2 * np.pi)
        assert np.allclose(b.circular, wrapped.circular)

    def test_combined_is_both_halves_end_to_end(self):
        b = _bundle()
        assert b.branches["combined"].shape[1] == 8 + 8

    def test_all_three_branches_have_one_row_per_cell(self):
        assert all(len(lat) == N for lat in _bundle().branches.values())


class TestBaselineAlignment:
    def test_baseline_labels_are_gathered_by_index_not_sliced(self):
        """The bug this guards: the reservoir keeps a RANDOM subset.

        Slicing the first len(baseline) labels would pair each sampled cell
        with an unrelated cell's label, and nothing downstream would notice.
        """
        rows = np.array([191, 4, 77, 130])
        b = _bundle(baseline=np.zeros((4, 12)), baseline_rows=rows)
        got = b.baseline_meta("patient_id")
        assert list(got) == [f"P{i % 7}" for i in rows]

    def test_no_baseline_reads_as_none_rather_than_raising(self):
        assert _bundle().baseline_meta("patient_id") is None

    def test_an_absent_column_reads_as_none(self):
        b = _bundle(baseline=np.zeros((3, 12)),
                    baseline_rows=np.array([0, 1, 2]))
        assert b.baseline_meta("not_a_column") is None

    def test_the_baseline_is_not_in_branches(self):
        # It covers a subset of rows, so it cannot be indexed alongside them.
        b = _bundle(baseline=np.zeros((4, 12)),
                    baseline_rows=np.array([0, 1, 2, 3]))
        assert "raw counts" not in b.branches


class TestHarmonised:
    def test_two_studies_merge_into_one_column(self):
        meta = {
            "patient_id": np.array(["P0"] * 4, dtype=object),
            "AnnotationMain": np.array(["stromal", "myeloid", None, None],
                                       dtype=object),
            "lineage": np.array([None, None, "Mesenchymal", "Immune"],
                                dtype=object),
        }
        got = _bundle(4, meta=meta).harmonised(LINEAGE, LINEAGE_SOURCE_COLUMNS)
        assert list(got) == ["stromal", "immune", "stromal", "immune"]


class TestNumeric:
    def test_non_numeric_values_become_nan_rather_than_raising(self):
        meta = {"patient_id": np.array(["P0"] * 4, dtype=object),
                "CycleDay": np.array(["14", None, "unknown", 7], dtype=object)}
        got = _bundle(4, meta=meta).numeric("CycleDay")
        assert got[0] == 14.0 and got[3] == 7.0
        assert np.isnan(got[1]) and np.isnan(got[2])

    def test_an_absent_column_gives_an_empty_array(self):
        assert _bundle().numeric("not_a_column").size == 0


class TestSample:
    def test_it_returns_every_row_when_there_are_fewer_than_asked_for(self):
        assert len(_bundle().sample(10_000)) == N

    def test_it_is_reproducible_for_a_seed(self):
        b = _bundle()
        assert np.array_equal(b.sample(50, seed=7), b.sample(50, seed=7))

    def test_different_seeds_give_different_samples(self):
        b = _bundle()
        assert not np.array_equal(b.sample(50, seed=1), b.sample(50, seed=2))

    def test_the_sample_has_no_repeats(self):
        got = _bundle().sample(50, seed=SEED)
        assert len(set(got.tolist())) == 50


class TestLabelCoverage:
    def test_counts_are_reported_per_study_not_pooled(self):
        meta = {"patient_id": np.array(["P0"] * 6, dtype=object),
                "AnnotationMain": np.array(["stromal"] * 6, dtype=object)}
        b = _bundle(6, meta=meta, study_id=np.array([0, 0, 0, 1, 1, 2]))
        got = b.label_coverage(["AnnotationMain"])
        assert got["duempelmann_2026"]["AnnotationMain"] == 3
        assert got["marecko_2024"]["AnnotationMain"] == 2

    def test_placeholders_are_not_counted_as_coverage(self):
        meta = {"patient_id": np.array(["P0"] * 4, dtype=object),
                "Stage": np.array(["I", "nan", None, "unknown"], dtype=object)}
        b = _bundle(4, meta=meta, study_id=np.zeros(4, dtype=int))
        assert b.label_coverage(["Stage"])["duempelmann_2026"]["Stage"] == 1

    def test_a_study_with_no_labels_appears_with_an_empty_row(self):
        # Coverage is the figure that shows which studies supply nothing, so an
        # empty study must be present rather than dropped.
        got = _bundle().label_coverage(["not_a_column"])
        assert set(got) == {"duempelmann_2026", "marecko_2024", "tan_2022"}


class TestStudyLabels:
    def test_indices_become_names(self):
        b = _bundle(3, study_id=np.array([0, 1, 2]),
                    meta={"patient_id": np.array(["P0"] * 3, dtype=object)})
        assert list(b.study_labels()) == ["duempelmann_2026", "marecko_2024",
                                          "tan_2022"]

    def test_an_unnamed_study_falls_back_to_its_index(self):
        b = _bundle(1, study_id=np.array([9]), study_names={},
                    meta={"patient_id": np.array(["P0"], dtype=object)})
        assert list(b.study_labels()) == ["9"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
